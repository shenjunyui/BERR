# """
# by JunyiShen
# SAFE (Semantic-aware Adaptive Frequency Enhancer)
# """
import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureLapPyramid(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=256):
        super().__init__()
        self.num_high = num_high
        self.channels = channels
        kernel = self._gauss_kernel(kernel_size, channels)
        self.register_buffer("kernel", kernel)

        self.interp_mode = 'bilinear'
        self.align_corners = False

    def _gauss_kernel(self, k, c):
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        x = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, k, k)
        kernel = kernel.repeat(c, 1, 1, 1)
        return kernel

    def conv_gauss(self, x):
        c = x.shape[1]
        pad = self.kernel.shape[-1] // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(x, self.kernel, groups=c)

    def down(self, x):
        B, C, H, W = x.shape
        if H % 2 == 1 or W % 2 == 1:
            H_even = H if H % 2 == 0 else H - 1
            W_even = W if W % 2 == 0 else W - 1
            x = F.interpolate(x, size=(H_even, W_even),
                              mode=self.interp_mode,
                              align_corners=self.align_corners)

        return self.conv_gauss(x)[:, :, ::2, ::2]

    def up(self, x):
        B, C, H, W = x.shape
        up = torch.zeros((B, C, H * 2, W * 2), dtype=x.dtype, device=x.device)
        up[:, :, ::2, ::2] = x * 4
        upsampled = self.conv_gauss(up)

        return upsampled

    def decompose(self, x):
        pyr = []
        cur = x
        original_size = cur.shape[2:]

        for _ in range(self.num_high):
            down = self.down(cur)
            up = self.up(down)

            if up.shape[2:] != cur.shape[2:]:
                up = F.interpolate(up, size=cur.shape[2:],
                                   mode=self.interp_mode,
                                   align_corners=self.align_corners)

            pyr.append(cur - up)  # Laplacian
            cur = down

        pyr.append(cur)  # lowest freq
        return pyr

    def reconstruct(self, pyr):
        img = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.up(img)

            if up.shape[2:] != level.shape[2:]:
                up = F.interpolate(up, size=level.shape[2:],
                                   mode=self.interp_mode,
                                   align_corners=self.align_corners)

            img = up + level
        return img

# class FeatureContextBranch(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.f2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.f1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.softmax = nn.Softmax(dim=2)  # Paddle 的 axis 对应 PyTorch 的 dim
#         self.act = nn.LeakyReLU(0.2)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         attn = self.f2(x).reshape(B, C, H * W)
#         attn = self.sigmoid(attn).reshape(B, C, H, W)
#         x_hat = attn * x
#         return x + self.act(self.f1(x_hat))

class FeatureContextBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.f2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.f1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        attn = torch.sigmoid(self.f2(x))
        x_hat = attn * x
        return x + self.act(self.f1(x_hat))

class FeatureEdgeBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        edge = self.conv(x) - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x + edge

class FeatureDPM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cb = FeatureContextBranch(channels)
        self.eb = FeatureEdgeBranch(channels)

    def forward(self, x):
        return self.cb(x) + self.eb(x)

class FeatureLEF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, 32, kernel_size=1)
        self.scales = [1, 2, 3, 6]
        self.conv_out = nn.Conv2d(32, channels, kernel_size=1)

    def forward(self, x):
        feat = self.conv_in(x)
        B, C, H, W = feat.shape
        parts = torch.chunk(feat, chunks=4, dim=1)
        outs = []
        for f, s in zip(parts, self.scales):
            p = F.adaptive_avg_pool2d(f, (s, s))
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(p)
        return self.conv_out(torch.cat(outs, dim=1))

class FeatureAdaptiveEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dpm = FeatureDPM(channels)
        self.lef = FeatureLEF(channels)

    def forward(self, x):
        return self.dpm(x) + self.lef(x)

class FeaturePENet(nn.Module):
    def __init__(self, channels=256, num_high=2):
        super().__init__()
        self.pyramid = FeatureLapPyramid(num_high, channels=channels)
        self.enhancers = nn.ModuleList(
            [FeatureAdaptiveEnhance(channels) for _ in range(num_high + 1)]
        )

    def forward(self, x):
        pyr = self.pyramid.decompose(x)
        out = []
        for i in range(len(pyr)):
            if i < len(pyr) - 1:
                high_freq = self.enhancers[i].dpm(pyr[i])
                out.append(high_freq)
            else:
                out.append(self.enhancers[i](pyr[i]))

        return self.pyramid.reconstruct(out)
