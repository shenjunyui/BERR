import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.core import register

from .iidm import iidm

__all__ = ['BERR', ]


@register
class BERR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self,
                 backbone: nn.Module,
                 encoder,
                 decoder,
                 multi_scale=None,
                 use_iim=True,):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

        if use_iim:
            pass

        else:
            self.iim_block = None
            self.iim_adapter = None

    def forward(self, x, targets=None):
        pass
