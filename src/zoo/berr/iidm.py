import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Chromatic(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3):
        pass

    def init_weights(self):
        pass

    def mean_constraint(self, kernel):
        pass

    def forward(self, img):

        pass


class HighFrequency(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, kernel_size=5):
        pass

    def _init_gaussian(self):
        pass

    def forward(self, x):
        pass

class iidm(nn.Module):
    def __init__(self, kernel_nums=8, kernel_size=3, Gtheta=[0.6, 0.8]):
        pass

    def forward(self, x):
        pass
