from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F

from lightonml import OPU
from .funcs import Xconv2D, Xconv3D, Brelu

__all__ = ['Xconv2D', 'Xconv3D', 'BReLU']

conv2d = Xconv2D.apply
conv3d = Xconv3D.apply
brelu = Brelu.apply


class Xconv2D(torch.nn.modules.conv.Conv2d):
    def __init__(self, *args, ps=8, mode='gaussian', backend=None, **kwargs):
        super(Xconv2D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()
        self.opu = None
        self.backend = backend

    def forward(self, input):
        if self.opu is None and self.backend == 'opu':
            N = reduce(mul, input.shape[1:])
            self.opu = OPU(n_components=self.ps, max_n_features=N*self.in_channels, simulated=True)

        if self.ps > 0:
            return conv2d(input, self.weight, self.ps, self.mode, self.bias, self.stride,
                          self.padding, self.dilation, self.groups, self.opu)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Xconv3D(torch.nn.modules.conv.Conv3d):
    def __init__(self, *args, ps=8, mode='gaussian', **kwargs):
        super(Xconv3D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()

    def forward(self, input):
        if self.ps > 0:
            return conv3d(input, self.weight, self.ps, self.mode, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BReLU(torch.nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(BReLU, self).__init__(*args, **kwargs)

    def forward(self, input):
        return brelu(input, self.inplace)
