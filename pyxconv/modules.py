import torch
import torch.nn.functional as F

import pyxconv

__all__ = ['Xconv2D', 'Xconv3D', 'BReLU', 'BLRReLU']


_pair = torch.nn.modules.utils._pair
_triple = torch.nn.modules.utils._triple

conv2d = pyxconv.funcs.Xconv2D.apply
conv3d = pyxconv.funcs.Xconv3D.apply
brelu = pyxconv.funcs.Brelu.apply
blrrelu = pyxconv.funcs.BLRrelu.apply


class Xconv2D(torch.nn.modules.conv.Conv2d):
    def __init__(self, *args, ps=8, mode='independent', **kwargs):
        super(Xconv2D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()

    def forward(self, input):
        if self.ps > 0:
            return conv2d(input, self.weight, self.ps, self.mode, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class Xconv3D(torch.nn.modules.conv.Conv3d):
    def __init__(self, *args, ps=8, mode='independent', **kwargs):
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


class BLRReLU(torch.nn.ReLU):
    """
    Low rank ReLU. Computes relu(r'*r*x) where r is a low rank flat matrix
    Unlike RAD work, we propafate forward through the network this low rank approximation
    and comput the true gradient of this approximation. This means we have the true gradient
    of an unbiased estimate of the network rather than an unbiased gradient of the true network.
    """
    def __init__(self, *args, **kwargs):
        self.r = kwargs.pop('r', 32)
        super(BReLU, self).__init__(*args, **kwargs)

    def forward(self, input):
        return blrrelu(input, self.r, self.inplace)
