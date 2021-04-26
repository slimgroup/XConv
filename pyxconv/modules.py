import torch

import pyxconv

__all__ = ['Xconv2D', 'Xconv3D', 'Zconv2D', 'Zconv3D']


_pair = torch.nn.modules.utils._pair
_triple = torch.nn.modules.utils._triple

xconv2d = pyxconv.funcs.Xconv2d.apply
xconv3d = pyxconv.funcs.Xconv3d.apply
zconv2d = pyxconv.funcs.Zconv2d.apply
zconv3d = pyxconv.funcs.Zconv3d.apply
brelu = pyxconv.funcs.Brelu.apply

class Xconv(torch.nn.modules.conv._ConvNd):

    def _tt(self, inp):
        raise NotImplementedError

    def __init__(self, chi, cho, k, ps=8, bias=True, stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros'):
        kernel_size = self._tt(k)
        stride = self._tt(stride)
        padding = self._tt(padding)
        dilation = self._tt(dilation)
        super(Xconv, self).__init__(chi, cho, kernel_size, stride, padding, dilation,
                                    False, self._tt(0), groups, bias, padding_mode)
        self.ps = ps


class Xconv2D(Xconv):

    def _tt(self, inp):
        return _pair(inp)

    def forward(self, input):
        return xconv2d(input, self.weight, self.ps, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class Xconv3D(Xconv):

    def _tt(self, inp):
        return _triple(inp)

    def forward(self, input):
        return xconv3d(input, self.weight, self.ps, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class Zconv2D(Xconv):

    def _tt(self, inp):
        return _pair(inp)

    def forward(self, input):
        return zconv2d(input, self.weight, self.ps, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class Zconv3D(Xconv):

    def _tt(self, inp):
        return _triple(inp)

    def forward(self, input):
        return zconv3d(input, self.weight, self.ps, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)


class BReLU(torch.nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(BReLU, self).__init__(*args, **kwargs)

    def forward(self, input):
        return brelu(input, self.inplace)
