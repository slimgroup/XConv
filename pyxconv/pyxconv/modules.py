import torch

import pyxconv

__all__ = ['Xconv2D', 'Xconv3D']


_pair = torch.nn.modules.utils._pair
_triple = torch.nn.modules.utils._triple

conv2d = pyxconv.funcs.Xconv2D.apply
conv3d = pyxconv.funcs.Xconv3D.apply


class Xconv2D(torch.nn.modules.conv._ConvNd):
    def __init__(self, chi, cho, k, ps=8, bias=None, stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros'):
        kernel_size = _pair(k)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Xconv2D, self).__init__(chi, cho, kernel_size, stride, padding, dilation,
                                      False, _pair(0), groups, bias, padding_mode)
        self.ps = ps

    def forward(self, input):
        return conv2d(input, self.weight, self.ps, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)


class Xconv3D(torch.nn.modules.conv._ConvNd):
    def __init__(self, chi, cho, k, ps=8, bias=None, stride=1, padding=0, dilation=1,
                 groups=1, padding_mode='zeros'):
        kernel_size = _triple(k)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Xconv3D, self).__init__(chi, cho, kernel_size, stride, padding, dilation,
                                      False, _triple(0), groups, bias, padding_mode)
        self.ps = ps

    def forward(self, input):
        return conv3d(input, self.weight, self.ps, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)
