import torch
import torch.nn.functional as F

from . import funcs

__all__ = ['Xconv2D', 'Xconv3D', 'XconvTranspose2D']


_pair = torch.nn.modules.utils._pair
_triple = torch.nn.modules.utils._triple

conv2d = funcs.Xconv2D.apply
conv_transpose2d = funcs.XconvTranspose2D.apply
conv3d = funcs.Xconv3D.apply
brelu = funcs.Brelu.apply


class Xconv2D(torch.nn.modules.conv.Conv2d):
    def __init__(self, *args, ps=8, mode='independent', **kwargs):
        super(Xconv2D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()

    def forward(self, input):
        if not getattr(self, '_padding_mode_logged', False):
            print('[pyxconv_facies] padding_mode', self.padding_mode)
            self._padding_mode_logged = True
        if self.ps > 0:
            return conv2d(input, self.weight, self.ps, self.mode, self.bias, self.stride,
                          self.padding, self.dilation, self.groups, self.padding_mode)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class XconvTranspose2D(torch.nn.modules.conv.ConvTranspose2d):
    def __init__(self, *args, ps=8, mode='independent', **kwargs):
        super(XconvTranspose2D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()

    def forward(self, input):
        if self.ps > 0:
            return conv_transpose2d(
                input,
                self.weight,
                self.ps,
                self.mode,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
                self.groups,
            )
        return F.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )


class Xconv3D(torch.nn.modules.conv.Conv3d):
    def __init__(self, *args, ps=8, mode='independent', **kwargs):
        super(Xconv3D, self).__init__(*args, **kwargs)
        self.ps = ps
        self.mode = mode.lower()

    def forward(self, input):
        if not getattr(self, '_padding_mode_logged', False):
            print('[pyxconv_facies] padding_mode', self.padding_mode)
            self._padding_mode_logged = True
        if self.ps > 0:
            return conv3d(input, self.weight, self.ps, self.mode, self.bias, self.stride,
                          self.padding, self.dilation, self.groups, self.padding_mode)
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BReLU(torch.nn.ReLU):
    def __init__(self, *args, **kwargs):
        super(BReLU, self).__init__(*args, **kwargs)

    def forward(self, input):
        return brelu(input, self.inplace)
