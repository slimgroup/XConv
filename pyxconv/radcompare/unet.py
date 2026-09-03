"""A standard U-Net for the RAD-vs-XConv comparison.

Ronneberger-style encoder-decoder: double-conv blocks, max-pool down, bilinear
upsample + conv up (so every convolution is an ``nn.Conv2d`` that XConv/RAD can
replace), skip connections, and ``nn.ReLU`` *modules* so ``convert_net(mode=
'all')`` swaps them for BitReLU -- as in the paper's other examples. No
normalization, to keep the conv-weight gradient clean for the AGE comparison.
Accepts any input size divisible by ``2 ** depth``.

This is a representative U-Net (not the paper's SIPS diffusion U-Net): it is
conv-activation-dominated, so XConv's compressed activation beats both exact
convolution and RAD (which reconstructs the full activation in backward) on peak
memory, and BitReLU lets XConv beat exact (the activation is no longer pinned by
a full-precision ReLU).
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyxconv.utils import convert_net

from pyxconv.radcompare import rad_layers

__all__ = ["UNet", "build_unet_exact", "build_unet_xconv", "build_unet_rad"]


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout, conv_layer=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_layer(cin, cout, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv_layer(cout, cout, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))


class UNet(nn.Module):
    """U-Net with ``depth`` levels; ``conv_layer`` is a Conv2d-compatible factory."""

    def __init__(self, in_channels=3, out_channels=3, base=32, depth=4,
                 conv_layer=nn.Conv2d):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth + 1)]  # e.g. 32,64,128,256,512

        self.downs = nn.ModuleList()
        cin = in_channels
        for i in range(depth):
            self.downs.append(_DoubleConv(cin, chs[i], conv_layer))
            cin = chs[i]
        self.bottleneck = _DoubleConv(chs[depth - 1], chs[depth], conv_layer)

        self.up_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_convs.append(conv_layer(chs[i + 1], chs[i], 3, padding=1))
            self.ups.append(_DoubleConv(chs[i] * 2, chs[i], conv_layer))

        self.out_conv = conv_layer(chs[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = up_conv(x)
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.out_conv(x)


def build_unet_exact(**kw) -> UNet:
    """Exact-convolution U-Net baseline."""
    return UNet(conv_layer=nn.Conv2d, **kw)


def build_unet_xconv(ps: int, xmode: str, **kw) -> UNet:
    """XConv U-Net: conv layers -> Xconv2D AND ReLU -> BitReLU (``mode='all'``)."""
    model = UNet(conv_layer=nn.Conv2d, **kw)
    convert_net(model, ps=ps, xmode=xmode, mode="all")
    return model


def build_unet_rad(keep_frac: float, sparse: bool, full_random: bool = False,
                   **kw) -> UNet:
    """RAD U-Net: conv layers -> RandConv2dLayer; ReLU left exact (RAD default)."""
    conv_layer = functools.partial(
        rad_layers.RandConv2dLayer,
        keep_frac=keep_frac,
        sparse=sparse,
        full_random=full_random,
    )
    return UNet(conv_layer=conv_layer, **kw)
