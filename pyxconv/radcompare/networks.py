"""The convolutional network shared by RAD and XConv, and the three builders.

The architecture is the four-conv CIFAR-10 network of Oktay et al. (2021)
(``RAD/nn_experiments/models.py::CIFARConvNet``), which is *also* the CIFAR-10
network of the XConv paper (Table 5, App. A.5 -- "obtained from Oktay et al.
(2021)"). Using one architecture for both methods makes the memory/gradient
comparison apples-to-apples.

    conv 5x5: 3 -> 16 -> 32, avg-pool, 32 -> 32 -> 32, avg-pool, fc 2048 -> 10.

A single ``forward`` serves all three methods because ``Xconv2D`` and RAD's
``RandConv2dLayer`` both subclass ``nn.Conv2d`` and are callable as ``conv(x)``
(RAD's extra ``retain``/``skip_rand`` flags default to a fresh draw each call,
which is exactly RAD's training behaviour). ReLU is left exact for every method
(RAD default ``rand_relu=False``; XConv converted with ``mode='conv'``), so the
comparison isolates the convolution-gradient estimators.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyxconv.utils import convert_net

from pyxconv.radcompare import rad_layers

__all__ = [
    "CIFARConvNet",
    "build_exact",
    "build_xconv",
    "build_rad",
    "conv_weight_names",
]


class CIFARConvNet(nn.Module):
    """Shared CIFAR-10 conv net; ``conv_layer`` is a Conv2d-compatible factory.

    ``adaptive_head=False`` is the exact RAD/XConv-paper net (flatten 8x8x32 ->
    fc 2048->10), valid only at 32x32 input. ``adaptive_head=True`` swaps in a
    global-average-pool head (32->10) so the net accepts any input size, for the
    image-dimension sweep. The four conv layers (where XConv/RAD act) are
    identical either way, so the conv-weight gradients being compared match.
    """

    def __init__(self, conv_layer=nn.Conv2d, adaptive_head=False):
        super().__init__()
        self.adaptive_head = adaptive_head
        self.conv1 = conv_layer(3, 16, 5, padding=2)
        self.conv2 = conv_layer(16, 32, 5, padding=2)
        self.conv3 = conv_layer(32, 32, 5, padding=2)
        self.conv4 = conv_layer(32, 32, 5, padding=2)
        self.fc5 = nn.Linear(32 if adaptive_head else 2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.avg_pool2d(x, 2)
        if self.adaptive_head:
            x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


def build_exact(adaptive_head: bool = False) -> CIFARConvNet:
    """Standard backprop baseline (exact convolution gradients)."""
    return CIFARConvNet(nn.Conv2d, adaptive_head=adaptive_head)


def build_xconv(ps: int, xmode: str, adaptive_head: bool = False) -> CIFARConvNet:
    """XConv on the conv layers only.

    Args:
        ps: number of probing vectors ``r``.
        xmode: probing scheme -- ``'independent'`` (Indep.), ``'gaussian'``
            (Multi), or ``'orthogonal'`` (Multi-Ortho).
        adaptive_head: use the variable-input global-pool head.
    """
    model = CIFARConvNet(nn.Conv2d, adaptive_head=adaptive_head)
    # mode='conv' converts nn.Conv2d -> Xconv2D and leaves ReLU/Linear untouched.
    convert_net(model, ps=ps, xmode=xmode, mode="conv")
    return model


def build_rad(
    keep_frac: float,
    sparse: bool,
    full_random: bool = False,
    adaptive_head: bool = False,
) -> CIFARConvNet:
    """RAD on the conv layers only (Oktay et al., 2021).

    Args:
        keep_frac: fraction of the spatial (H*W) dimension retained.
        sparse: ``False`` -> random projection; ``True`` -> sampling.
        full_random: per-(batch, channel) draws (sampling only).
        adaptive_head: use the variable-input global-pool head.
    """
    conv_layer = functools.partial(
        rad_layers.RandConv2dLayer,
        keep_frac=keep_frac,
        sparse=sparse,
        full_random=full_random,
    )
    return CIFARConvNet(conv_layer, adaptive_head=adaptive_head)


def conv_weight_names(model: nn.Module) -> list[str]:
    """Dotted names of the convolution weight parameters (the gradients both
    estimators approximate). Matches across exact/XConv/RAD models because
    Xconv2D and RandConv2dLayer subclass nn.Conv2d."""
    return [
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
