"""Converting a network leaves it a network.

convert_net is meant to be a drop-in: the caller hands over a model, gets the same
model back with its convolutions probing instead of storing, and nothing else about
the model changes. These check that contract rather than any particular gradient.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn

from pyxconv import convert_net
from pyxconv.modules import Xconv2D
from pyxconv.utils import adaptive_convert_net, update_ps


def _net() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
    )


def test_every_convolution_is_replaced():
    net = _net()
    convert_net(net, ps=4)
    convs = [m for m in net.modules() if isinstance(m, nn.Conv2d)]
    assert convs, "the network should still contain convolutions"
    assert all(isinstance(m, Xconv2D) for m in convs)


def test_the_output_shape_is_unchanged():
    x = torch.randn(2, 3, 16, 16)
    net = _net()
    before = net(x).shape
    convert_net(net, ps=4)
    assert net(x).shape == before


def test_the_probing_size_is_the_one_asked_for():
    net = _net()
    convert_net(net, ps=6)
    assert {m.ps for m in net.modules() if isinstance(m, Xconv2D)} == {6}
    update_ps(net, 12)
    assert {m.ps for m in net.modules() if isinstance(m, Xconv2D)} == {12}


def test_without_probing_the_layer_is_the_exact_convolution():
    """ps = 0 is the escape hatch: the layer must fall back to torch's own result."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    net = _net()
    reference = net(x)
    convert_net(net, ps=0)
    assert torch.allclose(net(x), reference, atol=1e-6)


def test_the_adaptive_conversion_skips_small_activations():
    """A layer is only worth probing when its activation is larger than the probe."""
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),      # 32x32, worth compressing
        nn.AvgPool2d(16),
        nn.Conv2d(8, 8, 3, padding=1),      # 2x2, not worth compressing
    )
    adaptive_convert_net(net, torch.randn(1, 3, 32, 32), ps=64)
    kinds = [isinstance(m, Xconv2D) for m in net.modules() if isinstance(m, nn.Conv2d)]
    assert kinds[0] is True
    assert kinds[1] is False
