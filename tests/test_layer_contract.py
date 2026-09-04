"""The probing layer stands in for torch's convolution.

Whatever the estimator does internally, the layer has to behave like the module it
replaces: same output shape, same parameters, gradients of the right shape for both
the filter and the input, in 2D and in 3D. These are the contract a network relies on
when convert_net swaps the layer in underneath it.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch
import torch.nn as nn

from pyxconv.modules import Xconv2D, Xconv3D

# The estimator maps a filter tap to a shift, so it needs an odd kernel centred by
# its padding.
ARGS_2D = dict(kernel_size=3, padding=1)
ARGS_3D = dict(kernel_size=3, padding=1)


@pytest.mark.parametrize("ps", [4, 16])
def test_2d_output_matches_the_convolution_it_replaces(ps):
    x = torch.randn(2, 3, 16, 16)
    ref = nn.Conv2d(3, 5, **ARGS_2D)
    layer = Xconv2D(3, 5, ps=ps, **ARGS_2D)
    assert layer(x).shape == ref(x).shape


def test_3d_output_matches_the_convolution_it_replaces():
    x = torch.randn(1, 2, 8, 8, 8)
    ref = nn.Conv3d(2, 4, **ARGS_3D)
    layer = Xconv3D(2, 4, ps=4, **ARGS_3D)
    assert layer(x).shape == ref(x).shape


def test_the_parameters_are_the_convolution_s_parameters():
    """A converted layer must load and save like the layer it replaced."""
    ref = nn.Conv2d(3, 5, **ARGS_2D)
    layer = Xconv2D(3, 5, ps=4, **ARGS_2D)
    assert set(layer.state_dict()) == set(ref.state_dict())
    layer.load_state_dict(ref.state_dict())


def test_both_gradients_have_the_shapes_the_optimizer_expects():
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    layer = Xconv2D(3, 5, ps=8, **ARGS_2D)
    layer(x).sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.shape == layer.weight.shape
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_the_estimate_is_random_but_the_seed_pins_it():
    """Two draws differ; the same seed reproduces one exactly."""
    x = torch.randn(2, 3, 16, 16)
    layer = Xconv2D(3, 5, ps=4, **ARGS_2D)

    def grad(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        layer.zero_grad()
        layer(x).sum().backward()
        return layer.weight.grad.clone()

    assert torch.equal(grad(0), grad(0))
    assert not torch.equal(grad(0), grad(1))
