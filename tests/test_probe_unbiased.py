"""Unbiasedness of XConv's probed convolution-filter gradient.

The exact filter gradient of a convolution is a shifted correlation of the
layer's input with the incoming gradient,

    dL/dW[co, ci, dx, dy]
        = sum_{b, x, y} dY[b, co, x, y] X[b, ci, x + dx, y + dy],

with (dx, dy) running over the filter taps centred at zero. XConv never
stores X; it stores the probed input X e and replaces the correlation by
the trace estimate

    (1/ps) sum_p (dY e_s)_p (X e)_p,    e_s[x, y] = e[x + dx, y + dy],

which is unbiased because E[e e^T] = I. That identity holds only when the
probing vectors are shifted the way the convolution shifts the image: once
per spatial axis, zero-filled outside the image for a zero-padded
convolution and wrapped for a circular one. Shifting the flattened pixel
axis by a single scalar offset wraps every image row into the next, which
is neither boundary, and biases the filter gradient for any filter wider
than a single tap.

Each test averages the estimate over many independent probe draws and
compares the mean against the autograd gradient of the same convolution.
The Monte-Carlo error of the settings below shrinks as 1/sqrt(ps * trials)
and lands a few percent from the exact gradient, well inside `TOL`; a
boundary mismatch in the shift leaves a bias several times larger than
`TOL` that no amount of averaging removes. Everything runs on the CPU with
small tensors.
"""

from __future__ import annotations

import os

# Probing is exercised on small tensors, so pin the run to the CPU: these
# tests must never touch a GPU nor depend on one being present.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pyxconv.modules import Xconv2D, Xconv3D  # noqa: E402

SEED = 0
TOL = 0.12


def _mean_probed_grad(layer, x: torch.Tensor, grad_out: torch.Tensor,
                      trials: int) -> torch.Tensor:
    """Average the probed filter gradient over independent probe draws.

    Each forward pass draws its own probing seed, so the `trials` gradients
    are independent samples of the same estimator.

    Args:
        layer: An `Xconv2D` or `Xconv3D` module whose weight requires grad.
        x: Layer input; must not require grad, since the input gradient is
            unrelated to the estimator under test.
        grad_out: Incoming gradient, the dY of the module docstring.
        trials: Number of probe draws to average.

    Returns:
        The mean estimate, shaped like `layer.weight`.
    """
    total = torch.zeros_like(layer.weight)
    for _ in range(trials):
        layer.zero_grad(set_to_none=True)
        layer(x).backward(grad_out)
        total += layer.weight.grad
    return total / trials


def _relative_error(estimate: torch.Tensor,
                    exact: torch.Tensor) -> float:
    """Frobenius-norm relative error of an estimate against the exact value."""
    return (estimate - exact).norm().item() / exact.norm().item()


@pytest.mark.parametrize(
    "mode, padding_mode, kernel",
    [
        ("independent", "zeros", 3),
        ("gaussian", "zeros", 3),
        ("orthogonal", "zeros", 3),
        ("independent", "circular", 3),
        # A single-tap filter shifts by zero, so it isolates the estimator
        # from the boundary handling entirely.
        ("independent", "zeros", 1),
    ],
)
def test_probe_gradient_unbiased_2d(mode: str, padding_mode: str,
                                    kernel: int) -> None:
    """The averaged 2D estimate matches the autograd filter gradient."""
    b, ci, co, n, ps, trials = 2, 2, 3, 8, 128, 600
    pad = kernel // 2

    torch.manual_seed(SEED)
    x = torch.randn(b, ci, n, n)
    grad_out = torch.randn(b, co, n, n)
    layer = Xconv2D(ci, co, kernel, ps=ps, mode=mode, padding=pad,
                    bias=False, padding_mode=padding_mode)

    weight = layer.weight.detach().clone().requires_grad_(True)
    if padding_mode == "circular":
        padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
        F.conv2d(padded, weight).backward(grad_out)
    else:
        F.conv2d(x, weight, padding=pad).backward(grad_out)

    estimate = _mean_probed_grad(layer, x, grad_out, trials)
    err = _relative_error(estimate, weight.grad)
    assert err < TOL, (
        f"2D {mode}/{padding_mode} k={kernel}: relative error {err:.4f} "
        f"exceeds {TOL}; the probed filter gradient is biased.")


@pytest.mark.parametrize("mode", ["independent", "gaussian"])
def test_probe_gradient_unbiased_3d(mode: str) -> None:
    """The averaged 3D estimate matches the autograd filter gradient."""
    b, ci, co, n, kernel, ps, trials = 1, 2, 2, 5, 3, 192, 600
    pad = kernel // 2

    torch.manual_seed(SEED)
    x = torch.randn(b, ci, n, n, n)
    grad_out = torch.randn(b, co, n, n, n)
    layer = Xconv3D(ci, co, kernel, ps=ps, mode=mode, padding=pad,
                    bias=False)

    weight = layer.weight.detach().clone().requires_grad_(True)
    F.conv3d(x, weight, padding=pad).backward(grad_out)

    estimate = _mean_probed_grad(layer, x, grad_out, trials)
    err = _relative_error(estimate, weight.grad)
    assert err < TOL, (
        f"3D {mode}: relative error {err:.4f} exceeds {TOL}; the probed "
        f"filter gradient is biased.")


def test_unsupported_padding_mode_raises() -> None:
    """A boundary the probe shift cannot represent is refused, not ignored."""
    layer = Xconv2D(1, 1, 3, ps=8, padding=1, bias=False,
                    padding_mode="reflect")
    with pytest.raises(NotImplementedError):
        layer(torch.randn(1, 1, 8, 8))
