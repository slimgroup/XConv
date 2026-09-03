"""Average Gradient Error (AGE), paper Eq. (9), shared by every method.

    AGE(theta) = (1/M) sum_b || g(theta) - g^(b)(theta) ||_2^2 ,

with g(theta) the exact full-dataset (mean) gradient and g^(b) the gradient on
minibatch b, both evaluated at the *same* weights theta and restricted to the
convolution weights (the parameters both estimators approximate). For the exact
model g^(b) is the exact minibatch gradient, so AGE measures sampling noise
alone (the reference floor); for XConv/RAD it additionally captures the
estimator's approximation noise.

This is the squared, full-vector form of Eq. (9). (The squeezenet script in
this repo accumulates a *sum of per-layer L2 norms* without squaring; we use the
paper's definition here so RAD and XConv are compared on the same, correct
metric.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["exact_full_gradient", "average_gradient_error"]


def _conv_weight_grads(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    return {n: model.get_parameter(n).grad.detach().clone() for n in names}


def _maybe_half(images, labels, precision: str):
    if precision == "fp16":
        images = images.half()
        if labels.is_floating_point():
            labels = labels.half()
    return images, labels


def exact_full_gradient(
    model: nn.Module,
    loader,
    names: list[str],
    device: str,
    loss_fn=None,
    precision: str = "fp32",
) -> dict[str, torch.Tensor]:
    """g(theta): exact full-dataset mean gradient over ``loader``.

    Each minibatch (mean) loss is scaled by ``b / N`` and accumulated, so the
    summed gradient is the dataset-mean gradient (matches comp_full_gradient.py).
    ``model`` must use exact convolutions and be at the target weights theta.
    ``loss_fn`` defaults to NLLLoss (classification); pass MSELoss for regression.
    ``precision='fp16'`` casts model + inputs to half (the repo's bf16 reference),
    so the AGE reference matches the half-precision estimator it is compared to.
    """
    if loss_fn is None:
        loss_fn = nn.NLLLoss(reduction="mean")
    if precision == "fp16":
        model = model.half()
    model.train()
    model.zero_grad()
    total = len(loader.dataset)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, labels = _maybe_half(images, labels, precision)
        scale = images.size(0) / total
        loss = loss_fn(model(images), labels) * scale
        loss.backward()
    return _conv_weight_grads(model, names)


def average_gradient_error(
    model: nn.Module,
    loader,
    full_grad: dict[str, torch.Tensor],
    names: list[str],
    device: str,
    loss_fn=None,
    precision: str = "fp32",
) -> float:
    """AGE of ``model`` against ``full_grad``, per Eq. (9) (mean over minibatches
    of the squared L2 distance summed over conv weights).

    The per-tensor squared error is accumulated in fp32 even when gradients are
    fp16, so the metric itself is not degraded by half-precision summation.
    """
    if loss_fn is None:
        loss_fn = nn.NLLLoss(reduction="mean")
    if precision == "fp16":
        model = model.half()
    batch_errors = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images, labels = _maybe_half(images, labels, precision)
        model.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        squared = 0.0
        for name in names:
            grad = model.get_parameter(name).grad.detach()
            squared += (full_grad[name].float() - grad.float()).pow(2).sum().item()
        batch_errors.append(squared)
    return float(sum(batch_errors) / len(batch_errors))
