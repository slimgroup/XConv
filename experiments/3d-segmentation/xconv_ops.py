"""XConv conversion and conv-layer introspection (network-agnostic).

Works on any ``nn.Module`` with regular ``nn.Conv3d`` (groups=1) — here the
bundle's UNet. ``apply_xconv`` swaps every conv for the in-house ``Xconv3D``
(probed, low-memory weight gradient) and REUSES each conv's weight ``Parameter``,
so a finetuning init is preserved; ``assert_convert_preserved`` checks that.

``pyxconv`` must be the boundary-fixed in-house package, installed editable from
``~/Codes/xconv_pv`` (branch ``ali``) — see the README. That branch fixes the
probe's boundary bias for BOTH 2D and 3D convolutions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import pyxconv  # luqigroup/xconv_pv @ ali (boundary-fixed), installed editable


def conv_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Conv3d-family layers (``Xconv3D`` subclasses ``nn.Conv3d``)."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Conv3d)]


def count_xconv(model: nn.Module) -> int:
    return sum(isinstance(m, pyxconv.Xconv3D) for _, m in model.named_modules())


def conv_report(model: nn.Module) -> dict:
    convs = conv_layers(model)
    return {
        "n_conv_layers": len(convs),
        "n_conv_trainable": sum(m.weight.requires_grad for _, m in convs),
        "n_conv_weight_params": int(sum(m.weight.numel() for _, m in convs)),
        "n_xconv_layers": count_xconv(model),
    }


@torch.no_grad()
def snapshot_conv_weights(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.weight.detach().clone() for n, m in conv_layers(model)}


@torch.no_grad()
def conv_update_deltas(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative L2 change ``||W_after - W_before|| / ||W_before||`` per conv."""
    out = {}
    for n, m in conv_layers(model):
        w0 = before[n]
        out[n] = (m.weight.detach() - w0).norm().item() / (w0.norm().item() + 1e-12)
    return out


def apply_xconv(model: nn.Module, ps: int, xmode: str, target: str) -> nn.Module:
    """Replace every Conv3d (and optionally ReLU) by its XConv counterpart, in place.

    ``target='conv'`` converts only convolutions (controlled comparison);
    ``target='all'`` also swaps ReLU for the memory-saving ``BReLU``. Conversion
    reuses each conv's weight Parameter, so the loaded init is preserved.
    """
    if ps <= 0:
        raise ValueError(f"xconv requires ps > 0, got {ps}")
    pyxconv.convert_net(model, ps=ps, xmode=xmode, mode=target)
    return model


@torch.no_grad()
def assert_convert_preserved(model: nn.Module, before: dict[str, torch.Tensor]) -> float:
    """Raise if conversion changed any conv weight; return the max relative delta.

    Guards the invariant that ``convert_net`` does not destroy the finetuning init.
    """
    deltas = conv_update_deltas(model, before)
    if not deltas:
        return 0.0
    worst = max(deltas.values())
    if worst > 0.0:
        bad = [n for n, d in deltas.items() if d > 0][:5]
        raise RuntimeError(f"convert_net altered conv weights (max rel delta {worst:.3e}); "
                           f"init not preserved. First offenders: {bad}")
    return worst
