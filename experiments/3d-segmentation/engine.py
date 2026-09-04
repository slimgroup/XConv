"""Training step and finetuning loop (pure: no data/model/bundle coupling).

The loss/optimizer/dataloader are supplied by the caller (here, all from the
MONAI bundle). Peak memory is measured separately by the caller via the canonical
``pyxconv.pyxconv.radcompare.peak_memory_mib`` (``torch_peak``, 2-iteration warm-up), so this loop
only trains and returns the loss curve.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def train_step(model: nn.Module, img: torch.Tensor, lbl: torch.Tensor,
               loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """One forward/backward/update on a batch; returns the scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(img), lbl)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def finetune(model: nn.Module, loader, loss_fn: nn.Module,
             optimizer: torch.optim.Optimizer, max_steps: int,
             device: torch.device, scheduler=None, val_batch=None, val_every: int = 0) -> dict:
    """Finetune for ``max_steps`` batches, returning the training (and optional
    validation) loss curves.

    Peak memory is measured separately via the canonical ``peak_memory_mib``.
    ``scheduler`` (the bundle's StepLR) is stepped per iteration. If ``val_batch``
    (a fixed ``(image, label)`` tuple) and ``val_every`` are given, the loss on it
    is recorded (no grad) every ``val_every`` steps.
    """
    losses: list[float] = []
    val_losses: list[tuple[int, float]] = []
    step = 0
    while step < max_steps:
        for batch in loader:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            losses.append(train_step(model, img, lbl, loss_fn, optimizer))
            if scheduler is not None:
                scheduler.step()
            step += 1
            if val_batch is not None and val_every and (step % val_every == 0 or step == 1):
                model.eval()
                with torch.no_grad():
                    val_losses.append((step, float(loss_fn(model(val_batch[0]), val_batch[1]))))
                model.train()
            if step % 10 == 0 or step == 1:
                print(f"    step {step:>4}/{max_steps}  loss={losses[-1]:.4f}")
            if step >= max_steps:
                break
    return {"losses": losses, "val_losses": val_losses}


@torch.no_grad()
def evaluate_dice(model, val_loader, patch, device) -> float:
    """Mean foreground Dice via sliding-window inference — the bundle's metric."""
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import AsDiscrete
    from monai.data import decollate_batch

    model.eval()
    metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_lbl = AsDiscrete(to_onehot=2)
    for batch in val_loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)
        logits = sliding_window_inference(img, (patch, patch, patch), 4, model, overlap=0.25)
        metric(y_pred=[post_pred(p) for p in decollate_batch(logits)],
               y=[post_lbl(g) for g in decollate_batch(lbl)])
    return float(metric.aggregate().item())
