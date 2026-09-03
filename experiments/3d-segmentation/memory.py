"""Capacity sizing via the canonical peak metric.

ALL peak memory in this experiment goes through ``pyxconv.pyxconv.radcompare.memory.peak_memory_mib``
(``torch_peak`` from the repo ``MemoryTracker``, 2-iteration warm-up, SGD lr=0) so the
sizing search and the reported comparison are the SAME number — consistent with the
rad-vs-xconv methodology. ``find_max_batch`` sizes the baseline; ``find_max_ps`` /
``maximize_ps`` size the XConv probing count r (maximize r, dropping batch as needed).
"""
from __future__ import annotations

import gc
from typing import Callable

import torch
from pyxconv.radcompare import peak_memory_mib

# probing-count ladder reaching far past 2D ranges — 3D + small batch can afford a lot
_PS_LADDER = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def peak_gb(build_model: Callable[[], torch.nn.Module],
            make_inputs: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
            batch: int, loss_fn, device: torch.device) -> float | None:
    """Peak memory (GiB) of one warmed-up train step via peak_memory_mib, or None on OOM."""
    model = build_model()
    try:
        x, y = make_inputs(batch)
        return peak_memory_mib(model, x, y, n_iters=2, loss_fn=loss_fn) / 1024.0
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            return None
        raise
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def find_max_batch(build_model, make_inputs, loss_fn, device, budget_gb,
                   candidates=(1, 2, 3, 4, 6, 8, 12, 16), margin: float = 0.9):
    """Largest candidate batch whose peak stays under ``margin*budget_gb``."""
    cap = margin * budget_gb
    best, best_mem = 0, 0.0
    for b in candidates:
        m = peak_gb(build_model, make_inputs, b, loss_fn, device)
        fits = m is not None and m <= cap
        print(f"  [batch probe] b={b:<3} peak={'OOM' if m is None else f'{m:5.2f} GB'}"
              f"  (cap {cap:.2f}) -> {'fits' if fits else 'reject'}")
        if not fits:
            break
        best, best_mem = b, m
    if best == 0:
        raise RuntimeError("Even batch=1 does not fit the memory budget.")
    return best, best_mem


def find_max_ps(build_converted_model, make_inputs, batch, loss_fn, device, budget_gb,
                candidates=_PS_LADDER, margin: float = 0.9):
    """Largest XConv ``ps`` whose peak stays under ``margin*budget_gb`` at ``batch``."""
    cap = margin * budget_gb
    best, best_mem = 0, 0.0
    for ps in candidates:
        m = peak_gb(lambda: build_converted_model(ps), make_inputs, batch, loss_fn, device)
        fits = m is not None and m <= cap
        print(f"  [ps probe] ps={ps:<5} peak={'OOM' if m is None else f'{m:5.2f} GB'}"
              f"  (cap {cap:.2f}) -> {'fits' if fits else 'reject'}")
        if not fits:
            break
        best, best_mem = ps, m
    if best == 0:
        raise RuntimeError("No probing count fits the memory budget at this batch.")
    return best, best_mem


def maximize_ps(build_converted_model, make_inputs, loss_fn, device, budget_gb, max_batch,
                batch_ladder=(16, 12, 8, 6, 4, 3, 2, 1), min_batch: int = 1,
                margin: float = 0.9):
    """Maximize ``ps`` (``r``), dropping batch toward ``min_batch`` only as needed."""
    batches = sorted({b for b in batch_ladder if min_batch <= b <= max_batch} | {min_batch},
                     reverse=True)
    best_ps, best_b, best_mem = 0, 0, 0.0
    for b in batches:
        try:
            ps, m = find_max_ps(build_converted_model, make_inputs, b, loss_fn, device,
                                budget_gb, _PS_LADDER, margin)
        except RuntimeError:
            ps, m = 0, 0.0
        print(f"  [max-r] batch={b:<3} -> best ps={ps:<6} (peak {m:5.2f} GB)")
        if ps > best_ps:
            best_ps, best_b, best_mem = ps, b, m
    if best_ps == 0:
        raise RuntimeError("No (batch, ps) combination fits the memory budget.")
    return best_ps, best_b, best_mem
