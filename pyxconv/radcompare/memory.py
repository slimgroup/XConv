"""Peak-memory measurement, identical methodology to the XConv peak-memory code.

``scripts/compute_peak_memory.py`` (``compute_iteration_memory``) loops over the
loader with ``if i > 1: break`` -- i.e. it runs iterations ``i=0`` and ``i=1``
(TWO tracked iterations), then stops. Each iteration brackets one
``forward + loss + zero_grad + backward`` step with ``MemoryTracker`` (whose
``__enter__`` calls ``torch.cuda.reset_peak_memory_stats()`` and whose
``__exit__`` reads ``torch.cuda.max_memory_allocated()``) and runs
``optimizer.step()`` OUTSIDE the tracker afterwards. ``peak_mem`` is overwritten
every iteration, so the value returned is the LAST iteration's
``t.torch_peak / 2**20`` (MiB); the first iteration is effectively warm-up
(cuDNN/allocator state settles). We replicate that exactly: ``n_iters`` tracked
iterations (default 2 == the repo's ``i>1`` break), peak read from the last one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pyxconv.nvidia_mem_tracker import MemoryTracker

__all__ = ["peak_memory_mib"]


def peak_memory_mib(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_iters: int = 2,
    loss_fn=None,
    precision: str = "fp32",
) -> float:
    """Peak CUDA memory (MiB) of one training step, measured as in the paper.

    Mirrors ``compute_peak_memory.py`` EXACTLY: ``n_iters`` tracked iterations
    (default 2, matching the repo's ``if i > 1: break`` -> iterations ``i=0,1``),
    each bracketing ``forward + loss + zero_grad + backward`` with a fresh
    ``MemoryTracker`` and running ``optimizer.step()`` outside the tracker; the
    peak returned is the LAST iteration's ``torch_peak / 2**20``, so the first
    iteration acts as warm-up just as the repo's ``i=0`` does.

    Default loss is ``NLLLoss`` (the classification net ends in log_softmax); the
    loss term is negligible for peak memory, which is dominated by conv
    activations. Pass ``loss_fn`` (e.g. ``nn.MSELoss()``) for other tasks.

    Args:
        model: network on CUDA.
        inputs, targets: a batch (use ``data.random_inputs``).
        n_iters: tracked fwd+bwd iterations; the LAST iteration's peak is
            returned (default 2 == the repo's protocol).
        loss_fn: callable ``(output, targets) -> scalar``; defaults to NLLLoss.
        precision: ``'fp32'`` or ``'fp16'``. ``'fp16'`` casts the model and
            (float) inputs to ``torch.half`` -- this is what the repo's
            ``--bf16_precision`` flag does (``.half()``), halving activation memory
            so a larger batch fits the budget.
    """
    if loss_fn is None:
        loss_fn = nn.NLLLoss()
    if precision == "fp16":
        model = model.half()
        inputs = inputs.half()
        if targets.is_floating_point():
            targets = targets.half()
    model.train()
    # lr=0 + step does not change weights: the optimizer is only here for
    # zero_grad/step parity with compute_peak_memory.py (step is outside the
    # tracker and so does not enter the peak).
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    peak_mib = 0.0
    for _ in range(n_iters):
        with MemoryTracker() as tracker:
            output = model(inputs)
            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
        peak_mib = tracker.torch_peak / 2**20  # last iteration's peak, as in repo
        optimizer.step()
    return peak_mib
