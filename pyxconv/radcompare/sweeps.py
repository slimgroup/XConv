"""Compute orchestration: sweep each method's knob, record peak memory and AGE.

Peak memory and AGE are measured at the same operating point (batch size, native
32x32 resolution) so they can be joined into the AGE-vs-memory tradeoff. AGE uses
one fixed set of weights theta and one fixed minibatch partition for every
method, so differences reflect the estimator alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from torch.utils.data import DataLoader

from pyxconv.radcompare.age import average_gradient_error, exact_full_gradient
from pyxconv.radcompare.data import (
    LazyRegressionDataset,
    random_dataset_loader,
    random_inputs,
    random_regression_loader,
)
from pyxconv.radcompare.memory import peak_memory_mib
from pyxconv.radcompare.networks import (
    build_exact,
    build_rad,
    build_xconv,
    conv_weight_names,
)
from pyxconv.radcompare.unet import build_unet_exact, build_unet_rad, build_unet_xconv

__all__ = [
    "run_memory_sweep",
    "run_age_sweep",
    "join_records",
    "run_imgsize_sweep",
    "run_unet_imgsize_sweep",
    "max_batch_for_budget",
    "run_unet_age_vs_imgdim",
    "run_squeezenet_peak_memory_sweep",
    "per_sample_max_numel",
    "int32_batch_cap",
]

# CUDA's launch grid / 32-bit element indexing limit: a single tensor with
# >= 2**31 elements overflows the int32 index used in many CUDA kernels and
# crashes with "RuntimeError: integer out of range" followed by a STICKY
# "CUDA error: invalid configuration argument" (subsequent CUDA calls all
# fail, so a catch-and-retry CANNOT recover -- the overflow must be PREVENTED).
INT32_NUMEL_LIMIT = 2 ** 31
# Safety margin: bound the largest activation tensor to <= INT32_SAFETY *
# 2**31 < 2**31 so we stay strictly under the limit with slack for any
# allocator/shape rounding.
INT32_SAFETY = 0.9


class _MaxNumelMode(TorchFunctionMode):
    """Records the largest ``numel()`` of any tensor produced by a torch op
    during the forward pass it wraps.

    Unlike ``nn.Module.register_forward_hook`` (which only sees *module*
    outputs), a ``TorchFunctionMode`` observes the output of EVERY torch
    function, so it also captures tensors created by FUNCTIONAL ops --
    ``torch.cat``, ``F.interpolate``, ``F.unfold``, etc. This matters because
    the largest activation in some models is a functional-op output, not a
    module output: e.g. the U-Net's decoder level-0 ``torch.cat`` (and the
    ``F.interpolate`` feeding it) is ``2 * base_channels * N**2`` per sample --
    TWICE the largest module output (``base_channels * N**2``). A module-hook
    cap would therefore be 2x too large for the U-Net and could still let the
    concat tensor overflow int32; this mode is exact for it.
    """

    def __init__(self):
        super().__init__()
        self.max_numel = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        stack = [out]
        while stack:
            o = stack.pop()
            if torch.is_tensor(o):
                if o.numel() > self.max_numel:
                    self.max_numel = o.numel()
            elif isinstance(o, (tuple, list)):
                stack.extend(o)
        return out


def per_sample_max_numel(model: nn.Module, sample_input: torch.Tensor) -> int:
    """Largest single-tensor element count produced by ``model`` on a B=1
    input, including FUNCTIONAL-op tensors (see ``_MaxNumelMode``).

    Run ONCE at batch=1: every activation scales linearly in batch, so the
    largest tensor at batch ``B`` is ``B * per_sample_max_numel`` and this value
    alone determines the int32-safe batch cap. The forward is done under
    ``no_grad`` in ``eval`` mode (cheap, no graph, and avoids BatchNorm's
    "more than 1 value per channel" check at B=1); the model's train/eval state
    is restored afterwards. Pass the SAME ``model``/``build_fn`` result whose
    batch is being searched -- e.g. a RAD layer materializes a large internal
    projection tensor, so its per-sample max is larger than the bare
    activation, and capping on it is exactly what prevents a RAD-internal
    overflow."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _MaxNumelMode() as mode:
            model(sample_input)
    finally:
        model.train(was_training)
    return int(mode.max_numel)


def int32_batch_cap(per_sample_numel: int) -> int:
    """Largest batch whose largest activation tensor stays safely under CUDA's
    2**31-element 32-bit indexing limit, given the per-sample max numel.

    ``floor(INT32_SAFETY * 2**31 / per_sample_numel)``, clamped to >= 1. At
    batch ``B`` the largest tensor is ``B * per_sample_numel`` elements, so this
    bounds it to ``<= INT32_SAFETY * 2**31 < 2**31`` for any model -- no
    architecture hardcoding. Returns a very large number when ``per_sample_numel
    <= 0`` (no tensors seen) so the cap never spuriously binds."""
    if per_sample_numel <= 0:
        return INT32_NUMEL_LIMIT
    return max(1, int(INT32_SAFETY * INT32_NUMEL_LIMIT / per_sample_numel))


def run_squeezenet_peak_memory_sweep(
    img_dims: list[int],
    batch_sizes: list[int],
    probing_vectors: list,
    device: str,
    n_iters: int = 2,
    checkpoint_cb=None,
    precision: str = "fp32",
) -> list[dict]:
    """Peak memory of SqueezeNet over (image dim x batch x probing vector),
    OOM-safe, reproducing the paper's peak-memory experiment: torchvision
    ``squeezenet1_0``, ``convert_net`` (mode='all' -> XConv conv + BitReLU), and
    this repo's ``MemoryTracker`` (``torch_peak/2**20``). ``'base'`` in
    ``probing_vectors`` = exact convolution. CUDA-OOM configs are recorded as NaN
    (the paper's 'infeasible' regime) instead of crashing.

    ``precision='fp16'`` measures the half-precision step (model.half() + half
    inputs, matching the AGE scripts' ``--bf16_precision`` path) -- roughly halved
    activation memory -- for the Fig-8 fp16 panels. ``'fp32'`` is the original.

    If ``checkpoint_cb`` is given it is called as ``checkpoint_cb(records)`` after
    EVERY measured (pv, batch) config (records = all rows measured so far in THIS
    call), so the caller can checkpoint + re-plot incrementally and never lose
    more than one config's worth of work on a crash.
    """
    import torch.nn as nn
    from torchvision import models

    from pyxconv.utils import convert_net

    loss_fn = nn.CrossEntropyLoss()
    records = []
    for img_dim in img_dims:
        for pv in probing_vectors:
            for batch in batch_sizes:
                def build_and_measure(pv=pv, batch=batch, img_dim=img_dim):
                    model = models.squeezenet1_0(weights=None)
                    if pv != "base":
                        convert_net(model, ps=int(pv), xmode="independent")
                    model = model.to(device)
                    if precision == "fp16":
                        model = model.half()
                    x = torch.randn(batch, 3, img_dim, img_dim, device=device)
                    if precision == "fp16":
                        x = x.half()
                    y = torch.randint(0, 1000, (batch,), device=device)
                    # model + x are ALREADY in the target dtype here, so pass
                    # the default fp32 to peak_memory_mib (its own .half() cast
                    # would be a redundant no-op on already-half tensors).
                    mib = peak_memory_mib(model, x, y, n_iters=n_iters,
                                          loss_fn=loss_fn)
                    del model, x, y
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    return mib
                mib = _oom_guard(build_and_measure, device)
                records.append(dict(img_dim=img_dim, pv=pv, batch=batch,
                                    peak_mib=mib, precision=precision))
                peak_str = "OOM" if mib != mib else f"{mib:8.1f} MiB"
                print(f"[sq-peak] img={img_dim:5d}  pv={str(pv):>5}  "
                      f"B={batch:5d}  peak={peak_str}", flush=True)
                if checkpoint_cb is not None:
                    checkpoint_cb(list(records))
    return records


def max_batch_for_budget(
    build_fn,
    image_dim: int,
    budget_mib: float,
    loss_fn,
    device: str,
    channels: int = 3,
    out_channels: int = 3,
    cap: int = 4096,
    precision: str = "fp32",
    headroom: float = 0.92,
) -> tuple:
    """Largest integer batch B whose one fwd+bwd step's peak memory <= budget_mib.

    Doubling search then binary search to integer precision (the paper's
    'maximum batch size that fits in memory'). Returns ``(max_batch, probes)``
    where ``probes`` is the list of ``(batch, peak_mib)`` measured during the
    search -- i.e. the peak-memory curve used to find the max batch (peak is NaN
    for an OOM candidate). ``max_batch`` is 0 if even B=1 exceeds the budget, and
    never exceeds ``cap`` (the dataset size). Peak is measured with the repo's
    MemoryTracker; CUDA-OOM candidates count as not fitting.

    ``headroom`` (default 0.92) shrinks the budget used for the search to
    ``headroom * budget_mib`` so the chosen max batch leaves ~8% slack: the
    measured ``MemoryTracker`` peak is the steady-state allocation, but the real
    fwd+bwd can spike higher from allocator FRAGMENTATION, which previously made
    a zero-headroom B* OOM (-> NaN AGE) at ~15.5 GB on a 16 GB board. The AGE then
    runs at this headroom-reduced max batch. ``headroom=1.0`` disables the slack.

    The search is additionally capped so it never tries a batch whose largest
    activation tensor would exceed CUDA's 2**31-element 32-bit indexing limit
    (``int32_batch_cap``): at large budgets (e.g. 80 GB) the memory-feasible
    batch can be huge, and an over-large activation crashes with a STICKY CUDA
    error that no try/except can recover from -- so the overflow is PREVENTED
    here, not caught. At small budgets (e.g. 16 GB) the memory-feasible batch is
    far below this cap, so the cap does not bind and results are unchanged.
    """
    budget_mib = headroom * budget_mib
    probes = []

    # int32-safe cap: one B=1 forward of the SAME build records the largest
    # per-sample activation tensor (incl. functional ops); the search must not
    # exceed the batch that keeps it under ~0.9 * 2**31 elements. Combined with
    # the dataset-size ``cap`` via min().
    probe_model = build_fn().to(device)
    sample = torch.randn(1, channels, image_dim, image_dim, device=device)
    psm = per_sample_max_numel(probe_model, sample)
    del probe_model, sample
    if device == "cuda":
        torch.cuda.empty_cache()
    i32_cap = int32_batch_cap(psm)
    cap = min(cap, i32_cap)

    def fits(batch: int) -> bool:
        def run():
            model = build_fn().to(device)
            x = torch.randn(batch, channels, image_dim, image_dim, device=device)
            tgt = torch.randn(batch, out_channels, image_dim, image_dim, device=device)
            # n_iters=2 mirrors compute_peak_memory.py EXACTLY (its loop runs
            # iterations i=0,1 then breaks at i>1); the last iteration's peak is
            # returned, the first acts as warm-up.
            peak = peak_memory_mib(model, x, tgt, n_iters=2,
                                   loss_fn=loss_fn, precision=precision)
            del model, x, tgt
            if device == "cuda":
                torch.cuda.empty_cache()
            return peak
        peak = _oom_guard(run, device)
        probes.append((batch, peak))
        return peak == peak and peak <= budget_mib  # NaN (OOM) -> not fitting

    if not fits(1):
        return 0, probes
    lo, hi = 1, 2
    while hi <= cap and fits(hi):
        lo, hi = hi, hi * 2
    hi = min(hi, cap + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo, probes


def run_unet_age_vs_imgdim(
    img_dims: list[int],
    r_list: list[int],
    keep_fracs: list[float],
    budget_mib: float,
    subset_size: int,
    device: str,
    n_runs: int = 4,
    theta_seed: int = 0,
    base_seed: int = 1000,
    base_channels: int = 32,
    depth: int = 4,
    ref_chunk: int = 8,
    precision: str = "fp32",
    checkpoint_cb=None,
) -> dict:
    """AGE vs image dimension, the paper's methodology, on a U-Net.

    For each method (exact Conv, XConv at each r in ``r_list`` with BitReLU, RAD
    random-projection and sampling at each keep_frac) AND each image dimension,
    the batch size is the *maximum that fits a fixed memory budget* (found by
    binary search) -- so the batch shrinks as resolution grows and XConv's lower
    memory buys a larger batch. AGE (Eq. 9) is computed at that batch against the
    exact full-dataset gradient, averaged over ``n_runs`` (random minibatch
    partition + estimator draws) for a +/- sigma band. Infeasible points (no
    batch fits the budget) are recorded as NaN.

    If ``checkpoint_cb`` is given it is called as ``checkpoint_cb(results)`` after
    EACH image dimension fully completes (all methods measured at that
    resolution), so the caller can persist the results checkpoint and re-render
    the figures incrementally -- figures appear as resolutions finish and a crash
    never loses an already-completed resolution.
    """
    loss_fn = nn.MSELoss()
    configs = [("conv", "Conv", "#1f77b4",
                lambda: build_unet_exact(base=base_channels, depth=depth))]
    xconv_colors = ["#fdae6b", "#fd8d3c", "#d62728", "#a50f15"]
    for idx, r in enumerate(r_list):
        color = xconv_colors[min(idx, len(xconv_colors) - 1)]
        configs.append((
            f"xconv_r{r}", f"XConv ($r={r}$)", color,
            (lambda rr=r: (lambda: build_unet_xconv(rr, "independent",
                                                    base=base_channels, depth=depth)))(),
        ))
    for kf in keep_fracs:
        configs.append((
            f"rad_rp_{kf}", f"RAD-RP (keep={kf})", "#9467bd",
            (lambda k=kf: (lambda: build_unet_rad(k, sparse=False,
                                                  base=base_channels, depth=depth)))(),
        ))
        configs.append((
            f"rad_s_{kf}", f"RAD-S (keep={kf})", "#2ca02c",
            (lambda k=kf: (lambda: build_unet_rad(k, sparse=True,
                                                  base=base_channels, depth=depth)))(),
        ))

    torch.manual_seed(theta_seed)
    theta = {k: v.detach().clone()
             for k, v in build_unet_exact(base=base_channels, depth=depth).state_dict().items()}
    names = conv_weight_names(build_unet_exact(base=base_channels, depth=depth))

    results = {
        key: dict(label=label, color=color, img_dims=list(img_dims),
                  batch=[], age_mean=[], age_std=[], peak_curve={})
        for key, label, color, _ in configs
    }

    for image_dim in img_dims:
        # Lazy dataset: sample i is a pure function of (i, dataset_seed), so the
        # sequential reference pass and the shuffled minibatch pass see
        # BYTE-IDENTICAL samples (only the grouping/order differs) -- AGE then
        # measures estimator/sampling error, not a data mismatch. The full
        # (subset_size, C, H, W) tensor is never materialized, so host RAM stays
        # bounded even at large image_dim.
        dataset = LazyRegressionDataset(subset_size, image_dim, dataset_seed=theta_seed)

        # Exact full-dataset gradient reference (chunked; uses the full GPU, not
        # the budget -- it is the ground truth, not a method being compared).
        ref_loader = DataLoader(dataset, batch_size=ref_chunk, shuffle=False)
        exact_model = build_unet_exact(base=base_channels, depth=depth).to(device)
        exact_model.load_state_dict(theta)
        # Reference gradient in the SAME precision as the methods it is compared
        # to (repo bf16 convention: model + inputs in half, so the AGE floor is
        # measured in half too, not fp32).
        full_grad = _oom_guard(
            lambda: exact_full_gradient(exact_model, ref_loader, names, device,
                                        loss_fn=loss_fn, precision=precision),
            device,
            what=f"exact full-gradient reference resolution={image_dim} "
                 f"ref_chunk={ref_chunk}",
        )
        del exact_model
        if device == "cuda":
            torch.cuda.empty_cache()
        have_full = isinstance(full_grad, dict)

        for key, label, color, build in configs:
            # Compute this method's (batch, AGE) inside a guard so a single
            # method's NON-infeasible error (one NOT matched by _oom_guard's
            # infeasible fragments) cannot abort the whole resolution. Without
            # this, one such error drops EVERY method at this image dim AND
            # prevents the end-of-resolution checkpoint, so the persisted file
            # silently lacks that resolution -- exactly how img 512 was lost in
            # the 80 GB run. On any error we record this method NaN and continue,
            # so the resolution still completes and checkpoints.
            batch = 0
            age_mean = age_std = float("nan")
            try:
                batch, probes = max_batch_for_budget(
                    build, image_dim, budget_mib, loss_fn, device,
                    cap=subset_size, precision=precision)
                results[key]["peak_curve"][image_dim] = probes
                if batch < 1 or not have_full:
                    print(f"[age-budget] N={image_dim:4d}  {key:14s}  "
                          f"B*={batch:5d}  AGE=infeasible", flush=True)
                else:
                    # Cap at the dataset size (never request a batch larger than
                    # the subset) and warn if the subset yields fewer than 2
                    # minibatches: with drop_last=True, M<2 leaves <=1 batch, so
                    # the AGE mean/sigma over minibatches is meaningless.
                    age_batch = min(batch, subset_size)
                    n_minibatches = subset_size // age_batch  # drop_last -> floor
                    if n_minibatches < 2:
                        print(f"[age-budget][WARN] degenerate AGE point: "
                              f"method={key} resolution={image_dim} "
                              f"batch={age_batch} M={n_minibatches} (<2 "
                              f"minibatches over subset_size={subset_size}; "
                              f"AGE/sigma unreliable)", flush=True)
                    ages = []
                    for run in range(n_runs):
                        shuffle_gen = torch.Generator().manual_seed(base_seed + run)
                        loader = DataLoader(dataset, batch_size=age_batch,
                                            shuffle=True, generator=shuffle_gen,
                                            drop_last=True)
                        torch.manual_seed(base_seed + run)  # estimator draws
                        model = build().to(device)
                        model.load_state_dict(theta)
                        ages.append(_oom_guard(
                            lambda mdl=model: average_gradient_error(
                                mdl, loader, full_grad, names, device,
                                loss_fn=loss_fn, precision=precision),
                            device,
                            what=f"AGE method={key} resolution={image_dim} "
                                 f"batch={age_batch} run={run}",
                        ))
                        del model
                        if device == "cuda":
                            torch.cuda.empty_cache()
                    valid = [a for a in ages if a == a]
                    n_oom = len(ages) - len(valid)
                    if valid:
                        ages_t = torch.tensor(valid)
                        age_mean = float(ages_t.mean())
                        age_std = float(ages_t.std(unbiased=False))
                        oom_note = (f"  ({n_oom}/{n_runs} runs OOMed)"
                                    if n_oom else "")
                        print(f"[age-budget] N={image_dim:4d}  {key:14s}  "
                              f"B*={batch:5d}  AGE={age_mean:.4g}{oom_note}",
                              flush=True)
                    else:
                        # Every run OOMed: NaN (not a silent average of fewer).
                        print(f"[age-budget][WARNING] N={image_dim:4d}  "
                              f"{key:14s}  B*={batch:5d}  AGE=NaN (all {n_runs} "
                              f"runs OOMed)", flush=True)
            except Exception as e:  # noqa: BLE001 -- last-resort per-method net
                results[key]["peak_curve"].setdefault(image_dim, [])
                batch = 0
                age_mean = age_std = float("nan")
                print(f"[age-budget][ERROR] N={image_dim:4d}  {key:14s}  "
                      f"{type(e).__name__}: {e} -> recording NaN (resolution "
                      f"continues so it still checkpoints)", flush=True)
            results[key]["batch"].append(batch)
            results[key]["age_mean"].append(age_mean)
            results[key]["age_std"].append(age_std)
        # This resolution is fully measured (all methods): checkpoint + re-plot
        # so figures appear incrementally and a crash keeps finished resolutions.
        if checkpoint_cb is not None:
            checkpoint_cb(results)
            print(f"[checkpoint] resolution N={image_dim} done; "
                  f"checkpoint + figures updated", flush=True)
    return results


# Error-message fragments that mean "this config is infeasible on this GPU"
# rather than a genuine bug: plain OOM, plus the int32-overflow / launch-grid
# family ("integer out of range" -> sticky "invalid configuration argument" /
# "CUDA error"). NOTE: the int32 cap (``int32_batch_cap``) is what actually
# PREVENTS the overflow crash -- once a CUDA error is raised it is STICKY, so
# every later CUDA call in the process fails and this guard can only turn the
# crash into a (still unrecoverable) NaN. This broadened match is a graceful-
# fail SAFETY NET, not the primary mechanism.
_INFEASIBLE_ERR_FRAGMENTS = (
    "out of memory",
    "integer out of range",
    "invalid configuration argument",
    "cuda error",
)


def _oom_guard(fn, device: str, what: str | None = None):
    """Run ``fn``; return NaN if it raises an "infeasible config" error (CUDA
    out-of-memory or the int32-overflow/launch-grid family -- see
    ``_INFEASIBLE_ERR_FRAGMENTS``), so a sweep records an 'infeasible' point
    instead of crashing; re-raise any other error.

    A swallowed error is otherwise invisible (it just becomes a NaN that gets
    dropped from the AGE mean); pass ``what`` (e.g. "AGE method=... N=... B=...")
    to ``print`` a clear WARNING identifying exactly which config failed.

    Catches ``Exception`` (not just ``RuntimeError``) because some CUDA launch
    failures surface as ``torch.AcceleratorError`` (a subclass on newer torch)."""
    try:
        return fn()
    except Exception as e:  # OutOfMemoryError/AcceleratorError are subclasses
        msg = str(e).lower()
        if any(frag in msg for frag in _INFEASIBLE_ERR_FRAGMENTS):
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # CUDA errors are sticky; empty_cache may itself fail
            if what is not None:
                print(f"[oom][WARNING] infeasible config (recorded as NaN): "
                      f"{what} :: {type(e).__name__}: {e}", flush=True)
            return float("nan")
        raise


def run_memory_sweep(
    specs,
    batch_size: int,
    image_dim: int,
    device: str,
    n_warmup: int = 1,
    n_measure: int = 3,
) -> list[dict]:
    """Peak memory (MiB) for every (method, knob) on a random batch.

    ``n_warmup``/``n_measure`` are kept for the CIFAR ``rad_vs_xconv.py`` JSON
    config; they map to ``peak_memory_mib(n_iters=n_warmup + n_measure)`` (the
    same total tracked iterations, last iteration's peak returned)."""
    records = []
    images, labels = random_inputs(batch_size, image_dim, device=device)
    for spec in specs:
        for knob in spec.knob_values:
            torch.manual_seed(0)  # deterministic build for the measurement
            model = spec.build(knob).to(device)
            mib = peak_memory_mib(
                model, images, labels, n_iters=n_warmup + n_measure
            )
            records.append(
                dict(
                    key=spec.key, label=spec.label, family=spec.family,
                    color=spec.color, knob_name=spec.knob_name, knob=knob,
                    peak_mib=mib,
                )
            )
            print(f"[mem] {spec.key:22s} {spec.knob_name}={knob!s:>6}  peak={mib:8.1f} MiB")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    return records


def run_age_sweep(
    specs,
    loader,
    device: str,
    n_seeds: int = 3,
    theta_seed: int = 0,
    base_seed: int = 1000,
) -> list[dict]:
    """AGE (Eq. 9) for every (method, knob), averaged over ``n_seeds`` estimator
    draws. Ground truth g(theta) is the exact full-dataset gradient at theta."""
    torch.manual_seed(theta_seed)
    theta_model = build_exact().to(device)
    names = conv_weight_names(theta_model)
    theta = {k: v.detach().clone() for k, v in theta_model.state_dict().items()}
    full_grad = exact_full_gradient(theta_model, loader, names, device)
    del theta_model
    if device == "cuda":
        torch.cuda.empty_cache()

    records = []
    for spec in specs:
        for knob in spec.knob_values:
            ages = []
            for s in range(n_seeds):
                torch.manual_seed(base_seed + s)
                model = spec.build(knob).to(device)
                model.load_state_dict(theta)  # identical theta across all methods
                ages.append(
                    average_gradient_error(model, loader, full_grad, names, device)
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            ages_t = torch.tensor(ages)
            records.append(
                dict(
                    key=spec.key, label=spec.label, family=spec.family,
                    color=spec.color, knob_name=spec.knob_name, knob=knob,
                    age_mean=float(ages_t.mean()),
                    age_std=float(ages_t.std(unbiased=False)),
                    age_runs=ages,
                )
            )
            print(
                f"[age] {spec.key:22s} {spec.knob_name}={knob!s:>6}  "
                f"AGE={ages_t.mean():.4g} +/- {ages_t.std(unbiased=False):.2g}"
            )
    return records


def join_records(memory_records: list[dict], age_records: list[dict]) -> list[dict]:
    """Join memory and AGE on (key, knob) for the tradeoff plot."""
    age_by = {(r["key"], r["knob"]): r for r in age_records}
    combined = []
    for mem in memory_records:
        age = age_by.get((mem["key"], mem["knob"]))
        if age is None:
            continue
        combined.append(
            {**mem, "age_mean": age["age_mean"], "age_std": age["age_std"]}
        )
    return combined


def run_imgsize_sweep(
    img_dims: list[int],
    xconv_r: int,
    rad_keep_frac: float,
    batch_size: int,
    subset_size: int,
    device: str,
    n_seeds: int = 3,
    theta_seed: int = 0,
    base_seed: int = 1000,
) -> dict:
    """Peak memory and AGE vs image dimension, at a FIXED XConv r and RAD
    keep_frac, on random inputs (the repo's AGE-vs-N protocol). Uses the
    adaptive-head net so every image size is valid; the conv weights theta are
    image-size-independent, so one theta is reused across all dimensions.

    Returns ``{method_key: {label, color, img_dims, peak_mib[], age_mean[],
    age_std[]}}`` with one entry per image dimension.
    """
    methods = [
        ("exact", "Exact conv", "#1f77b4",
         lambda: build_exact(adaptive_head=True)),
        ("xconv_independent", f"XConv (independent, $r={xconv_r}$)", "#d62728",
         lambda: build_xconv(xconv_r, "independent", adaptive_head=True)),
        ("rad_rp", f"RAD (random proj., keep={rad_keep_frac})", "#9467bd",
         lambda: build_rad(rad_keep_frac, sparse=False, adaptive_head=True)),
        ("rad_sample", f"RAD (sampling, keep={rad_keep_frac})", "#8c564b",
         lambda: build_rad(rad_keep_frac, sparse=True, adaptive_head=True)),
    ]

    torch.manual_seed(theta_seed)
    theta = {k: v.detach().clone()
             for k, v in build_exact(adaptive_head=True).state_dict().items()}
    names = conv_weight_names(build_exact(adaptive_head=True))

    results = {
        key: dict(label=label, color=color, img_dims=list(img_dims),
                  peak_mib=[], age_mean=[], age_std=[])
        for key, label, color, _ in methods
    }

    for image_dim in img_dims:
        images, labels = random_inputs(batch_size, image_dim, device=device)
        loader = random_dataset_loader(subset_size, batch_size, image_dim, seed=theta_seed)

        exact_model = build_exact(adaptive_head=True).to(device)
        exact_model.load_state_dict(theta)
        full_grad = exact_full_gradient(exact_model, loader, names, device)
        del exact_model
        if device == "cuda":
            torch.cuda.empty_cache()

        for key, label, color, build in methods:
            mem_model = build().to(device)
            mib = peak_memory_mib(mem_model, images, labels)
            del mem_model
            if device == "cuda":
                torch.cuda.empty_cache()

            ages = []
            for s in range(n_seeds):
                torch.manual_seed(base_seed + s)
                model = build().to(device)
                model.load_state_dict(theta)
                ages.append(
                    average_gradient_error(model, loader, full_grad, names, device)
                )
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            ages_t = torch.tensor(ages)
            results[key]["peak_mib"].append(mib)
            results[key]["age_mean"].append(float(ages_t.mean()))
            results[key]["age_std"].append(float(ages_t.std(unbiased=False)))
            print(
                f"[imgsweep] N={image_dim:4d}  {key:18s}  "
                f"peak={mib:8.1f} MiB  AGE={ages_t.mean():.4g}"
            )
    return results


def run_unet_imgsize_sweep(
    img_dims: list[int],
    xconv_r: int,
    rad_keep_frac: float,
    batch_size: int,
    subset_size: int,
    device: str,
    n_seeds: int = 3,
    theta_seed: int = 0,
    base_seed: int = 1000,
    base_channels: int = 32,
    depth: int = 4,
) -> dict:
    """Peak memory + AGE vs image dimension on a standard U-Net (MSE regression,
    random inputs). XConv uses BitReLU (``mode='all'``); RAD uses its conv
    estimator with exact ReLU. CUDA-OOM points are recorded as NaN (e.g. RAD or
    exact may be infeasible at high resolution) rather than crashing.
    """
    loss_fn = nn.MSELoss()
    methods = [
        ("exact", "Exact conv", "#1f77b4",
         lambda: build_unet_exact(base=base_channels, depth=depth)),
        ("xconv_independent", f"XConv (independent, $r={xconv_r}$)", "#d62728",
         lambda: build_unet_xconv(xconv_r, "independent", base=base_channels, depth=depth)),
        ("rad_rp", f"RAD (random proj., keep={rad_keep_frac})", "#9467bd",
         lambda: build_unet_rad(rad_keep_frac, sparse=False, base=base_channels, depth=depth)),
        ("rad_sample", f"RAD (sampling, keep={rad_keep_frac})", "#8c564b",
         lambda: build_unet_rad(rad_keep_frac, sparse=True, base=base_channels, depth=depth)),
    ]

    torch.manual_seed(theta_seed)
    theta = {k: v.detach().clone()
             for k, v in build_unet_exact(base=base_channels, depth=depth).state_dict().items()}
    names = conv_weight_names(build_unet_exact(base=base_channels, depth=depth))

    results = {
        key: dict(label=label, color=color, img_dims=list(img_dims),
                  peak_mib=[], age_mean=[], age_std=[])
        for key, label, color, _ in methods
    }

    for image_dim in img_dims:
        inputs = torch.randn(batch_size, 3, image_dim, image_dim, device=device)
        targets = torch.randn(batch_size, 3, image_dim, image_dim, device=device)
        loader = random_regression_loader(subset_size, batch_size, image_dim, seed=theta_seed)

        exact_model = build_unet_exact(base=base_channels, depth=depth).to(device)
        exact_model.load_state_dict(theta)
        full_grad = _oom_guard(
            lambda: exact_full_gradient(exact_model, loader, names, device, loss_fn=loss_fn),
            device,
        )
        have_full = isinstance(full_grad, dict)
        del exact_model
        if device == "cuda":
            torch.cuda.empty_cache()

        for key, label, color, build in methods:
            mem_model = build().to(device)
            mib = _oom_guard(
                lambda m=mem_model: peak_memory_mib(m, inputs, targets, loss_fn=loss_fn),
                device,
            )
            del mem_model
            if device == "cuda":
                torch.cuda.empty_cache()

            ages = []
            if have_full:
                for s in range(n_seeds):
                    torch.manual_seed(base_seed + s)
                    model = build().to(device)
                    model.load_state_dict(theta)
                    ages.append(_oom_guard(
                        lambda mdl=model: average_gradient_error(
                            mdl, loader, full_grad, names, device, loss_fn=loss_fn),
                        device,
                    ))
                    del model
                    if device == "cuda":
                        torch.cuda.empty_cache()
            valid = [a for a in ages if a == a]  # drop NaN (OOM)
            if valid:
                ages_t = torch.tensor(valid)
                age_mean, age_std = float(ages_t.mean()), float(ages_t.std(unbiased=False))
            else:
                age_mean = age_std = float("nan")

            results[key]["peak_mib"].append(mib)
            results[key]["age_mean"].append(age_mean)
            results[key]["age_std"].append(age_std)
            peak_str = "OOM" if mib != mib else f"{mib:8.1f} MiB"
            print(f"[unet-imgsweep] N={image_dim:4d}  {key:18s}  peak={peak_str}  AGE={age_mean:.4g}")
    return results
