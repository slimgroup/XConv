# XConv finetuning of the MONAI spleen UNet — memory comparison

**Run (from `research/xconv_finetune/`):** `python run.py --method baseline --batch <B> --max_steps 60` and `python run.py --method xconv --batch <B> --ps <R> --max_steps 60` (`<B>` = batch, `<R>` = probing count from `calibrate.py`).

Finetune the **pretrained** `spleen_ct_segmentation` MONAI bundle **with and
without XConv**, and compare **peak GPU memory**. No pretraining (the init is
the bundle's `models/model.pt`) and no custom pipeline — the UNet, `DiceCELoss`,
`Novograd`, `StepLR`, and data all come from the bundle's `configs/train.json`.
XConv only swaps the convolutions for a probed, low-memory weight gradient.

## REQUIRED: use the boundary-fixed pyxconv (branch `ali`)

`import pyxconv` must resolve to the in-house package on the **`ali`** branch of
`~/Codes/xconv_pv`, which fixes the probe's boundary bias (the backward used a
circular `roll` that estimated the *circular*-conv gradient, biasing it ~20–45%
for k>1 vs the zero-padded conv). The `ali` branch is **padding-aware for both 2D
and 3D**. Install it **editable** so any further upstream fixes are picked up
automatically (do **not** vendor/copy — it goes stale):

```bash
pip install -e ~/Codes/xconv_pv          # branch ali; --no-deps if you must not touch torch
python -c "import pyxconv; print(pyxconv.__file__)"   # must point inside ~/Codes/xconv_pv
```

Env: the `sips` conda env (`/home/al289197/miniconda3/envs/sips`), torch 2.9 +
CUDA, MONAI 1.5.2, `pynvml`.

## The two comparisons (exact commands)

Run from this directory. Peak memory is the **rad-vs-xconv canonical metric**,
`pyxconv.radcompare.memory.peak_memory_mib` — `torch_peak` from the repo `MemoryTracker`
with a **2-iteration warm-up** (i=0 warms cuDNN, i=1 measured; SGD lr=0 so optimizer
state is not counted). The **same** function is used in every script here, so all
memory numbers are comparable.

```bash
# 1) BASELINE — finetune the pretrained UNet with regular Conv3d, repo recipe
#    (Novograd lr=0.002, StepLR(5000,0.1), DiceCELoss). Records the peak = ceiling.
python run.py --method baseline --batch 64 --max_steps 60
#    -> results/baseline_unet_spleen_b64.json   (field: peak_mib)

# 2) XCONV — same batch, probed convs. Pick the largest probing count r whose
#    peak stays <= the baseline's, then finetune at that r.
python calibrate.py --batch 64 --rs 128,256,384,512        # -> largest r with peak <= baseline
python run.py --method xconv --batch 64 --ps <R> --max_steps 60
#    -> results/xconv_unet_spleen_b64_r<R>_independent_conv.json
```

Both JSONs carry `peak_mib`, the conv-update proof, and (for XConv)
`convert_preserved_max_delta` (must be 0 — conversion preserves the pretrained
init). Compare the two `peak_mib`: **XConv must be ≤ baseline.**

### Choosing the batch

XConv's footprint is nearly **batch-independent** (the probe `e=(spatial×r)` does
not grow with batch) while the baseline grows ~linearly, so XConv only wins at
**larger batch**. `sweep_batch.py` locates the crossover:

```bash
python sweep_batch.py --r 256 --batches 8,16,32,64,96    # baseline vs xconv peak per batch
```
On this UNet (patch 96, 16 GB card) the crossover is ~batch 64; both OOM by 96
(the UNet's skip connections pin the high-res activations, which XConv cannot
compress). Use batch 64 as the operating point above.

## AGE + peak-memory figures (the rad-vs-xconv methodology)

`age_memory.py` produces the two canonical figures for this example, reusing the
in-house `pyxconv.radcompare` metrics verbatim (so numbers match the paper methodology):
**Average Gradient Error** (paper Eq. 9, `pyxconv.radcompare.age`) and **peak memory**
(`pyxconv.radcompare.memory.peak_memory_mib` — `torch_peak` via the repo `MemoryTracker`
with a 2-iteration warm-up). It sweeps the probing count `r` at the bundle's
native 96³ patch and a fixed batch, on the pretrained UNet:

```bash
python age_memory.py --rs 2,4,8,16,32,64,128,256 --batch 8 --subset 64 --n_runs 3
#  -> results/age_memory/{age_vs_r,peak_memory_vs_r}.{pdf,png} + age_memory.json
```

- **AGE vs r**: XConv's gradient error decays toward the exact sampling floor as
  `r` grows — no residual plateau (the `ali` boundary fix makes it unbiased).
- **Peak memory vs r**: XConv vs the exact-baseline line (where it stays ≤ exact).
- AGE uses a **fixed** set of spleen patches (cropped once); peak memory uses
  random 3D inputs of the same shape. `--synthetic`/`--skip_memory` give a
  CPU-only smoke. Figures use the repo paper theme (`pyxconv.radcompare.plotting`).

## Files (committed)

| file | role |
|---|---|
| `bundle.py` | load the bundle's pretrained UNet, loss, lr, scheduler, data loader |
| `xconv_ops.py` | swap Conv3d→Xconv3D, conv introspection, init-preservation guard |
| `memory.py` | sizing via `pyxconv.radcompare.peak_memory_mib` (`find_max_ps`, `maximize_ps`) |
| `engine.py` | train step + finetune loop (pure) |
| `run.py` | CLI: finetune a method, record `peak_mib` + conv-update + preservation |
| `calibrate.py` | largest `r` whose `peak_memory_mib` ≤ baseline (operating-point picker) |
| `sweep_batch.py` | baseline-vs-XConv `peak_memory_mib` across batch (finds the crossover) |
| `age_memory.py` | AGE + peak-memory vs `r` (reuses `pyxconv.radcompare`; paper figures) |
| `visualize.py` | paper figures: CT/GT/prediction overlay + 3D grid (MONAI `blend_images`/`matshow3d`) + memory/Dice comparison |
| `auto_run.py` | size (under `peak_memory_mib`) then run baseline + XConv back to back |

`run.py` reports **Dice** (`--val_volumes`, MONAI `SlidingWindowInferer` + `MeanDice`)
and takes an optional `--lr` override. Figures land in `results/figures/`.

## Notes / caveats

- **XConv beats *exact-conv* memory only with BReLU.** With ReLU left exact
  (`--xconv_target conv`, the default), XConv only *matches* exact-conv memory —
  the stored activations are pinned by the activation, not the conv. The in-house
  paper uses BReLU (`mode='all'`) to turn this into a saving. **The MONAI spleen
  UNet uses PReLU**, which `BReLU` does not convert, so `--xconv_target all` does
  not help here; expect XConv ≈ baseline (a win only past the batch crossover).
- **Boundary fix** lives in `pyxconv` (branch `ali`), 2D **and** 3D — verified
  upstream (`verify_fix2.py`). Memory results do not depend on it; gradient
  fidelity does.
- **`r` is the memory/gradient-noise knob**; `xmode='independent'` is the cleanest
  probe and the one used here.
- The bundle + Spleen data auto-download to `bundles/` and `data/` on first use.
