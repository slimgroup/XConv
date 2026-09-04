"""Finetune the pretrained spleen-bundle UNet, with or without XConv, and measure
peak GPU memory. No pretraining (init comes from the bundle), no custom pipeline
(data/net/loss are the bundle's). XConv only swaps the convs for a probed,
low-memory weight gradient and lets us push the probing count ``r`` up.

    python run.py --method baseline
    python run.py --method xconv                 # max-r: drop batch as needed
    python run.py --method xconv --size_strategy fixed_batch
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from monai.optimizers import Novograd
from pyxconv.radcompare import peak_memory_mib

import bundle as B
import engine
import memory as mem
import xconv_ops as xc

_HERE = os.path.dirname(os.path.abspath(__file__))
PS_BATCH_LADDER = (32, 24, 16, 12, 8, 4)           # XConv batch ladder (multiples of NUM_SAMPLES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["baseline", "xconv"], default="baseline")
    p.add_argument("--size_strategy", choices=["max_ps", "fixed_batch"], default="max_ps")
    p.add_argument("--batch", type=int, default=0)     # GPU batch B (0=auto, multiple of 4)
    p.add_argument("--ps", type=int, default=0)        # probing count r (0=auto)
    p.add_argument("--xmode", default="independent")
    p.add_argument("--xconv_target", choices=["conv", "all"], default="conv")
    p.add_argument("--max_steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.0)        # 0 -> bundle lr (0.002); else override
    p.add_argument("--val_volumes", type=int, default=5)   # held-out volumes for Dice
    p.add_argument("--val_every", type=int, default=20)    # steps between val-loss evals (0=off)
    p.add_argument("--save_ckpt", type=int, default=1)     # save finetuned state_dict
    p.add_argument("--mem_budget_gb", type=float, default=15.0)
    p.add_argument("--dataset_dir", default=os.path.join(_HERE, "data", "Task09_Spleen"))
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default=os.path.join(_HERE, "results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; CPU (memory study is GPU-only).")
        return torch.device("cpu")
    return torch.device(name)


def run_name(a) -> str:
    base = f"{a.method}_unet_spleen_b{a.batch}"
    return base + (f"_r{a.ps}_{a.xmode}_{a.xconv_target}" if a.method == "xconv" else "")


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    device = _device(a.device)
    os.makedirs(a.out_dir, exist_ok=True)

    loss_fn = B.loss_fn(a.dataset_dir)
    lr = a.lr if a.lr > 0 else B.optimizer_lr(a.dataset_dir)

    make_inputs = lambda b: B.synthetic_batch(b, device)   # for peak-memory sizing
    xbare = lambda ps: xc.apply_xconv(B.bare_net(a.dataset_dir, device), ps, a.xmode, a.xconv_target)

    # --- 1) baseline follows the REPO hyperparams: its batch is the bundle's own
    #    GPU batch (no auto-sizing). That batch is also the ceiling for the XConv
    #    max-r search. --batch overrides.
    b0 = a.batch if a.batch > 0 else B.repo_batch(a.dataset_dir)
    print(f"[recipe] repo batch={b0}, Novograd(lr={lr}), StepLR, DiceCELoss "
          f"(bundle defaults; --max_steps controls length, repo uses 800 epochs)")

    # --- 2) resolve (batch, ps) ---
    if a.method == "baseline":
        if a.batch <= 0:
            a.batch = b0
    else:
        if a.size_strategy == "max_ps" and a.ps <= 0:
            print(f"[sizing] maximize r (drop batch as needed, ceiling B0={b0})")
            a.ps, a.batch, m = mem.maximize_ps(xbare, make_inputs, loss_fn, device,
                                               a.mem_budget_gb, max_batch=b0,
                                               batch_ladder=PS_BATCH_LADDER,
                                               min_batch=B.NUM_SAMPLES)
            note = "" if a.batch == b0 else f" (batch {b0}->{a.batch} for larger r)"
            print(f"[sizing] r={a.ps} at batch={a.batch} (peak {m:.2f} GB){note}")
        else:
            if a.batch <= 0:
                a.batch = b0
            if a.ps <= 0:
                a.ps, m = mem.find_max_ps(xbare, make_inputs, a.batch, loss_fn, device,
                                          a.mem_budget_gb)
                print(f"[sizing] r={a.ps} at fixed batch={a.batch} (peak {m:.2f} GB)")

    # --- 3) build the finetuning model: load pretrained THEN convert (preserve init) ---
    model = B.pretrained_net(a.dataset_dir, device)
    preserved = None
    if a.method == "xconv":
        before = xc.snapshot_conv_weights(model)
        xc.apply_xconv(model, ps=a.ps, xmode=a.xmode, target=a.xconv_target)
        preserved = xc.assert_convert_preserved(model, before)
        print(f"[guard] convert preserved the pretrained init (max rel delta {preserved:.1e})")
    print(f"[model] {a.method}: {xc.conv_report(model)}")

    loader = B.train_loader(a.dataset_dir, a.batch, a.num_workers)
    optimizer = Novograd(model.parameters(), lr=lr)        # bundle optimizer
    scheduler = B.make_scheduler(optimizer, a.dataset_dir)  # bundle StepLR
    torch.backends.cudnn.benchmark = True

    init = xc.snapshot_conv_weights(model)

    # --- 4) canonical peak memory: pyxconv.radcompare.peak_memory_mib (torch_peak from the
    #    repo MemoryTracker, 2-iteration warm-up, SGD lr=0). Same metric everywhere.
    #    GPU-only metric; on a CPU run (e.g. when the GPU is contended) we skip it and
    #    rely on the batch-64 GPU headline for memory — this run is for dynamics/Dice.
    if device.type == "cuda":
        xp, yp = make_inputs(a.batch)
        peak_mib = peak_memory_mib(model, xp, yp, loss_fn=loss_fn)
        print(f"[memory] {a.method} peak {peak_mib:.0f} MiB (torch_peak, 2-iter, batch={a.batch})")
    else:
        peak_mib = None
        print(f"[memory] CPU run — peak memory not measured (batch={a.batch}); "
              f"use the batch-64 GPU headline for memory")

    # --- 5) finetune, proving the convs actually move (memory already measured) ---
    val_batch = B.fixed_val_patches(a.dataset_dir, 8, device) if a.val_every else None
    print(f"[finetune] {a.method} {a.max_steps} steps (batch={a.batch}"
          + (f", r={a.ps}" if a.method == "xconv" else "") + ")")
    hist = engine.finetune(model, loader, loss_fn, optimizer, a.max_steps, device,
                           scheduler=scheduler, val_batch=val_batch, val_every=a.val_every)
    deltas = list(xc.conv_update_deltas(model, init).values())
    gate = 1e-4  # bundle Novograd has no weight decay, so any motion is gradient-driven
    conv_update = {"min": min(deltas), "max": max(deltas), "mean": sum(deltas) / len(deltas),
                   "convs_updated": bool(min(deltas) > gate)}
    print(f"[verify] convs updated: {conv_update['convs_updated']} "
          f"(min {conv_update['min']:.2e}, mean {conv_update['mean']:.2e})")

    # --- 6) segmentation accuracy: mean foreground Dice on held-out volumes ---
    val_dice = None
    try:
        vl = B.val_loader(a.dataset_dir, a.val_volumes, a.num_workers)
        val_dice = engine.evaluate_dice(model, vl, B.PATCH, device)
        print(f"[eval] mean foreground Dice on {a.val_volumes} val volumes: {val_dice:.4f}")
    except RuntimeError as err:
        print(f"[eval] Dice skipped ({'OOM' if 'out of memory' in str(err).lower() else 'error'}): {err}")

    losses = hist["losses"]
    np.save(os.path.join(a.out_dir, run_name(a) + "_losses.npy"),
            np.asarray(losses, dtype=np.float32))
    if hist.get("val_losses"):
        np.save(os.path.join(a.out_dir, run_name(a) + "_val_losses.npy"),
                np.asarray(hist["val_losses"], dtype=np.float32))   # rows: (step, loss)
    if a.save_ckpt:
        ckpt_dir = os.path.join(a.out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, run_name(a) + ".pth"))
        print(f"[ckpt] saved {run_name(a)}.pth")
    result = {
        "method": a.method, "run_name": run_name(a),
        "recipe": {"optimizer": "Novograd", "lr": lr, "scheduler": "StepLR(5000,0.1)",
                   "loss": "DiceCELoss", "repo_batch": B.repo_batch(a.dataset_dir),
                   "max_steps": a.max_steps},
        "baseline_b0": b0, "resolved_batch": a.batch,
        "resolved_ps": a.ps if a.method == "xconv" else None,
        "xmode": a.xmode, "xconv_target": a.xconv_target,
        "convert_preserved_max_delta": preserved,
        "conv_report": xc.conv_report(model), "conv_update": conv_update,
        "peak_mib": peak_mib, "peak_metric": "pyxconv.radcompare.peak_memory_mib (torch_peak, 2-iter)",
        "val_dice": val_dice, "val_volumes": a.val_volumes,
        "n_steps": len(losses),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
    }
    out = os.path.join(a.out_dir, run_name(a) + ".json")
    json.dump(result, open(out, "w"), indent=2)
    peak_str = f"peak {peak_mib:.0f} MiB" if peak_mib is not None else "peak n/a (CPU)"
    print(f"[done] {a.method}: {peak_str}"
          + (f", Dice {val_dice:.4f}" if val_dice is not None else "")
          + (f", r={a.ps} at batch {a.batch}" if a.method == "xconv" else f", batch {a.batch}")
          + f" -> {out}")


if __name__ == "__main__":
    main()
