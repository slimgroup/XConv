import os
import sys
import torch
import torch.nn as nn
import argparse
import pandas as pd

# Anti-fragmentation. Legacy name (vanillanet_env) + the torch>=2.9 name (sips),
# both set before torch CUDA init. Methodology-neutral.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import models.vanillanet  # noqa: F401  (registers vanillanet_* with timm)
from timm.models import create_model

# Repo root on sys.path so pyxconv resolve without a pip install
# into the shared sips env (this script lives in VanillaNet/, repo root is ../).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# The legacy `from nvidia_mem_tracker import MemoryTracker` resolved against
# scripts/ on sys.path; the tracker now lives in pyxconv/, so import it from
# there (works in both envs).
from pyxconv.nvidia_mem_tracker import MemoryTracker
from pyxconv.utils import convert_net, adaptive_convert_net

"""
Peak memory of VanillaNet-10, the paper's Fig-10 peak-memory experiment.

TWO entry points:

  * Legacy per-config (driven by bash_seq_compute_vanillanet_10_peak_memory.sh):
        python3 compute_vanillanet_peak_memory.py --exp_name ... --mem_log_dir ...
            --batch_size B --probing_vector PV --img_dim N --xconv_varn adaptive_xconv ...
    measures ONE (batch, pv, img) config and appends a CSV row, exactly as
    before. (Bug fix: adaptive_convert_net is now called with sample_input=,
    its real argument; the old input_shape= kwarg did not exist and raised.)

  * Single-process sweep (mirrors scripts/squeezenet_peak_memory.py), Fig-10:
        python3 compute_vanillanet_peak_memory.py --sweep \
            --img_dims 64 128 256 512 \
            --probing_vectors base 2 4 ... 4096 \
            --batch_sizes 32 64 ... 512 \
            --mem_log_dir <out> --xconv_varn adaptive_xconv
    sweeps probing-vectors x batch x img in ONE process, OOM-safe (OOM ->
    NaN/gap, the paper's 'infeasible' regime, so the grid never crashes),
    checkpoints + RE-PLOTS after every measured config, and writes the Fig-10
    figure per image dim (probing-vectors LOG x peak-memory-GB LOG, one line per
    batch size, with the exact-conv 'base' as a per-batch horizontal dashed
    reference) via pyxconv.radcompare.plotting.plot_peak_vs_probing.

Usage:
    sh bash_scripts/bash_seq_compute_vanillanet_10_peak_memory.sh   # legacy
    sh bash_scripts/sips_compute_vanillanet_10_peak_memory.sh        # sweep (sips)
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute peak memory of the VanillaNet model.'
    )
    parser.add_argument('--in_channel_size', type=int, default=3,
                        help='Number of input channels.')
    parser.add_argument(
        '--mem_log_dir',
        type=str,
        required=True,
        help='Path to save CSVs (legacy) / checkpoint + Fig-10 figures (sweep).'
    )
    parser.add_argument('--exp_name', type=str, default='vanillanet_10_adaptive_xconv',
                        help='Experiment variation (CSV/figure name prefix).')
    parser.add_argument('--batch_size', type=int, default=14,
                        help='Batch size (legacy single-config).')
    parser.add_argument('--probing_vector', type=str, default='base',
                        help='Probing vector (legacy single-config).')
    parser.add_argument('--model', type=str, default='vanillanet_10', help='Model name')
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--drop', type=float, default=0, metavar='PCT',
                        help='Drop rate (default: 0.0)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--img_dim', default=224, type=int,
                        help='input image dimension (legacy single-config)')
    parser.add_argument(
        '--xconv_varn',
        type=str,
        default='adaptive_xconv',
        help='Type of Xconv variation (adaptive_xconv/xconv).'
    )
    # --- sweep-mode args (mirror scripts/squeezenet_peak_memory.py) ---
    parser.add_argument('--sweep', action='store_true',
                        help='Run the full single-process Fig-10 sweep instead '
                             'of a single legacy config.')
    parser.add_argument('--img_dims', type=int, nargs='+',
                        default=[64, 128, 256, 512],
                        help='(sweep) image dimensions.')
    parser.add_argument('--probing_vectors', type=str, nargs='+',
                        default=['base', '2', '4', '8', '16', '32', '64', '128',
                                 '256', '512', '1024', '2048', '4096'],
                        help="(sweep) probing vectors; 'base' = exact conv.")
    parser.add_argument('--batch_sizes', type=int, nargs='+',
                        default=[32, 64, 128, 256, 512],
                        help='(sweep) batch sizes.')
    parser.add_argument('--budget_mib', type=float, default=0.92 * 16384,
                        help='(sweep) configs whose peak exceeds this are capped '
                             'to NaN/gap (16 GB card headroom).')
    parser.add_argument('--bf16_precision', action='store_true',
                        help='Measure the HALF-precision step (model.half() + '
                             'half inputs), matching the AGE scripts\' '
                             '--bf16_precision path -- the fp16 Fig-10 panels. '
                             'Use a separate --mem_log_dir so fp32 results are '
                             'untouched.')
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------
# Shared build (legacy + sweep): timm VanillaNet -> Adaptive XConv, BN/Dropout
# eval (the AGE forward map, so peak memory matches the AGE run).
# ----------------------------------------------------------------------------
def _eval_bn_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.modules.batchnorm._BatchNorm,
                          nn.modules.dropout._DropoutNd)):
            m.eval()
    return model


def build_vanillanet(model_name, pv, img_dim, in_channel_size, nb_classes,
                     act_num, drop, xconv_varn, device, precision="fp32"):
    """timm create_model, then Adaptive XConv (sample_input=) for a numeric pv,
    matching comp_mini_batch_gradient_avg_grad_err.py. 'base' = exact net.

    ``precision='fp16'`` casts the model to ``torch.half`` (model+inputs .half(),
    matching the AGE scripts' ``--bf16_precision`` path) -- the fp16 Fig-10
    panels. The cast happens BEFORE adaptive_convert_net (mirroring the AGE
    script's model.half() -> adaptive_convert_net order) and the dry-forward dummy
    is cast too, so the converted layers and dummy share the half dtype."""
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=nb_classes,
        act_num=act_num,
        drop_rate=drop,
        deploy=False,
    ).to(device)
    if precision == "fp16":
        model = model.half()
    base = (str(pv) == 'base')
    if not base:
        if "adaptive" in xconv_varn:
            dummy = torch.randn(1, in_channel_size, img_dim, img_dim, device=device)
            if precision == "fp16":
                dummy = dummy.half()
            model = adaptive_convert_net(
                model,
                sample_input=dummy,   # FIX: real arg name (was input_shape=)
                ps=int(pv),
                xmode='independent',
                mode='all',
                maxc=32001,
            )
        else:
            convert_net(model, ps=int(pv), xmode='independent')
    return _eval_bn_dropout(model)


def _oom_guard(fn, device):
    """Run fn; return NaN on CUDA OOM (records an 'infeasible' point instead of
    crashing the sweep), re-raise other errors. Mirrors pyxconv.radcompare.sweeps."""
    try:
        return fn()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if device == "cuda":
                torch.cuda.empty_cache()
            return float("nan")
        raise


def measure_peak(model_name, pv, img_dim, batch_size, in_channel_size,
                 nb_classes, act_num, drop, xconv_varn, device, n_iters=2,
                 precision="fp32"):
    """Peak MiB of one fwd+bwd step, MemoryTracker (torch_peak/2**20), the repo's
    protocol: n_iters tracked iterations (default 2), optimizer.step OUTSIDE the
    tracker, last iteration's peak returned. OOM -> NaN.

    ``precision='fp16'`` measures the half-precision step (model.half() + half
    inputs)."""
    def run():
        model = build_vanillanet(model_name, pv, img_dim, in_channel_size,
                                 nb_classes, act_num, drop, xconv_varn, device,
                                 precision=precision)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        images = torch.randn(batch_size, in_channel_size, img_dim, img_dim,
                             device=device)
        if precision == "fp16":
            images = images.half()
        labels = torch.randint(0, nb_classes, (batch_size,), device=device)
        peak_mib = 0.0
        for _ in range(n_iters):
            with MemoryTracker() as t:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
            peak_mib = t.torch_peak / 2 ** 20
            optimizer.step()
        del model, images, labels, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()
        return peak_mib
    return _oom_guard(run, device)


# ----------------------------------------------------------------------------
# Sweep mode (Fig-10), mirrors scripts/squeezenet_peak_memory.py
# ----------------------------------------------------------------------------
def run_sweep(args):
    from pyxconv.radcompare import plotting

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if args.bf16_precision else "fp32"
    print(f"[device] {device}  budget={args.budget_mib:.1f} MiB  "
          f"precision={precision}", flush=True)
    os.makedirs(args.mem_log_dir, exist_ok=True)
    ckpt_path = os.path.join(args.mem_log_dir, "checkpoint.pth")

    plotting.apply_paper_style()
    done = []  # rows for image dims already fully finished
    prec_tag = " [fp16]" if precision == "fp16" else ""

    def checkpoint(current):
        """Persist + RE-PLOT all rows finished so far (finished dims + the
        in-progress dim's rows-so-far). One Fig-10 PDF/PNG per image dim."""
        records = done + current
        torch.save({"records": records, "args": vars(args)}, ckpt_path)
        for image_dim in sorted({r["img_dim"] for r in records}):
            plotting.plot_peak_vs_probing(
                records, image_dim,
                os.path.join(args.mem_log_dir,
                             f"vanillanet_peak_memory_img{image_dim}.pdf"),
                title=f"VanillaNet-10 peak memory{prec_tag} "
                      f"(Img-Size={image_dim})",
            )

    for image_dim in args.img_dims:
        dim_records = []
        for pv in args.probing_vectors:
            for batch in args.batch_sizes:
                mib = measure_peak(
                    args.model, pv, image_dim, batch, args.in_channel_size,
                    args.nb_classes, args.act_num, args.drop, args.xconv_varn,
                    device, precision=precision)
                # OOM-cap: treat over-budget peaks as infeasible (gap), same as
                # OOM, so the figure shows only configs that actually fit.
                if mib == mib and mib > args.budget_mib:
                    mib = float("nan")
                dim_records.append(dict(img_dim=image_dim, pv=pv, batch=batch,
                                        peak_mib=mib))
                peak_str = "OOM/cap" if mib != mib else f"{mib:8.1f} MiB"
                print(f"[vn-peak] img={image_dim:5d}  pv={str(pv):>5}  "
                      f"B={batch:5d}  peak={peak_str}", flush=True)
                checkpoint(dim_records)  # incremental: never lose > 1 config
        done += dim_records
        checkpoint([])
        print(f"[checkpoint] img_dim={image_dim} done; checkpoint + figures "
              f"updated at {args.mem_log_dir}", flush=True)

    # Also dump a tidy CSV for downstream inspection.
    pd.DataFrame(done).to_csv(
        os.path.join(args.mem_log_dir, f"{args.exp_name}_peak_memory_sweep.csv"),
        index=False)
    print(f"[sweep] done; CSV + figures in {args.mem_log_dir}", flush=True)


# ----------------------------------------------------------------------------
# Legacy single-config mode (driven by the original bash, one CSV row)
# ----------------------------------------------------------------------------
def main(args):
    if args.sweep:
        run_sweep(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if args.bf16_precision else "fp32"
    batch_size = args.batch_size
    probing_vector = args.probing_vector
    img_dim = args.img_dim

    print(f"Processing batch-size: {batch_size}  probing_vector: {probing_vector}"
          f"  img_dim: {img_dim}  precision: {precision}")
    peak_mem = measure_peak(
        args.model, probing_vector, img_dim, batch_size, args.in_channel_size,
        args.nb_classes, args.act_num, args.drop, args.xconv_varn, device,
        precision=precision)
    print(f"peak_memory (MiB): {peak_mem}")

    if str(probing_vector) != 'base':
        probing_vector = int(probing_vector)
    rows = [{
        "batch_size": batch_size,
        "probing_vector": probing_vector,
        "peak_memory": peak_mem,
    }]

    mem_bs_dir = "{}/{}_batch".format(args.mem_log_dir, batch_size)
    if not os.path.exists(mem_bs_dir):
        os.makedirs(mem_bs_dir)
    memory_csv_file = "{}/{}_batch_size_{}_probing_vector_{}_peak_memory.csv".format(
        mem_bs_dir, args.exp_name, batch_size, probing_vector)
    df = pd.DataFrame(rows)
    df.to_csv(
        memory_csv_file,
        columns=['batch_size', 'probing_vector', 'peak_memory'],
        index=False
    )
    print(f"Saved {memory_csv_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
