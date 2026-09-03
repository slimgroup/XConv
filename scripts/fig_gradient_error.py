"""
Render the *new* "AGE vs r at a fixed 16 GB budget" figures for all three
models (SqueezeNet, VanillaNet, U-Net), re-using the AGE data already on disk.

This is a re-ORIENTATION of the existing 16 GB AGE figures
(``scripts/plot_squeezenet_16gb_age.py`` / ``scripts/plot_vanillanet_16gb_age.py``
and the pyxconv.radcompare U-Net AGE plots). Instead of "AGE vs image dimension, one
panel per r", this script draws:

    x-axis = r (probing vectors, LOG)
    y-axis = Average Gradient Error, mean +/- std (LOG)
    one PANEL per image dimension.

In each panel:
  * XConv is a CURVE over r. Each point is the AGE mean +/- std at THAT r's
    max batch within the 16 GB budget, and the max batch is annotated next to
    the point (it shrinks as r grows -- a larger r costs more activation memory
    per sample, so it buys a smaller batch).
  * Conv is a horizontal DASHED line at the Conv AGE (Conv has no r), with its
    own max-batch annotation. This is the floor the XConv curve descends toward
    as r grows: a larger r drives the probing error down (better gradient) at
    the cost of more compute / a smaller batch.
  * For the U-Net (RAD comparison) we additionally draw RAD-S and RAD-RP
    (keep_frac=0.1) as horizontal reference lines -- they have no r. RAD-RP is
    infeasible at image dimension 512 (no batch fits the budget) and is omitted
    from that panel with a note in the title.

It is CPU-only, NEVER touches the GPU, and recomputes NOTHING -- it only reads:
  * SqueezeNet : avg_grad_err.pkl scalars under
        squeezenet1_0_xconv_16gb_avg_grad_err_results/probing_vector_*/...
    with batch sizes from configs/probing_vectors/squeezenet1_0_xconv_16gb_configs/*.csv.
  * VanillaNet : avg_grad_err.pkl scalars under
        experiments/vanillanet/vanillanet_10_adaptive_xconv_16gb_avg_grad_err_results/...
    with batch sizes from
        experiments/vanillanet/pv_configs/vanillanet_10_adaptive_xconv_16gb_configs/*.csv.
  * U-Net      : the precomputed (age_mean, age_std, batch) lists baked into the
        pyxconv.radcompare checkpoint.pth (results dict, per method, per image dim).

Mean/std are computed over EXACTLY the runs present on disk (no n_runs zero-fill,
the bug in the repo's own AGE plotters). The publication style (despined, log
axes, +/-std band, bold batch labels, vector PDF + PNG at dpi 300) matches
scripts/plot_squeezenet_16gb_age.py / scripts/plot_vanillanet_16gb_age.py.

In addition to the multi-panel composite (one panel per image dim), a standalone
single-panel figure is written per image dim under ``<plot_dir>/individual/``,
and ``--csv`` exports a tidy table of every plotted AGE point. ``--precision
fp16`` reads the ``*_fp16`` input dirs / checkpoint and writes to a separate
``plots/age_vs_r_16gb_fp16`` dir; every output stem is tagged with the precision.

Usage (from the repo root, CPU only):
    python3 scripts/fig_gradient_error.py                 # all three models (fp32)
    python3 scripts/fig_gradient_error.py --models squeezenet vanillanet
    python3 scripts/fig_gradient_error.py --precision fp16 --csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import pickle
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Palette. Conv (the floor) is blue and XConv is red, matching the existing
# 16 GB AGE plotters. For the U-Net RAD reference lines we reuse the pyxconv.radcompare
# checkpoint's own colors (RAD-S green, RAD-RP purple).
# --------------------------------------------------------------------------- #
BASE_PV = "base"
CONV_COLOR = "#1f77b4"    # blue  -- standard Conv (the dashed floor)
XCONV_COLOR = "#d62728"   # red   -- XConv curve over r
RAD_S_COLOR = "#2ca02c"   # green -- RAD-S reference line
RAD_RP_COLOR = "#9467bd"  # purple -- RAD-RP reference line

MEM_BUDGET_GB = 16
DEFAULT_BUDGET_GB = 16
DEFAULT_PLOT_DIR = "plots/age_vs_r_16gb"

# SqueezeNet (XConv) ---------------------------------------------------------
# Precision-INVARIANT metadata (the err_dir / config_dir paths get an "_fp16"
# suffix for fp16; see ``squeezenet_cfg`` / ``_suffix``).
SQ_IMG_DIMS = [128, 256, 512]
SQ_RS = [4, 16, 32, 64, 128]
SQ_NAME = "SqueezeNet [XConv]"

# VanillaNet (Adaptive XConv) ------------------------------------------------
VN_IMG_DIMS = [64, 128, 256, 512]
VN_RS = [16, 32, 64, 128, 256]
VN_NAME = "VanillaNet [Adaptive XConv]"

# U-Net (RAD comparison) -----------------------------------------------------
# fp32 default literal; ``unet_ckpt_for`` swaps "precision-fp32" -> the chosen
# precision. Kept as the default for the --unet_ckpt CLI override.
UNET_CKPT = (
    "data/checkpoints/"
    "rad_vs_xconv_unet_age_img_dims-128-256-512_r_list-16-64-256_keep_fracs-0.1_"
    "mem_budget_gb-16.0_subset_size-4096_n_runs-3_base_channels-32_depth-4_"
    "precision-fp32_seed-0/checkpoint.pth"
)
UNET_NAME = "U-Net [XConv vs RAD]"


def _suffix(prec):
    """Path suffix for the precision: "" for fp32, "_fp16" for fp16."""
    return "_fp16" if prec == "fp16" else ""


def _tag(budget_gb):
    """Budget token used in every output/input name: 16 -> "16gb", 60 -> "60gb".
    This must match the budget the results were produced at."""
    return f"{budget_gb}gb"


def _plot_dir_tag_suffix(budget_gb):
    """Suffix appended to the default plot dir / output stems / CSV name so a
    non-16 budget never overwrites the committed 16 GB figures. Empty at 16 (so
    16 GB outputs stay byte-identical to the committed artifacts)."""
    return "" if budget_gb == DEFAULT_BUDGET_GB else f"_{_tag(budget_gb)}"


def squeezenet_cfg(prec, budget_gb):
    """(err_dir, config_dir) for SqueezeNet at the given precision + budget.
    The literal "16gb" token is replaced by ``f"{budget_gb}gb"`` so a non-16
    budget reads the budget-tagged directories the measurement scripts write."""
    sfx = _suffix(prec)
    tag = _tag(budget_gb)
    return (
        f"squeezenet1_0_xconv_{tag}{sfx}_avg_grad_err_results",
        f"configs/probing_vectors/squeezenet1_0_xconv_{tag}{sfx}_configs",
    )


def vanillanet_cfg(prec, budget_gb):
    """(err_dir, config_dir) for VanillaNet at the given precision + budget."""
    sfx = _suffix(prec)
    tag = _tag(budget_gb)
    return (
        f"experiments/vanillanet/vanillanet_10_adaptive_xconv_{tag}{sfx}_avg_grad_err_results",
        f"experiments/vanillanet/pv_configs/vanillanet_10_adaptive_xconv_{tag}{sfx}_configs",
    )


def unet_ckpt_for(prec, budget_gb):
    """The radcompare U-Net checkpoint path for the given precision + budget,
    derived from the fp32/16 GB default literal by swapping BOTH the precision
    token and the ``mem_budget_gb-16.0`` token (16 -> "16.0", 60 -> "60.0",
    matching the budget the U-Net results were produced at)."""
    return (
        UNET_CKPT
        .replace("precision-fp32", f"precision-{prec}")
        .replace("mem_budget_gb-16.0", f"mem_budget_gb-{float(budget_gb)}")
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["squeezenet", "vanillanet", "unet"],
        choices=["squeezenet", "vanillanet", "unet"],
    )
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Which precision sweep to read (selects the *_fp16 input dirs / "
             "checkpoint and tags the output stems + plot_dir).",
    )
    parser.add_argument(
        "--budget_gb",
        type=int,
        default=DEFAULT_BUDGET_GB,
        help="GPU memory budget in GB (default 16). Selects the budget-tagged "
             "input dirs / U-Net checkpoint ('16gb' -> f'{budget_gb}gb', "
             "'mem_budget_gb-16.0' -> 'mem_budget_gb-{float(budget_gb)}') and, "
             "for budget != 16, tags the default plot_dir / output stems / CSV "
             "so a non-16 run does not overwrite the committed 16 GB figures. "
             "Must match the budget the results were produced at.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also export a tidy CSV of every plotted (model, img, series, r) "
             "AGE point to age_vs_r_<precision>[_<tag>].csv in the plot dir.",
    )
    # Default None -> resolved in main() to DEFAULT_PLOT_DIR + budget tag +
    # _suffix(precision).
    parser.add_argument("--plot_dir", default=None)
    parser.add_argument("--unet_ckpt", default=UNET_CKPT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--y_margin_frac",
        type=float,
        default=0.10,
        help="Log-axis padding as a fraction of the data decade span.",
    )
    return parser.parse_args()


def apply_publication_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "pdf.fonttype": 42,   # editable text in vector PDF
            "ps.fonttype": 42,
        }
    )


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def return_run_num(path: str):
    match = re.search(r"run_num_(\d+)", path)
    return int(match.group(1)) if match else None


def load_batch_sizes(config_dir, probing_vectors):
    """Read max batch per (probing vector, image dim) straight from the CSVs.

    Returns ``bs_xconv[pv][img]`` (``BATCH_SIZE``) and ``bs_conv[pv][img]``
    (``CONV_BATCH_SIZE`` -- the Conv baseline batch for that row). The config
    CSVs are CRLF; ``int(...)`` tolerates the trailing ``\\r`` on the last
    column, and the batch columns we read are interior columns anyway.
    """
    bs_xconv, bs_conv = {}, {}
    for pv in probing_vectors:
        csv_path = os.path.join(config_dir, f"configs_pv{pv}.csv")
        if not os.path.isfile(csv_path):
            # This probing vector was not swept at this budget (e.g. r=64 exists
            # in the 16 GB gap-fill but not the 80 GB run). Skip it: the caller
            # already renders whatever (r, img) points have data.
            continue
        per_img_x, per_img_c = {}, {}
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                img = int(row["IMG_SIZE"])
                per_img_x[img] = int(row["BATCH_SIZE"])
                per_img_c[img] = int(row["CONV_BATCH_SIZE"])
        bs_xconv[pv] = per_img_x
        bs_conv[pv] = per_img_c
    return bs_xconv, bs_conv


def age_stats_for_pv_img(pv_dir, img_dim):
    """Mean / std / per-run AGE for one (probing vector, image dim).

    One result dir per (image dim, run); the AGE pickle is a scalar. We match
    on ``img_<dim>_...run_num_<k>`` (guarding ``img_512`` vs ``img_5120``), over
    EXACTLY the runs present on disk -- no n_runs zero-fill. The directory names
    carry a trailing ``\\r`` (CRLF-derived); ``run_num_(\\d+)`` stops before it
    and ``glob`` matches the suffix fine. Returns ``None`` if no runs found.
    """
    runs = {}
    pattern = os.path.join(pv_dir, f"img_{img_dim}_*")
    for grad_dir in sorted(glob.glob(pattern)):
        if not re.search(rf"(^|/)img_{img_dim}_", grad_dir + "/"):
            continue
        run_num = return_run_num(grad_dir)
        if run_num is None:
            continue
        err_path = os.path.join(grad_dir, "avg_grad_err.pkl")
        if not os.path.isfile(err_path):
            continue
        with open(err_path, "rb") as f:
            runs[run_num] = float(pickle.load(f))
    if not runs:
        return None
    vals = np.array([runs[k] for k in sorted(runs)], dtype=float)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "runs": {k: runs[k] for k in sorted(runs)},
    }


def collect_curves_from_pickles(err_dir, config_dir, img_dims, rs):
    """Build the per-image-dim AGE-vs-r data for a pickle-backed model.

    Returns a dict keyed by image dim:
        {
          img: {
            "conv": {"mean","std","runs","batch"},
            "xconv": {r: {"mean","std","runs","batch"}, ...},  # present rs only
          }, ...
        }
    plus a flat ``rows`` list of (model, img, r-or-Conv, mean, std, batch, runs)
    for the cross-check table.
    """
    probing_vectors = [BASE_PV] + [str(r) for r in rs]
    bs_xconv, bs_conv = load_batch_sizes(config_dir, probing_vectors)

    data, rows = {}, []
    for img in img_dims:
        conv_stats = age_stats_for_pv_img(
            os.path.join(err_dir, f"probing_vector_{BASE_PV}"), img
        )
        if conv_stats is None:
            raise FileNotFoundError(f"No Conv (base) AGE for img_{img} in {err_dir}")
        conv_batch = bs_conv[BASE_PV][img]
        conv_entry = {**conv_stats, "batch": conv_batch}

        xconv_entries = {}
        for r in rs:
            if str(r) not in bs_xconv or img not in bs_xconv[str(r)]:
                continue  # this r not swept at this budget (no config) -- skip
            stats = age_stats_for_pv_img(
                os.path.join(err_dir, f"probing_vector_{r}"), img
            )
            if stats is None:
                continue  # partial sweep -- skip this (r, img) point
            batch = bs_xconv[str(r)][img]
            xconv_entries[r] = {**stats, "batch": batch}

        data[img] = {"conv": conv_entry, "xconv": xconv_entries}

        rows.append(("Conv", img, "Conv", conv_entry["mean"], conv_entry["std"],
                     conv_batch, conv_stats["runs"]))
        for r in rs:
            if r in xconv_entries:
                e = xconv_entries[r]
                rows.append(("XConv", img, r, e["mean"], e["std"], e["batch"],
                             e["runs"]))
    return data, rows


def collect_curves_from_unet_ckpt(ckpt_path):
    """Build the per-image-dim AGE-vs-r data from the radcompare U-Net ckpt.

    The checkpoint stores, per method, parallel lists over image dims:
    ``img_dims``, ``batch``, ``age_mean``, ``age_std``. XConv methods are
    keyed ``xconv_r{R}``; ``conv`` is the dashed floor; ``rad_s_*`` / ``rad_rp_*``
    are horizontal reference lines (no r). NaN AGE / batch==0 means infeasible
    at that image dim (RAD-RP at 512) and is dropped from that panel.
    """
    import torch  # local import; never used on GPU

    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    results = ckpt["results"]

    def per_img(method):
        m = results[method]
        out = {}
        for img, b, am, asd in zip(
            m["img_dims"], m["batch"], m["age_mean"], m["age_std"]
        ):
            if b is None or b == 0 or am is None or (isinstance(am, float)
                                                     and math.isnan(am)):
                continue  # infeasible at this image dim
            out[int(img)] = {"mean": float(am), "std": float(asd),
                             "batch": int(b)}
        return out

    img_dims = [int(d) for d in results["conv"]["img_dims"]]
    rs = sorted(
        int(k.split("_r")[1]) for k in results if k.startswith("xconv_r")
    )

    conv_by_img = per_img("conv")
    xconv_by_img = {r: per_img(f"xconv_r{r}") for r in rs}
    rad_s_by_img = per_img("rad_s_0.1") if "rad_s_0.1" in results else {}
    rad_rp_by_img = per_img("rad_rp_0.1") if "rad_rp_0.1" in results else {}

    data, rows, dropped = {}, [], []
    for img in img_dims:
        # The Conv (base) AGE is the floor every panel is anchored to. It can be
        # ABSENT from conv_by_img for two reasons, handled identically here:
        #   (a) the point is infeasible -- batch<1 / OOM / a failed exact-gradient
        #       reference -> NaN AGE, which per_img() drops; or
        #   (b) the run was interrupted before this resolution finished -- the
        #       checkpoint declares the full img_dims up front but appends
        #       age_mean per-resolution (checkpoint_cb saves incrementally), so a
        #       partial checkpoint lists an img whose AGE was never recorded and
        #       the zip in per_img() silently omits it.
        # Either way we cannot draw that panel; skip it (don't crash a long batch
        # plot job over one missing resolution) and report it at the end.
        conv_entry = conv_by_img.get(img)
        if conv_entry is None:
            dropped.append(img)
            continue
        xconv_entries = {r: xconv_by_img[r][img]
                         for r in rs if img in xconv_by_img[r]}
        refs = {}
        if img in rad_s_by_img:
            refs["RAD-S (keep=0.1)"] = (rad_s_by_img[img], RAD_S_COLOR)
        if img in rad_rp_by_img:
            refs["RAD-RP (keep=0.1)"] = (rad_rp_by_img[img], RAD_RP_COLOR)
        data[img] = {"conv": conv_entry, "xconv": xconv_entries, "refs": refs}

        rows.append(("Conv", img, "Conv", conv_entry["mean"],
                     conv_entry["std"], conv_entry["batch"], None))
        for r in rs:
            if r in xconv_entries:
                e = xconv_entries[r]
                rows.append(("XConv", img, r, e["mean"], e["std"], e["batch"],
                             None))
        for label, (e, _c) in refs.items():
            rows.append((label, img, "-", e["mean"], e["std"], e["batch"],
                         None))
    if dropped:
        print(f"  [unet] WARNING: no feasible Conv (base) AGE at image dim "
              f"{', '.join(str(d) for d in dropped)} -- omitting those panels. "
              f"Cause is either infeasible-at-budget (NaN) or a checkpoint that "
              f"was saved before that resolution finished; re-run the U-Net AGE "
              f"step to fill them if the run was merely interrupted.", flush=True)
    # Return only the resolutions actually plotted, so downstream figure/panel
    # code (which indexes data[img]) never sees a dropped dim.
    feasible_img_dims = [img for img in img_dims if img in data]
    return data, rows, feasible_img_dims, rs


# --------------------------------------------------------------------------- #
# Shared log-y limits over every plotted artifact (XConv points + Conv floor +
# any reference lines), so all panels of a model share a y range.
# --------------------------------------------------------------------------- #
# AGE at/below this is numerically zero (e.g. plain Conv at the full subset ->
# no minibatch sampling -> exactly 0); unplottable on a log axis and excluded
# from the lower y-bound so it cannot drag the floor toward 1e-30.
_ZERO_AGE_EPS = 1e-12


def shared_log_ylim(data, margin_frac):
    lows, highs = [], []

    def add(entry):
        m, s = entry["mean"], entry["std"]
        highs.append(m + s)
        # Only a MEANINGFUL (numerically non-zero) mean sets the lower bound.
        # Plain Conv at the full subset has exactly-0 AGE (one minibatch == the
        # whole dataset, so no sampling error) -- 0 is unplottable on a log axis
        # and must NOT pin the y-floor near 1e-30 (which then squashes the real
        # data range). Such points are still drawn; they just fall below view.
        if m > _ZERO_AGE_EPS:
            lows.append(max(m - s, 1e-30))

    for img, d in data.items():
        add(d["conv"])
        for r, e in d["xconv"].items():
            add(e)
        for label, (e, _c) in d.get("refs", {}).items():
            add(e)
    hi = max(highs)
    lo = min(lows) if lows else max(hi / 1e3, 1e-30)
    log_lo, log_hi = np.log10(lo), np.log10(hi)
    span = max(log_hi - log_lo, 0.05)
    pad = span * margin_frac
    # A small fixed floor on the bottom pad keeps the Conv dashed line (the
    # lowest artifact) from sitting flush against the x-axis; batch labels sit
    # ABOVE every line so they need no extra room below.
    pad_lo = max(pad, 0.12)
    return 10 ** (log_lo - pad_lo), 10 ** (log_hi + pad)


def annotate_point(ax, x, y, text, color, dx, dy, ha):
    ax.annotate(
        text,
        xy=(x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        fontsize=8,
        fontweight="bold",
        color=color,
    )


def plot_panel(ax, img, panel_data, rs_all, *, show_legend):
    """Draw one image-dimension panel: XConv AGE-vs-r curve + Conv dashed line
    (+ optional RAD reference lines)."""
    conv = panel_data["conv"]
    xconv = panel_data["xconv"]
    refs = panel_data.get("refs", {})

    # XConv curve over r (only r's present for this image dim).
    rs = [r for r in rs_all if r in xconv]
    xr = np.array(rs, dtype=float)
    ym = np.array([xconv[r]["mean"] for r in rs], dtype=float)
    ys = np.array([xconv[r]["std"] for r in rs], dtype=float)
    ax.plot(xr, ym, "-o", color=XCONV_COLOR, zorder=4, label="XConv (vs $r$)")
    ax.fill_between(xr, np.maximum(ym - ys, 1e-30), ym + ys,
                    color=XCONV_COLOR, alpha=0.2, zorder=3)

    # Bold max-batch annotation at each XConv point (shrinks as r grows).
    # Alternate above/below to reduce overlap on the descending curve.
    for i, r in enumerate(rs):
        above = (i % 2 == 0)
        annotate_point(ax, r, xconv[r]["mean"], str(xconv[r]["batch"]),
                       XCONV_COLOR, 0, (10 if above else -14), "center")

    # Conv dashed floor (no r): horizontal line spanning the panel. The batch
    # label is pinned to the LEFT edge and placed ABOVE the line: the XConv
    # curve's leftmost point (smallest r) is its highest value, far above the
    # floor, so there is always a clear gap just above the Conv line at the
    # left -- whereas the right edge is where the descending curve approaches
    # the floor, and the bottom of the axes is crowded by the x-tick labels.
    if conv["mean"] > _ZERO_AGE_EPS:
        ax.axhline(conv["mean"], ls="--", color=CONV_COLOR, lw=1.8, zorder=2,
                   label="Conv (no $r$)")
        ax.fill_between([0, 1], conv["mean"] - conv["std"], conv["mean"] + conv["std"],
                        transform=ax.get_yaxis_transform(), color=CONV_COLOR,
                        alpha=0.12, zorder=1)
        ax.annotate(
            f"batch {conv['batch']}", xy=(0.0, conv["mean"]),
            xycoords=ax.get_yaxis_transform(), xytext=(4, 3),
            textcoords="offset points", ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=CONV_COLOR,
        )
    else:
        # Conv AGE == 0 (batch == subset_size -> one minibatch == the whole
        # dataset, no sampling error): unplottable on a log axis. Note it at the
        # bottom edge and keep a legend handle so the panel still documents Conv.
        ax.plot([], [], ls="--", color=CONV_COLOR, lw=1.8,
                label="Conv (no $r$, $\\approx$ 0)")
        ax.annotate(
            f"Conv $\\approx$ 0  (batch {conv['batch']}, full subset)",
            xy=(0.02, 0.03), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=CONV_COLOR,
        )

    # Optional RAD reference lines (U-Net only): pinned to the left edge, above
    # each line, same side as the Conv label but well separated from it because
    # RAD AGE sits an order of magnitude or more above the Conv floor.
    for label, (e, color) in refs.items():
        ax.axhline(e["mean"], ls=":", color=color, lw=1.8, zorder=2, label=label)
        ax.annotate(
            f"batch {e['batch']}", xy=(0.0, e["mean"]),
            xycoords=ax.get_yaxis_transform(), xytext=(4, 3),
            textcoords="offset points", ha="left", va="bottom",
            fontsize=8, fontweight="bold", color=color,
        )

    ax.set_title(rf"Image dim $= {img}$")
    ax.set_xlabel("Probing vectors $r$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    rs_for_ticks = [r for r in rs_all if r in xconv]
    ax.set_xticks(rs_for_ticks)
    ax.set_xticklabels([str(r) for r in rs_for_ticks])
    ax.minorticks_off()
    # A little horizontal headroom so the leftmost/rightmost batch labels and
    # the RAD/Conv edge annotations are not clipped.
    lo_r, hi_r = min(rs_for_ticks), max(rs_for_ticks)
    ax.set_xlim(lo_r / 1.6, hi_r * 1.6)
    if show_legend:
        ax.legend(frameon=False, loc="best")


def resolve_grid(n_panels, ncols=0):
    if ncols <= 0:
        ncols = 2 if n_panels <= 4 else math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def save_figure(fig, plot_dir, stem, dpi):
    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout()
    written = []
    for ext in ("pdf", "png"):
        path = os.path.join(plot_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(os.path.abspath(path))
    plt.close(fig)
    return written


def plot_model_figure(model_name, data, img_dims, rs_all, plot_dir, stem, dpi,
                      y_margin_frac, infeasible_note="", budget_gb=MEM_BUDGET_GB):
    """One multi-panel figure for a model: one panel per image dimension.

    ``budget_gb`` only sets the budget quoted in the title/caption text; it
    defaults to 16 so the 16 GB figures stay byte-identical to the committed
    ones."""
    y_lo, y_hi = shared_log_ylim(data, y_margin_frac)
    n_panels = len(img_dims)
    nrows, ncols = resolve_grid(n_panels)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()
    for idx, img in enumerate(img_dims):
        ax = axes_flat[idx]
        if img not in data or not data[img].get("xconv"):
            ax.set_visible(False)  # no probed points yet (partial sweep)
            continue
        plot_panel(ax, img, data[img], rs_all, show_legend=(idx == 0))
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("Average Gradient Error" if idx % ncols == 0 else "")
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)
    suptitle = (
        f"AGE vs $r$ for {model_name} at a fixed {budget_gb} GB budget "
        f"(shared log-$y$)"
    )
    fig.suptitle(suptitle, y=1.005, fontsize=12)
    # Caption: the accuracy<->compute tradeoff story.
    caption = (
        "Larger $r$ drives AGE toward the Conv floor (better gradient) but "
        "costs more compute and shrinks the max batch within "
        f"{budget_gb} GB (annotated)."
    )
    if infeasible_note:
        caption += " " + infeasible_note
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9)
    return save_figure(fig, plot_dir, stem, dpi), (y_lo, y_hi)


def plot_individual_panels(model_name, data, img_dims, rs_all, plot_dir,
                           stem_prefix, dpi, y_margin_frac, precision,
                           y_lim=None):
    """One standalone single-panel figure PER image dimension (same panel as a
    cell of the composite), written under ``plot_dir/individual/``. ``y_lim`` is
    the model-shared (y_lo, y_hi) from the composite so every individual figure
    shares the composite's y range; if None it is recomputed from ``data``.
    Image dims absent from ``data`` (none collected) are skipped. Returns the
    list of written file paths."""
    written = []
    for img in img_dims:
        if img not in data or not data[img].get("xconv"):
            continue  # nothing probed yet for this dim (partial sweep)
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_panel(ax, img, data[img], rs_all, show_legend=True)
        ax.set_ylim(*(y_lim or shared_log_ylim(data, y_margin_frac)))
        ax.set_ylabel("Average Gradient Error")
        written += save_figure(
            fig, os.path.join(plot_dir, "individual"),
            f"{stem_prefix}_img{img}_{precision}", dpi,
        )
    return written


def write_rows_csv(plot_dir, precision, model_rows, name_tag=""):
    """Tidy AGE CSV across the models that ran.

    ``model_rows`` is a list of ``(model_label, rows)`` where each ``rows`` is the
    flat row list returned by the collectors -- tuples
    ``(series, img, r, mean, std, batch, runs)``. ``series`` is "Conv", an int
    rank (an XConv point), or a "RAD-* (...)" reference-line label. Columns:
    ``model,precision,image_dim,series,r,batch,age_mean,age_std,runs``.
    ``name_tag`` is appended to the filename for a non-16 budget (e.g.
    "_60gb") so it never overwrites the committed ``age_vs_r_{prec}.csv``; it is
    "" at 16 GB, keeping that filename byte-identical to the committed artifact.
    Returns the written CSV path."""
    os.makedirs(plot_dir, exist_ok=True)
    csv_path = os.path.join(plot_dir, f"age_vs_r_{precision}{name_tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "precision", "image_dim", "series", "r", "batch",
                    "age_mean", "age_std", "runs"])
        for model_label, rows in model_rows:
            for series, img, r, mean, std, batch, runs in rows:
                if series == "Conv":
                    series_name, r_cell = "Conv", ""
                elif isinstance(series, str) and series.startswith("RAD"):
                    series_name, r_cell = series.split(" (")[0], ""
                else:  # XConv point: ``series`` is the int rank, == r
                    series_name, r_cell = "XConv", r
                runs_cell = len(runs) if runs else ""
                w.writerow([model_label, precision, img, series_name, r_cell,
                            batch, f"{mean:.10g}", f"{std:.10g}", runs_cell])
    return os.path.abspath(csv_path)


def print_table(model_name, rows):
    print(f"\n=== {model_name}: AGE mean +/- sigma + batch (per img, per r) ===")
    header = (f"{'series':>16} {'img':>5} {'r':>5} {'batch':>7} "
              f"{'mean':>12} {'sigma':>12}  runs")
    print(header)
    print("-" * len(header))
    for series, img, r, mean, std, batch, runs in rows:
        run_str = ("[" + ", ".join(f"{v:.4g}" for v in runs.values()) + "]"
                   if runs else "[ckpt mean/std]")
        print(f"{series:>16} {img:>5} {str(r):>5} {batch:>7} "
              f"{mean:>12.6g} {std:>12.6g}  {run_str}")


def check_monotonic(model_name, data, rs_all):
    """Sanity check: does the XConv AGE curve fall monotonically toward the Conv
    floor as r grows, and stay above it? Print a per-image-dim verdict."""
    print(f"\n=== {model_name}: monotonic-descent-toward-Conv check ===")
    for img, d in data.items():
        conv = d["conv"]["mean"]
        rs = [r for r in rs_all if r in d["xconv"]]
        means = [d["xconv"][r]["mean"] for r in rs]
        if not means:
            print(f"  img {img:>4}: (no XConv points on disk yet)")
            continue
        decreasing = all(b <= a + 1e-15 for a, b in zip(means, means[1:]))
        above_floor = all(m >= conv for m in means)
        ratio_top = means[0] / conv if conv > 0 else float("inf")
        ratio_bot = means[-1] / conv if conv > 0 else float("inf")
        print(
            f"  img {img:>4}: r={rs} means="
            f"[{', '.join(f'{m:.3g}' for m in means)}]  Conv={conv:.3g}  "
            f"| monotone-down={decreasing} above-Conv={above_floor} "
            f"| AGE(r_min)/Conv={ratio_top:.1f}x -> AGE(r_max)/Conv={ratio_bot:.1f}x"
        )


# --------------------------------------------------------------------------- #
# Per-model drivers
# --------------------------------------------------------------------------- #
def run_squeezenet(args):
    err_dir, config_dir = squeezenet_cfg(args.precision, args.budget_gb)
    data, rows = collect_curves_from_pickles(
        err_dir, config_dir, SQ_IMG_DIMS, SQ_RS
    )
    y_lim = shared_log_ylim(data, args.y_margin_frac)
    written, ylim = plot_model_figure(
        SQ_NAME, data, SQ_IMG_DIMS, SQ_RS, args.plot_dir,
        f"squeezenet_age_vs_r_{args.precision}", args.dpi, args.y_margin_frac,
        budget_gb=args.budget_gb,
    )
    written += plot_individual_panels(
        SQ_NAME, data, SQ_IMG_DIMS, SQ_RS, args.plot_dir,
        stem_prefix="squeezenet_age_vs_r", dpi=args.dpi,
        y_margin_frac=args.y_margin_frac, precision=args.precision, y_lim=y_lim,
    )
    print_table(SQ_NAME, rows)
    check_monotonic(SQ_NAME, data, SQ_RS)
    print(f"  shared log-y: [{ylim[0]:.4g}, {ylim[1]:.4g}]")
    return written, rows


def run_vanillanet(args):
    err_dir, config_dir = vanillanet_cfg(args.precision, args.budget_gb)
    data, rows = collect_curves_from_pickles(
        err_dir, config_dir, VN_IMG_DIMS, VN_RS
    )
    y_lim = shared_log_ylim(data, args.y_margin_frac)
    written, ylim = plot_model_figure(
        VN_NAME, data, VN_IMG_DIMS, VN_RS, args.plot_dir,
        f"vanillanet_age_vs_r_{args.precision}", args.dpi, args.y_margin_frac,
        budget_gb=args.budget_gb,
    )
    written += plot_individual_panels(
        VN_NAME, data, VN_IMG_DIMS, VN_RS, args.plot_dir,
        stem_prefix="vanillanet_age_vs_r", dpi=args.dpi,
        y_margin_frac=args.y_margin_frac, precision=args.precision, y_lim=y_lim,
    )
    print_table(VN_NAME, rows)
    check_monotonic(VN_NAME, data, VN_RS)
    print(f"  shared log-y: [{ylim[0]:.4g}, {ylim[1]:.4g}]")
    return written, rows


def run_unet(args):
    # Honor an explicit --unet_ckpt verbatim; otherwise derive from --precision
    # and --budget_gb (default literal is fp32 / 16 GB).
    ckpt = (args.unet_ckpt if args.unet_ckpt != UNET_CKPT
            else unet_ckpt_for(args.precision, args.budget_gb))
    data, rows, img_dims, rs = collect_curves_from_unet_ckpt(ckpt)
    # Note any image dim where RAD-RP is infeasible (dropped from that panel).
    rp_infeasible = [img for img, d in data.items()
                     if "RAD-RP (keep=0.1)" not in d.get("refs", {})]
    note = ""
    if rp_infeasible:
        note = ("RAD-RP (keep=0.1) is infeasible (no batch fits the budget) at "
                f"image dim {', '.join(str(i) for i in rp_infeasible)} and is "
                "omitted there.")
    y_lim = shared_log_ylim(data, args.y_margin_frac)
    written, ylim = plot_model_figure(
        UNET_NAME, data, img_dims, rs, args.plot_dir,
        f"unet_age_vs_r_{args.precision}", args.dpi, args.y_margin_frac,
        infeasible_note=note, budget_gb=args.budget_gb,
    )
    written += plot_individual_panels(
        UNET_NAME, data, img_dims, rs, args.plot_dir,
        stem_prefix="unet_age_vs_r", dpi=args.dpi,
        y_margin_frac=args.y_margin_frac, precision=args.precision, y_lim=y_lim,
    )
    print_table(UNET_NAME, rows)
    check_monotonic(UNET_NAME, data, rs)
    print(f"  shared log-y: [{ylim[0]:.4g}, {ylim[1]:.4g}]")
    if note:
        print(f"  note: {note}")
    return written, rows


def main():
    args = parse_args()
    if args.plot_dir is None:
        # Default plot dir is budget-tagged: DEFAULT_PLOT_DIR carries the "16gb"
        # token, which we rewrite to f"{budget_gb}gb" so a non-16 run lands in
        # plots/age_vs_r_{budget_gb}gb[_fp16] and never overwrites the committed
        # 16 GB figures. At 16 the token is unchanged -> identical path.
        base = DEFAULT_PLOT_DIR.replace("16gb", _tag(args.budget_gb))
        args.plot_dir = base + _suffix(args.precision)
    apply_publication_style()

    written = []
    model_rows = []
    if "squeezenet" in args.models:
        w, rows = run_squeezenet(args)
        written += w
        model_rows.append((SQ_NAME, rows))
    if "vanillanet" in args.models:
        w, rows = run_vanillanet(args)
        written += w
        model_rows.append((VN_NAME, rows))
    if "unet" in args.models:
        w, rows = run_unet(args)
        written += w
        model_rows.append((UNET_NAME, rows))

    if args.csv:
        # Tag the CSV filename for a non-16 budget too (belt-and-suspenders: the
        # default plot_dir is already tagged, but an explicit un-tagged
        # --plot_dir at a non-16 budget must still not clobber age_vs_r_*.csv).
        csv_path = write_rows_csv(
            args.plot_dir, args.precision, model_rows,
            name_tag=_plot_dir_tag_suffix(args.budget_gb),
        )
        written.append(csv_path)

    print("\n=== Files written ===")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    main()
