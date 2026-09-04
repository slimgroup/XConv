"""Publication-grade figures for the RAD-vs-XConv comparison.

Three figures (see recent_style.md §14): a peak-memory panel and an AGE panel,
each split into an XConv sub-panel (vs probing vectors r) and a RAD sub-panel
(vs keep_frac) to match the per-knob figures of the XConv paper; and the
headline AGE-vs-peak-memory tradeoff, where every method shares one axis and the
exact baseline is a gold star. Despined axes, a fixed semantic palette (carried
on each record), dashed reference lines, vector-PDF output.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = [
    "apply_paper_style",
    "plot_memory_panels",
    "plot_age_panels",
    "plot_tradeoff",
    "plot_age_vs_imgdim",
    "plot_memory_vs_imgdim",
    "plot_age_vs_imgdim_with_batches",
    "plot_maxbatch_vs_imgdim",
    "plot_peak_vs_probing",
    "plot_peak_curve",
]

GOLD = "#FFD700"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig, save_path: str) -> None:
    """Save a publication figure as vector PDF and a PNG preview."""
    base, _ = os.path.splitext(save_path)
    for path in (f"{base}.pdf", f"{base}.png"):
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def _exact(records: list[dict]) -> dict | None:
    for r in records:
        if r["family"] == "exact":
            return r
    return None


def _series(records: list[dict], family: str) -> dict[str, list[dict]]:
    """Group a family's records by series key, sorted by knob."""
    out: dict[str, list[dict]] = {}
    for r in records:
        if r["family"] == family:
            out.setdefault(r["key"], []).append(r)
    for key in out:
        out[key].sort(key=lambda r: r["knob"])
    return out


def plot_memory_panels(memory_records: list[dict], save_path: str) -> None:
    exact = _exact(memory_records)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, family, xlabel, logx in (
        (axes[0], "xconv", r"Probing vectors $r$", True),
        (axes[1], "rad", "keep_frac", False),
    ):
        for key, recs in _series(memory_records, family).items():
            xs = [r["knob"] for r in recs]
            ys = [r["peak_mib"] for r in recs]
            ax.plot(xs, ys, "-o", color=recs[0]["color"], lw=1.6, ms=5,
                    label=recs[0]["label"])
        if exact is not None:
            ax.axhline(exact["peak_mib"], ls="--", color="black", lw=1.2,
                       alpha=0.7, label=exact["label"])
        if logx:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Peak memory (MiB)")
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        _despine(ax)
    axes[0].set_title("(a) XConv")
    axes[1].set_title("(b) RAD")
    fig.tight_layout()
    _save(fig, save_path)


def plot_age_panels(age_records: list[dict], save_path: str) -> None:
    exact = _exact(age_records)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for ax, family, xlabel, logx in (
        (axes[0], "xconv", r"Probing vectors $r$", True),
        (axes[1], "rad", "keep_frac", False),
    ):
        for key, recs in _series(age_records, family).items():
            xs = [r["knob"] for r in recs]
            ys = [r["age_mean"] for r in recs]
            lo = [r["age_mean"] - r["age_std"] for r in recs]
            hi = [r["age_mean"] + r["age_std"] for r in recs]
            ax.plot(xs, ys, "-o", color=recs[0]["color"], lw=1.6, ms=5,
                    label=recs[0]["label"])
            ax.fill_between(xs, lo, hi, color=recs[0]["color"], alpha=0.15)
        if exact is not None:
            ax.axhline(exact["age_mean"], ls="--", color="black", lw=1.2,
                       alpha=0.7, label=f"{exact['label']} (sampling floor)")
        if logx:
            ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Average gradient error")
        ax.legend(frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        _despine(ax)
    axes[0].set_title("(a) XConv")
    axes[1].set_title("(b) RAD")
    fig.tight_layout()
    _save(fig, save_path)


def plot_age_vs_imgdim(results: dict, save_path: str, title: str | None = None) -> None:
    """AGE vs image dimension, in the repo's AGE-figure theme (Figs 4-6): mean
    line with +/- sigma band per method, log-y, full box + grid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results.values():
        x = r["img_dims"]
        mean = r["age_mean"]
        std = r["age_std"]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.plot(x, mean, "-o", lw=2, ms=6, color=r["color"], label=r["label"],
                zorder=3)
        ax.fill_between(x, lo, hi, color=r["color"], alpha=0.2, zorder=2)
    ax.set_xlabel("Image dimension")
    ax.set_ylabel("Average gradient error")
    ax.set_yscale("log", base=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)


def plot_memory_vs_imgdim(results: dict, save_path: str, title: str | None = None) -> None:
    """Peak memory vs image dimension, repo theme; log-y memory, full box + grid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results.values():
        ax.plot(r["img_dims"], r["peak_mib"], "-o", lw=2, ms=6, color=r["color"],
                label=r["label"], zorder=3)
    ax.set_xlabel("Image dimension")
    ax.set_ylabel("Peak memory (MiB)")
    ax.set_yscale("log", base=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)


def plot_age_vs_imgdim_with_batches(results: dict, save_path: str, title: str | None = None) -> None:
    """AGE vs image dimension, the paper's Fig-4/5 protocol: each method at the
    maximum batch that fits a fixed memory budget (so AGE rises with resolution
    as the batch shrinks). The max batch is annotated on each point, matching the
    repo's AGE figures. NaN points (infeasible: no batch fits) leave a gap.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results.values():
        mean = r["age_mean"]
        std = r["age_std"]
        # Incremental re-plot: age_mean/std/batch grow one entry per finished
        # resolution, while img_dims is the full list; truncate x to the data so
        # partial figures render the completed resolutions (and matplotlib does
        # not choke on a length mismatch).
        x = r["img_dims"][:len(mean)]
        lo = [m - s for m, s in zip(mean, std)]
        hi = [m + s for m, s in zip(mean, std)]
        ax.plot(x, mean, "-o", lw=2, ms=6, color=r["color"], label=r["label"], zorder=3)
        ax.fill_between(x, lo, hi, color=r["color"], alpha=0.15, zorder=2)
        for xi, mi, bi in zip(x, mean, r["batch"]):
            if mi == mi and bi:  # finite AGE and batch > 0
                ax.annotate(str(int(bi)), xy=(xi, mi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7, color=r["color"])
    ax.set_xlabel("Image dimension")
    ax.set_ylabel("Average gradient error")
    ax.set_yscale("log", base=10)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)


def plot_maxbatch_vs_imgdim(results: dict, save_path: str, title: str | None = None) -> None:
    """Max batch that fits the memory budget vs image dimension, per method --
    the direct view of XConv's batch-size advantage (and infeasibility of exact/
    RAD at high resolution, where the batch falls to zero / NaN)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results.values():
        batch = [b if b else float("nan") for b in r["batch"]]
        # Truncate to measured resolutions for incremental re-plot (see
        # plot_age_vs_imgdim_with_batches).
        x = r["img_dims"][:len(batch)]
        ax.plot(x, batch, "-o", lw=2, ms=6, color=r["color"], label=r["label"], zorder=3)
    ax.set_xlabel("Image dimension")
    ax.set_ylabel("Max batch size within memory budget")
    ax.set_yscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_path)


def plot_peak_curve(results: dict, image_dim: int, budget_mib: float, save_path: str,
                    title: str | None = None) -> None:
    """Peak memory vs batch size at a fixed image dimension, per method, with the
    memory budget as a dashed line -- the curve used to find each method's max
    batch (where its curve meets the budget). NaN (OOM) probes are dropped."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results.values():
        probes = r.get("peak_curve", {}).get(image_dim, [])
        pts = sorted((b, p) for b, p in probes if p == p)  # drop NaN (OOM)
        if not pts:
            continue
        xs = [b for b, _ in pts]
        ys = [p / 1024 for _, p in pts]
        ax.loglog(xs, ys, "-o", color=r["color"], lw=1.6, ms=4, label=r["label"])
    ax.axhline(budget_mib / 1024, ls="--", color="black", lw=1.3, alpha=0.8,
               label="memory budget")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak memory (GB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title or f"Peak memory vs batch (Img-Size={image_dim})")
    fig.tight_layout()
    _save(fig, save_path)


def plot_peak_vs_probing(records: list[dict], image_dim: int, save_path: str,
                         title: str | None = None) -> None:
    """Peak memory vs probing vectors at a fixed image dimension, one line per
    batch size, with the exact-conv ('base') memory as a dashed reference per
    batch -- the paper's Fig-8 peak-memory curves. NaN (OOM) leaves gaps."""
    recs = [r for r in records if r["img_dim"] == image_dim]
    batches = sorted({r["batch"] for r in recs})
    pvs = sorted({r["pv"] for r in recs if r["pv"] != "base"}, key=lambda p: int(p))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.viridis
    for i, batch in enumerate(batches):
        color = cmap(i / max(1, len(batches) - 1))
        ys = []
        for pv in pvs:
            match = [r["peak_mib"] for r in recs if r["batch"] == batch and r["pv"] == pv]
            ys.append(match[0] / 1024 if match else float("nan"))
        if any(y == y for y in ys):
            ax.loglog([int(p) for p in pvs], ys, "-o", color=color, lw=1.6, ms=4,
                      label=f"B={batch}")
        base = [r["peak_mib"] for r in recs if r["batch"] == batch and r["pv"] == "base"]
        if base and base[0] == base[0]:
            ax.axhline(base[0] / 1024, ls="--", color=color, lw=1.0, alpha=0.6)
    ax.set_xlabel("Probing vectors")
    ax.set_ylabel("Peak memory (GB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title(title or f"SqueezeNet peak memory (Img-Size={image_dim})")
    fig.tight_layout()
    _save(fig, save_path)


def plot_tradeoff(combined_records: list[dict], save_path: str) -> None:
    """AGE vs peak memory; lower-left is better. Exact baseline = gold star."""
    exact = _exact(combined_records)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for family in ("xconv", "rad"):
        for key, recs in _series(combined_records, family).items():
            recs = sorted(recs, key=lambda r: r["peak_mib"])
            xs = [r["peak_mib"] for r in recs]
            ys = [r["age_mean"] for r in recs]
            ax.plot(xs, ys, "-o", color=recs[0]["color"], lw=1.6, ms=5,
                    label=recs[0]["label"])
    if exact is not None:
        ax.plot(exact["peak_mib"], exact["age_mean"], marker="*", ms=18,
                color=GOLD, markeredgecolor="black", linestyle="none",
                label=exact["label"], zorder=5)
    ax.set_yscale("log")
    ax.set_xlabel("Peak memory (MiB)")
    ax.set_ylabel("Average gradient error")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.3)
    _despine(ax)
    fig.tight_layout()
    _save(fig, save_path)
