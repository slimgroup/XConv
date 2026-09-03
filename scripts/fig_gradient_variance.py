"""
Plot gradient standard deviation vs probe size r (Figure 4 style).

Loads `stdis_40.npy` produced by `gradient_variance_sweep.py` and writes a PDF with
line plots and a single shared legend. Re-run this script to tweak styling
without recomputing gradients.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

BATCHES = [64, 128, 256, 1024]
P_SIZES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
R_VALUES = np.array([r for r in P_SIZES if r > 0], dtype=float)


def load_std(path: Path) -> np.ndarray:
    std = np.load(path)
    expected = (len(P_SIZES), len(BATCHES), 4)
    if std.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {std.shape} from {path}")
    return std


def plot_figure(std: np.ndarray, out_path: Path, dpi: int = 200) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(BATCHES)))

    for i, ax in enumerate(axes.flat):
        for bb, batch in enumerate(BATCHES):
            y_r = std[1:, bb, i]
            ax.plot(
                R_VALUES,
                y_r,
                "o-",
                color=colors[bb],
                label=f"batch={batch}",
                markersize=5,
                linewidth=1.8,
            )
            ax.axhline(
                std[0, bb, i],
                color=colors[bb],
                linestyle=":",
                linewidth=1.6,
                alpha=0.85,
                zorder=4,
            )

        ax.set_ylabel("Std. dev. of grad / batch")
        ax.set_title(f"conv{i + 1}")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.25, linestyle="--")

    xticks = list(R_VALUES)
    xlabels = [str(int(r)) for r in R_VALUES]
    for ax in axes.flat:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_xlim(left=R_VALUES.min() / np.sqrt(2), right=R_VALUES.max() * np.sqrt(2))
    axes[1, 0].set_xlabel("Probe size $r$")
    axes[1, 1].set_xlabel("Probe size $r$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="0.35", linestyle=":", linewidth=1.6, label="True conv")
    )
    labels.append("True conv")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(BATCHES) + 1,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    print(f"Wrote {out_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=repo_root / "stdis_40.npy",
        help="Path to stdis_40.npy (default: repo root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "var_conv_40.pdf",
        help="Output PDF path (default: repo root)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(
            f"Missing {args.data}. Run scripts/figures/gradient_variance_sweep.py first "
            "or pass --data to an existing stdis_40.npy."
        )

    plot_figure(load_std(args.data), args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
