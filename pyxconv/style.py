"""Shared plotting style, so every figure in the paper reads as one system.

One concept, one colour: a probing-vector count means the same colour wherever it
appears, and the exact gradient is always the dashed reference. Figures are sized to
the manuscript's text width so no figure is rescaled on inclusion.
"""

from __future__ import annotations

import matplotlib as mpl

# The manuscript's \linewidth, in inches.
LINEWIDTH_IN = 5.125

PALETTE = {
    "exact": "#444444",      # the true gradient, always the dashed reference
    "xconv": "#c2410c",      # XConv
    "rad": "#0369a1",        # randomized automatic differentiation
    "fp32": "#c2410c",
    "fp16": "#0891b2",
    "budget": "#71717a",     # a memory budget drawn as a limit line
}


def apply_paper_style() -> None:
    """Set the rcParams the paper's figures were rendered with."""
    mpl.rcParams.update({
        "figure.figsize": (LINEWIDTH_IN, LINEWIDTH_IN * 0.62),
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })
