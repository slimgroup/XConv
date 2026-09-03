"""Registry of the methods being compared and their compression knobs.

Each ``MethodSpec`` pairs a builder with the knob it sweeps. XConv sweeps the
number of probing vectors ``r``; RAD sweeps ``keep_frac``. The knobs live on
different axes, which is why the headline figure is AGE-vs-peak-memory (a shared
axis) while the per-knob panels are grouped by family. Colours follow a fixed
semantic palette (exact = black reference; one hue per estimator family).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pyxconv.radcompare.networks import build_exact, build_rad, build_xconv

__all__ = ["MethodSpec", "make_specs"]


@dataclass
class MethodSpec:
    key: str  # stable identifier, e.g. "xconv_independent"
    label: str  # legend label, e.g. "XConv (independent)"
    family: str  # "exact" | "xconv" | "rad"
    color: str
    knob_name: str  # "r" | "keep_frac" | ""
    knob_values: list  # values to sweep ([None] for the exact baseline)
    build: Callable  # build(knob) -> CIFARConvNet


def make_specs(ps_sweep: list[int], keep_frac_sweep: list[float]) -> list[MethodSpec]:
    """Build the comparison set: exact baseline, XConv (independent probing), and
    two RAD variants (random projection + sampling).

    Only the 'independent' XConv mode is compared (per the current scope). The
    'gaussian'/'multi' and 'orthogonal'/'multi-ortho' modes remain available via
    ``build_xconv`` and can be added back as extra MethodSpecs if needed.
    """
    return [
        MethodSpec(
            "exact", "Exact conv", "exact", "black", "", [None],
            lambda knob: build_exact(),
        ),
        MethodSpec(
            "xconv_independent", "XConv (independent)", "xconv", "#1f77b4",
            "r", list(ps_sweep),
            lambda knob: build_xconv(knob, "independent"),
        ),
        MethodSpec(
            "rad_rp", "RAD (random proj.)", "rad", "#9467bd",
            "keep_frac", list(keep_frac_sweep),
            lambda knob: build_rad(knob, sparse=False),
        ),
        MethodSpec(
            "rad_sample", "RAD (sampling)", "rad", "#8c564b",
            "keep_frac", list(keep_frac_sweep),
            lambda knob: build_rad(knob, sparse=True, full_random=False),
        ),
    ]
