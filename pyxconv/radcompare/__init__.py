"""RAD (Oktay et al., 2021) vs XConv on the shared CIFAR-10 conv net.

Compares two unbiased low-memory weight-gradient estimators -- XConv (probing,
knob = number of probing vectors r) and RAD (spatial random projection /
sampling, knob = keep_frac) -- on peak memory and Average Gradient Error, on the
four-conv CIFAR-10 network common to both papers. See ``networks`` for the
architecture, ``memory``/``age`` for the metrics, ``sweeps`` for orchestration,
and ``plotting`` for the figures.
"""

from __future__ import annotations

from pyxconv.radcompare.networks import (
    CIFARConvNet,
    build_exact,
    build_rad,
    build_xconv,
    conv_weight_names,
)
from pyxconv.radcompare.estimators import MethodSpec, make_specs
from pyxconv.radcompare.memory import peak_memory_mib
from pyxconv.radcompare.age import average_gradient_error, exact_full_gradient
from pyxconv.radcompare.sweeps import (
    join_records,
    max_batch_for_budget,
    run_age_sweep,
    run_imgsize_sweep,
    run_memory_sweep,
    run_squeezenet_peak_memory_sweep,
    run_unet_age_vs_imgdim,
    run_unet_imgsize_sweep,
)
from pyxconv.radcompare.unet import (
    UNet,
    build_unet_exact,
    build_unet_rad,
    build_unet_xconv,
)
from pyxconv.radcompare import plotting

__all__ = [
    "CIFARConvNet",
    "build_exact",
    "build_xconv",
    "build_rad",
    "conv_weight_names",
    "UNet",
    "build_unet_exact",
    "build_unet_xconv",
    "build_unet_rad",
    "MethodSpec",
    "make_specs",
    "peak_memory_mib",
    "exact_full_gradient",
    "average_gradient_error",
    "run_memory_sweep",
    "run_age_sweep",
    "run_imgsize_sweep",
    "run_unet_imgsize_sweep",
    "max_batch_for_budget",
    "run_unet_age_vs_imgdim",
    "run_squeezenet_peak_memory_sweep",
    "join_records",
    "plotting",
]
