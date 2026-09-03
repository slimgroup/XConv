"""
Facies-specific XConv conversion on top of the local pyxconv_facies copy.

Source: xconv_pv @ branch ali (see README). Do not edit site-packages pyxconv.
"""

from typing import Dict, Tuple

import torch.nn as nn

from .modules import Xconv2D, XconvTranspose2D
from .utils import adaptive_convert_facies, _collect_conv_input_spatial

# Skip fc6/fc7 (4096 channels) on patch_deconvnet.
FACIES_XCONV_MAX_CHANNELS = 512

SOURCE_REPO = 'xconv_pv'
SOURCE_BRANCH = 'ali'


def _layer_convertible(
    child: nn.Module,
    full_name: str,
    in_spatial: Dict[str, Tuple[int, int]],
    ps: int,
    maxc: int,
) -> bool:
    hw = in_spatial.get(full_name)
    if hw is None:
        return False
    h, w = hw
    return (
        (h * w) > ps
        and child.in_channels < maxc
        and child.out_channels < maxc
    )


def count_adaptive_xconv_layers(
    module,
    sample_input,
    ps: int,
    maxc: int = FACIES_XCONV_MAX_CHANNELS,
    mode: str = 'conv',
):
    """
    Count Conv2d and ConvTranspose2d layers adaptive_convert_facies would replace.

    Returns:
        (n_conv2d_converted, n_conv2d_total,
         n_transpose_converted, n_transpose_total,
         percent_all_converted)
    """
    hook_types = (nn.Conv2d, nn.ConvTranspose2d)
    in_spatial = _collect_conv_input_spatial(module, sample_input, hook_types)

    n_conv2d = n_transpose = 0
    n_conv2d_conv = n_transpose_conv = 0

    for name, child in module.named_modules():
        if isinstance(child, (Xconv2D, XconvTranspose2D)):
            continue
        if isinstance(child, nn.Conv2d):
            n_conv2d += 1
            if mode in ('all', 'conv') and _layer_convertible(
                child, name, in_spatial, ps, maxc
            ):
                n_conv2d_conv += 1
        elif isinstance(child, nn.ConvTranspose2d):
            n_transpose += 1
            if mode in ('all', 'conv') and _layer_convertible(
                child, name, in_spatial, ps, maxc
            ):
                n_transpose_conv += 1

    n_total = n_conv2d + n_transpose
    n_converted = n_conv2d_conv + n_transpose_conv
    pct = 100.0 * n_converted / n_total if n_total else 0.0
    return (
        n_conv2d_conv,
        n_conv2d,
        n_transpose_conv,
        n_transpose,
        pct,
    )


def count_adaptive_xconv_conv2d_layers(
    module,
    sample_input,
    ps: int,
    maxc: int = FACIES_XCONV_MAX_CHANNELS,
    mode: str = 'conv',
):
    """Backward-compatible wrapper: Conv2d counts only."""
    n2, t2, ntr, ttr, _ = count_adaptive_xconv_layers(
        module, sample_input, ps, maxc, mode
    )
    pct = 100.0 * n2 / t2 if t2 else 0.0
    return n2, t2, pct


def format_conversion_title_suffix(
    probing_vectors,
    module,
    sample_input,
    maxc: int = FACIES_XCONV_MAX_CHANNELS,
):
    """Title fragment: % of Conv2d + ConvTranspose2d converted for each probing vector."""
    pcts = []
    counts = []
    for pv in probing_vectors:
        if pv == 'base':
            continue
        n2, t2, ntr, ttr, pct = count_adaptive_xconv_layers(
            module, sample_input, ps=int(pv), maxc=maxc
        )
        pcts.append(pct)
        counts.append((n2 + ntr, t2 + ttr))

    if not pcts:
        return ""

    n_total = counts[0][1]
    n_min = min(c[0] for c in counts)
    n_max = max(c[0] for c in counts)
    pct_min = min(pcts)
    pct_max = max(pcts)

    if pct_min == pct_max:
        return f"{pct_min:.0f}% Conv→XConv ({n_max}/{n_total})"
    return f"{pct_min:.0f}–{pct_max:.0f}% Conv→XConv ({n_min}–{n_max}/{n_total})"


def apply_xconv_to_facies_model(
    model,
    sample_input,
    probing_vector,
    xmode='independent',
):
    """
    Replace selected Conv2d and ConvTranspose2d layers with Xconv2D / XconvTranspose2D.

    Uses adaptive_convert_facies: spatial size > ps and channels < maxc (512).
    """
    ps = int(probing_vector)
    adaptive_convert_facies(
        model,
        sample_input,
        ps=ps,
        xmode=xmode,
        mode='conv',
        maxc=FACIES_XCONV_MAX_CHANNELS,
    )
    return model
