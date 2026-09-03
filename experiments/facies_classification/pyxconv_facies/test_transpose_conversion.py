"""
Exhaustive checks that ConvTranspose2d -> XconvTranspose2D conversion works.

Run from repo root:
  conda activate facies_classn_env
  PYTHONPATH=. python pyxconv_facies/test_transpose_conversion.py
"""
import copy
import sys

import torch
import torch.nn as nn

import core.loss
from core.models import get_model
from pyxconv.utils import adaptive_convert_net
from pyxconv_facies.modules import XconvTranspose2D
from pyxconv_facies.facies_convert import (
    apply_xconv_to_facies_model,
    count_adaptive_xconv_layers,
    FACIES_XCONV_MAX_CHANNELS,
)
from compute_mini_batch_gradients_avg_grad_err import apply_transpose_xconv_only


PATCH_INPUT = (1, 1, 99, 99)
PROBING_VECTORS = (2, 4, 8, 16, 32, 64, 128)


def _require_cuda():
    if not torch.cuda.is_available():
        print('SKIP: CUDA required')
        sys.exit(0)


def _is_plain_transpose(m):
    # XconvTranspose2D subclasses nn.ConvTranspose2d; use exact type.
    return type(m) is nn.ConvTranspose2d


def _count_layers(model):
    n_std_conv = sum(1 for m in model.modules() if type(m) is nn.Conv2d)
    n_std_tr = sum(1 for m in model.modules() if _is_plain_transpose(m))
    n_xtr = sum(1 for m in model.modules() if isinstance(m, XconvTranspose2D))
    return n_std_conv, n_std_tr, n_xtr


def _list_xconv_transpose_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, XconvTranspose2D)]


def _list_remaining_transpose(model):
    return [n for n, m in model.named_modules() if _is_plain_transpose(m)]


def test_count_predictions_match_conversion(ps):
    """count_adaptive_xconv_layers transpose prediction == actual after apply."""
    x = torch.randn(*PATCH_INPUT, device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    _, _, ntr_pred, ttr, _ = count_adaptive_xconv_layers(m, x, ps=ps)
    m2 = get_model('patch_deconvnet', False, 6).cuda().eval()
    apply_xconv_to_facies_model(m2, x, probing_vector=ps)
    ntr_actual = sum(1 for mod in m2.modules() if isinstance(mod, XconvTranspose2D))
    assert ntr_pred == ntr_actual, f'ps={ps}: predicted {ntr_pred} != actual {ntr_actual}'
    print(f'  OK ps={ps:3d}: count predicts {ntr_pred}/{ttr} transpose, got {ntr_actual} XconvTranspose2D')


def test_script_path_transpose_only(ps):
    """apply_transpose_xconv_only on fresh model (no Conv2d XConv)."""
    x = torch.randn(*PATCH_INPUT, device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    _, _, ntr_pred, ttr, _ = count_adaptive_xconv_layers(m, x, ps=ps)
    apply_transpose_xconv_only(m, x, probing_vector=ps)
    _, n_std_tr, n_xtr = _count_layers(m)
    assert n_xtr == ntr_pred, f'ps={ps}: script converted {n_xtr}, expected {ntr_pred}'
    assert n_std_tr == ttr - ntr_pred, (
        f'ps={ps}: {n_std_tr} standard transpose left, expected {ttr - ntr_pred}'
    )
    print(f'  OK ps={ps:3d}: script-only -> {n_xtr} XconvTranspose2D, {n_std_tr} nn.ConvTranspose2d left')


def test_script_path_full_pipeline(ps):
    """Production path: adaptive_convert_net + apply_transpose_xconv_only."""
    x = torch.randn(*PATCH_INPUT, device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    _, _, ntr_pred, _, _ = count_adaptive_xconv_layers(m, x, ps=ps)
    adaptive_convert_net(
        m, x, ps=ps, xmode='independent', mode='conv', maxc=FACIES_XCONV_MAX_CHANNELS
    )
    apply_transpose_xconv_only(m, x, probing_vector=ps)
    n_xtr = sum(1 for mod in m.modules() if isinstance(mod, XconvTranspose2D))
    assert n_xtr == ntr_pred
    print(f'  OK ps={ps:3d}: full pipeline -> {n_xtr} XconvTranspose2D')


def test_forward_backward_smoke(ps):
    """One forward + backward through converted patch_deconvnet."""
    x = torch.randn(2, *PATCH_INPUT[1:], device='cuda', requires_grad=False)
    labels = torch.randint(0, 6, (2, 1, 99, 99), device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().train()
    adaptive_convert_net(
        m, x[:1], ps=ps, xmode='independent', mode='conv', maxc=FACIES_XCONV_MAX_CHANNELS
    )
    apply_transpose_xconv_only(m, x[:1], probing_vector=ps)
    assert sum(1 for mod in m.modules() if isinstance(mod, XconvTranspose2D)) > 0
    out = m(x)
    loss = core.loss.cross_entropy_mean(out, labels)
    loss.backward()
    has_xtr_grad = any(
        isinstance(mod, XconvTranspose2D) and mod.weight.grad is not None
        for mod in m.modules()
    )
    assert has_xtr_grad, f'ps={ps}: no XconvTranspose2D weight grad'
    print(f'  OK ps={ps:3d}: forward+backward, XconvTranspose2D weight grad present')


def test_idempotent_transpose(ps):
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    x = torch.randn(*PATCH_INPUT, device='cuda')
    apply_transpose_xconv_only(m, x, probing_vector=ps)
    n1 = sum(1 for mod in m.modules() if isinstance(mod, XconvTranspose2D))
    names1 = _list_xconv_transpose_names(m)
    apply_transpose_xconv_only(m, x, probing_vector=ps)
    n2 = sum(1 for mod in m.modules() if isinstance(mod, XconvTranspose2D))
    names2 = _list_xconv_transpose_names(m)
    assert n1 == n2 and names1 == names2
    print(f'  OK ps={ps:3d}: idempotent ({n1} layers)')


def test_expected_layers_ps16():
    """At ps=16, print which transpose layers convert and which stay standard."""
    x = torch.randn(*PATCH_INPUT, device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    apply_transpose_xconv_only(m, x, probing_vector=16)
    converted = _list_xconv_transpose_names(m)
    remaining = _list_remaining_transpose(m)
    print(f'  Converted ({len(converted)}):')
    for n in converted:
        mod = dict(m.named_modules())[n]
        print(f'    {n}: in={mod.in_channels} out={mod.out_channels} ps={mod.ps}')
    print(f'  Remaining nn.ConvTranspose2d ({len(remaining)}):')
    for n in remaining:
        mod = dict(m.named_modules())[n]
        print(f'    {n}: in={mod.in_channels} out={mod.out_channels}')
    assert len(converted) == 7, f'expected 7 converted at ps=16, got {len(converted)}'
    assert len(remaining) == 7, (
        f'expected 7 remaining (4096-ch + 512-ch layers), got {len(remaining)}'
    )
    assert any(n.endswith('deconv_block8.0') for n in remaining)
    for n in remaining:
        mod = dict(m.named_modules())[n]
        assert (
            mod.in_channels >= FACIES_XCONV_MAX_CHANNELS
            or mod.out_channels >= FACIES_XCONV_MAX_CHANNELS
        ), f'{n} should stay standard due to channel limit'
    print(
        '  OK ps=16: 7 XconvTranspose2D + 7 standard ConvTranspose2d '
        '(4096-ch bottleneck + 512-ch deconv blocks)'
    )


def test_facies_vs_script_same_transpose_set(ps=16):
    """apply_xconv_to_facies_model and script path yield same transpose layer names."""
    x = torch.randn(*PATCH_INPUT, device='cuda')
    m1 = get_model('patch_deconvnet', False, 6).cuda().eval()
    apply_xconv_to_facies_model(m1, x, probing_vector=ps)
    names_facies = set(_list_xconv_transpose_names(m1))

    m2 = get_model('patch_deconvnet', False, 6).cuda().eval()
    adaptive_convert_net(
        m2, x, ps=ps, xmode='independent', mode='conv', maxc=FACIES_XCONV_MAX_CHANNELS
    )
    apply_transpose_xconv_only(m2, x, probing_vector=ps)
    names_script = set(_list_xconv_transpose_names(m2))

    assert names_facies == names_script, (
        f'mismatch facies {names_facies} vs script {names_script}'
    )
    print(f'  OK ps={ps}: facies and script paths convert same {len(names_facies)} transpose layers')


def main():
    _require_cuda()
    print('=== 1. Count predictions vs apply_xconv_to_facies_model ===')
    for ps in PROBING_VECTORS:
        test_count_predictions_match_conversion(ps)

    print('\n=== 2. Script-only apply_transpose_xconv_only ===')
    for ps in PROBING_VECTORS:
        test_script_path_transpose_only(ps)

    print('\n=== 3. Full pipeline (adaptive_convert_net + transpose script) ===')
    for ps in PROBING_VECTORS:
        test_script_path_full_pipeline(ps)

    print('\n=== 4. Forward + backward smoke ===')
    for ps in (16, 32, 128):
        test_forward_backward_smoke(ps)

    print('\n=== 5. Idempotent double apply_transpose_xconv_only ===')
    for ps in (16, 64):
        test_idempotent_transpose(ps)

    print('\n=== 6. Layer names at ps=16 ===')
    test_expected_layers_ps16()

    print('\n=== 7. Facies vs script path same transpose set ===')
    test_facies_vs_script_same_transpose_set(16)

    print('\nALL TRANSPOSE CONVERSION TESTS PASSED')


if __name__ == '__main__':
    main()
