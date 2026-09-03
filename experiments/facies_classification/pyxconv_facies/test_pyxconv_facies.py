"""Validation suite for pyxconv_facies (run from repo root with PYTHONPATH=.)."""
import copy
import sys

import torch
import torch.nn as nn
from torch.autograd import gradcheck

from pyxconv_facies.modules import Xconv2D, XconvTranspose2D
from pyxconv_facies.utils import adaptive_convert_facies, align_grad_output_to_input_spatial, dilate2d
from pyxconv_facies.facies_convert import apply_xconv_to_facies_model, count_adaptive_xconv_layers


def _require_cuda():
    if not torch.cuda.is_available():
        print('SKIP: CUDA not available')
        sys.exit(0)


def test_transpose_gradcheck():
    for kw in (
        dict(ps=8, stride=1, padding=1),
        dict(ps=16, stride=2, padding=1),
        dict(ps=8, stride=1, padding=0, dilation=2),
    ):
        layer = XconvTranspose2D(4, 8, 3, bias=True, mode='independent', **kw).double().cuda()
        x = torch.randn(2, 4, 16, 16, dtype=torch.double, device='cuda', requires_grad=True)
        assert gradcheck(layer, (x,), eps=1e-4, atol=1e-3, rtol=1e-2, fast_mode=False)
        print('OK gradcheck', kw)


def test_no_double_conversion():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Conv2d(3, 8, 3, padding=1)
            self.dec = nn.ConvTranspose2d(8, 4, 3, padding=1)

        def forward(self, x):
            return self.dec(self.enc(x))

    m = Tiny().cuda()
    x = torch.randn(1, 3, 32, 32, device='cuda')
    m1 = copy.deepcopy(m)
    apply_xconv_to_facies_model(m1, x, probing_vector=8)
    n1 = sum(1 for mod in m1.modules() if isinstance(mod, (Xconv2D, XconvTranspose2D)))
    m2 = copy.deepcopy(m1)
    apply_xconv_to_facies_model(m2, x, probing_vector=8)
    n2 = sum(1 for mod in m2.modules() if isinstance(mod, (Xconv2D, XconvTranspose2D)))
    assert n1 == n2 >= 2
    print('OK idempotent conversion')


def test_patch_deconvnet_counts():
    from core.models import get_model

    x = torch.randn(1, 1, 99, 99, device='cuda')
    m = get_model('patch_deconvnet', False, 6).cuda().eval()
    n2, t2, ntr, ttr, pct = count_adaptive_xconv_layers(m, x, ps=16)
    assert t2 > 0 and ttr > 0
    m2 = get_model('patch_deconvnet', False, 6).cuda().eval()
    apply_xconv_to_facies_model(m2, x, probing_vector=16)
    nx = sum(1 for mod in m2.modules() if isinstance(mod, Xconv2D))
    nt = sum(1 for mod in m2.modules() if isinstance(mod, XconvTranspose2D))
    print(f'OK patch_deconvnet ps=16: {pct:.0f}% ({n2+ntr}/{t2+ttr}), Xconv2D={nx}, XconvTranspose2D={nt}')


def main():
    _require_cuda()
    test_transpose_gradcheck()
    test_no_double_conversion()
    test_patch_deconvnet_counts()
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
