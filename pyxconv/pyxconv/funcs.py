import torch
import torch.nn.functional as F

from typing import List

from pyxconv.utils import *

@torch.jit.script
def back_probe(seed: int, N: int, ci: int, co: int,
               b: int, ps: int, nw: int, offs: List[int],
               grad_output, eX):
    # Redraw e
    torch.random.manual_seed(seed)
    e = torch.randn(ci, N, ps, device=eX.device).permute(0, 2, 1)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.T, Ye)

    # Init gradien
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)

    # Loop over offsets (can be improved later on)
    se = offs[-1]
    eend = N - se
    # LRE as ncho x (N x ps)
    LRe = torch.narrow(LRe.view(ps, co, N), 2, se, eend-se).permute(1, 0, 2).reshape(co, -1).t()
    for i, o in enumerate(offs):
        torch.mm(torch.narrow(e, 2, se+o, eend-se).reshape(ci, -1), LRe, out=grad_weight[:, :, i])
    return grad_weight.permute(1,0,2)/ps


@torch.jit.script
def fwd_probe(ps: int, X):
    Xv = X.view(X.shape[0], -1)
    e = torch.randn(Xv.shape[1], ps, device=X.device)
    eX = torch.mm(Xv, e)
    return eX


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        torch.random.manual_seed(seed)
        eX = fwd_probe(ps, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.no_grad():
            Y = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, groups=groups)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors
        nw = weight.shape[2]
        _, ps = eX.shape
        b, ci, nx, ny = ctx.xshape
        co = grad_output.shape[1]

        offs = offsets2d((nx, ny), nw)
        delta = dilate2d(grad_output, co, (nx, ny), b, ctx.stride)
        dw = back_probe(seed, nx*ny, ci, co, b, ps, nw**2, offs, delta, eX)

        return None, dw.reshape(co, ci, nw, nw), None, None, None, None, None, None


class Xconv3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        torch.random.manual_seed(seed)
        eX = fwd_probe(ps, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.no_grad():
            Y = F.conv3d(input, weight, bias=bias, stride=stride, padding=padding, groups=groups)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors
        nw = weight.shape[2]
        _, ps = eX.shape
        b, ci, nx, ny, nz = ctx.xshape
        co = grad_output.shape[1]

        offs = offsets3d((nx, ny, nz), nw)
        delta = dilate3d(grad_output, co, (nx, ny, nz), b, ctx.stride)
        dw = back_probe(seed, nx*ny*nz, ci, co, b, ps, nw**3, offs, delta, eX)
        return None, dw.reshape(co, ci, nw, nw, nw), None, None, None, None, None, None
