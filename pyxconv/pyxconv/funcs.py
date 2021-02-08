import torch
import torch.nn.functional as F

import opt_einsum as oe
from typing import List

from pyxconv.utils import *


#@torch.jit.script
def back_probe(seed: int, N: int, ci: int, co: int, b: int, ps: int, nw: int,
              offs: List[int], grad_output, eX):
    """
    Backward pass of probing-based convolution filter gradient.

    Arguments:
        seed (int): Random seed for probing vectors
        N (int): Number of pixels
        ci (int): Number of input channels
        co (int): Number of output channels
        b (int): Batch size
        ps (int): Number of probing vectors
        nw (int): Convolution filter width (nw in each direction)
        offs (List): List of offsets for the convolution filter coefficients
        grad_output (Tensor): Backward input
        eX (Tensor): Forward pass probed Tensor (b x ps)

    Returns:
        gradient w.r.t convolution filter
    """
    # Redraw e
    torch.random.manual_seed(seed)
    e = torch.randn(ci, N, ps, device=eX.device)

    # Y' X e
    Ye = grad_output.reshape(b, -1)
    LRe = torch.mm(eX.t(), Ye)

    # Init gradien
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)

    # Loop over offsets (can be improved later on)
    se = offs[-1]
    eend = N - se
    # LRE as ncho x (N x ps)
    LRe = LRe.reshape(ps, co, N).narrow(2, se, eend-se)
    for i, o in enumerate(offs):
        ev = e.narrow(1, se+o, eend-se)
        grad_weight[:, :, i] = oe.contract('bjk, lkb -> jl', LRe, ev)
    return grad_weight/ps


@torch.jit.script
def fwd_probe(ps: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.
    
    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor

    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.reshape(X.shape[0], -1)
    e = torch.randn(ci, N, ps, device=X.device).view(ci*N, ps)
    eX = torch.mm(Xv, e)
    return eX


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        torch.random.manual_seed(seed)
        b, ci, nx, ny = input.shape
        eX = fwd_probe(ps, ci, nx*ny, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.no_grad():
            Y = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, groups=groups)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.nn.grad.conv2d_input(ctx.xshape, weight,
                                                    grad_output,
                                                    stride=ctx.stride,
                                                    padding=ctx.padding,
                                                    dilation=ctx.dilation,
                                                    groups=ctx.groups)

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            _, ps = eX.shape
            b, ci, nx, ny = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets2d((nx, ny), nw)
            delta = dilate2d(grad_output, co, (nx, ny), b, ctx.stride)
            dw = back_probe(seed, nx*ny, ci, co, b, ps, nw**2, offs, delta, eX)
            dw = dw.reshape(co, ci, nw, nw)

        return dx, dw, None, None, None, None, None, None


class Xconv3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        torch.random.manual_seed(seed)
        b, ci, nx, ny, nz = input.shape
        eX = fwd_probe(ps, ci, nx*ny*nz, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.no_grad():
            Y = F.conv3d(input, weight, bias=bias, stride=stride, padding=padding, groups=groups)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.nn.grad.conv3d_input(ctx.xshape, weight,
                                                    grad_output,
                                                    stride=ctx.stride,
                                                    padding=ctx.padding,
                                                    dilation=ctx.dilation,
                                                    groups=ctx.groups)

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            _, ps = eX.shape
            b, ci, nx, ny, nz = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets3d((nx, ny, nz), nw)
            delta = dilate3d(grad_output, co, (nx, ny, nz), b, ctx.stride)
            dw = back_probe(seed, nx*ny*nz, ci, co, b, ps, nw**3, offs, delta, eX)
            dw = dw.reshape(co, ci, nw, nw, nw)

        return dx, dw, None, None, None, None, None, None
