import torch
import torch.nn.functional as F

from pyxconv.utils import *
from pyxconv.probe import *


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, mode='all', bias=None, stride=1, padding=0,
                dilation=1, groups=1, opu=None):
        seed = torch.randint(100000, (1,))
        b, ci, nx, ny = input.shape
        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = fwd_probe(mode, opu)(ps, b, ci, nx*ny, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.mode = mode
        ctx.ps = ps
        ctx.opu = opu

        with torch.autograd.grad_mode.no_grad():
            Y = F.conv2d(input, weight, bias=bias, stride=stride,
                         padding=padding, groups=groups)

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            b, ci, nx, ny = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets2d((nx, ny), nw)
            delta = dilate2d(grad_output, co, (nx, ny), b, ctx.stride)
            with random_seed_torch(int(seed)):
                with torch.autograd.grad_mode.no_grad():
                    dw = back_probe(ctx.mode, ctx.opu)(nx*ny, ci, co, b, ctx.ps,
                                              nw**2, offs, delta, eX)
                dw = dw.reshape(co, ci, nw, nw)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.nn.grad.conv2d_input(ctx.xshape, weight, grad_output,
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups)

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            db = grad_output.sum((0, 2, 3))

        return dx, dw, None, None, db, None, None, None, None


class Xconv3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, mode='all', bias=None, stride=1,
                padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        b, ci, nx, ny, nz = input.shape
        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = fwd_probe[mode](ps, b, ci, nx*ny*nz, input)

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.mode = mode
        ctx.ps = ps

        with torch.autograd.grad_mode.no_grad():
            Y = F.conv3d(input, weight, bias=bias, stride=stride,
                         padding=padding, groups=groups)

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            b, ci, nx, ny, nz = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets3d((nx, ny, nz), nw)
            delta = dilate3d(grad_output, co, (nx, ny, nz), b, ctx.stride)
            with random_seed_torch(int(seed)):
                dw = back_probe[ctx.mode](nx*ny*nz, ci, co, b, ctx.ps,
                                          nw**3, offs, delta, eX)
            dw = dw.reshape(co, ci, nw, nw, nw)

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.nn.grad.conv3d_input(ctx.xshape, weight, grad_output,
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups)

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            db = grad_output.sum((0, 2, 3, 4))

        return dx, dw, None, None, db, None, None, None, None


class Brelu(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, inplace=False):
        with torch.autograd.grad_mode.no_grad():
            Y = F.relu(input, inplace=inplace)
            sx = (Y > 0).byte()
        ctx.save_for_backward(sx)

        return Y

    @staticmethod
    def backward(ctx, grad_output):
        binp, = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            return grad_output*binp, None
        return None, None
