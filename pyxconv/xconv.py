import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt 

_pair = torch.nn.modules.utils._pair

@torch.jit.script
def dilate(y, co: int, nx: int, ny: int, b: int, stride: Tuple[int, int]):
    sx, sy = stride
    if sx == 1 and sy == 1:
        return y
    yd = torch.zeros(b, co, nx, ny)
    yd[:, :, ::sx, ::sy] = y
    return yd

@torch.jit.script
def back_probe(seed: int, nx: int, ny: int, ci: int, co: int,
               b: int, ps: int, nw: int,
               grad_output, eX):
    # Redraw e
    torch.random.manual_seed(seed)
    e = torch.randn(nx*ny*ci, ps).T

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.T, Ye)

    # Init gradien
    grad_weight = torch.zeros(co, ci, nw*nw)

    # reshape
    # e as N x nchi x ps
    e = e.view(ps, ci, nx*ny).permute(0, 2, 1)
    # Loop over offsets (can be improved later on)
    r = torch.arange(-(nw//2), nw//2+1)
    offs = torch.stack([r + i*nx for i in r]).reshape(-1)
    se = torch.max(offs)
    eend = nx*ny - se
    # LRE as ncho x N x ps
    LRe = LRe.view(ps, co, nx*ny)[:, :, se:eend]
    for i, o in enumerate(offs):
        # View shifted e
        ev = e[:, (se+o):(eend+o), :]
        bm = torch.bmm(LRe, ev)
        grad_weight[:, :, i] += torch.sum(bm, 0)
    return grad_weight.reshape(co, ci, nw, nw)/ps


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1):
        seed = torch.randint(100000, (1,))
        torch.random.manual_seed(seed)

        Xv = input.view(input.shape[0], -1)
        e = torch.randn(Xv.shape[1], ps)
        eX = torch.mm(Xv, e)

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

        delta = dilate(grad_output, co, nx, ny, b, ctx.stride)
        dw = back_probe(seed, nx, ny, ci, co, b, ps, nw, delta, eX)

        return None, dw, None, None, None, None, None, None
    
conv2d = Xconv2D.apply

class Xconv2D(torch.nn.modules.conv._ConvNd):
    def __init__(self, chi, cho, k, ps=8, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'):
        kernel_size = _pair(k)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Xconv2D, self).__init__(chi, cho, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)
        self.ps = ps

    def forward(self, input):       
        return conv2d(input, self.weight, self.ps, self.bias, self.stride, self.padding, self.dilation, self.groups)


if __name__ == "__main__":
    c = Xconv2D(4, 4, (3, 3), bias=False, ps=16, stride=2)
    c2 = nn.Conv2d(4, 4, (3, 3), bias=False, padding=1, stride=2)
 
    c2.weight = c.weight

    X = torch.randn(64, 4, 256, 256)
    y = c(X)
    loss = .5*torch.norm(y)**2
    grad_t = torch.autograd.grad(loss, c.parameters())
    y2 = c2(X)
    loss2 = .5*torch.norm(y2)**2
    grad_t2 = torch.autograd.grad(loss2, c2.parameters())

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        y = c(X)
        loss = .5*torch.norm(y)**2
        grad_t = torch.autograd.grad(loss, c.parameters())
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        y2 = c2(X)
        loss2 = .5*torch.norm(y2)**2
        grad_t2 = torch.autograd.grad(loss2, c2.parameters())
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


    plt.plot(grad_t[0].numpy().reshape(-1), label='ev')
    plt.plot(grad_t2[0].numpy().reshape(-1), label='torch')
    plt.legend()
    plt.show()