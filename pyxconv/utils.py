import torch
from typing import Tuple

__all__ = ['convert_net', 'dilate2d', 'dilate3d', 'offsets2d', 'offsets3d']


def convert_net(module, name='net', ps=16):
    """
    Recursively replaces all nn.Conv2d by XConv2D

    """
    from .modules import Xconv2D
    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d):
            print('replaced: ', name, child_name)
            newconv = Xconv2D(child.in_channels, child.out_channels,
                              child.kernel_size, ps=ps, stride=child.stride,
                              padding=child.padding)
            newconv.weight = child.weight
            setattr(module, child_name, newconv)
        else:
            convert_net(child, child_name)


@torch.jit.script
def dilate2d(y, co: int, N: Tuple[int, int], b: int, stride: Tuple[int, int]):
    sx, sy = stride
    if sx == 1 and sy == 1:
        return y
    yd = torch.zeros(b, co, *N, device=y.device)
    yd[:, :, ::sx, ::sy][:, :, :y.shape[2], :y.shape[3]] = y
    return yd


@torch.jit.script
def dilate3d(y, co: int, N: Tuple[int, int, int], b: int, stride: Tuple[int, int, int]):
    sx, sy, sz = stride
    if sx == 1 and sy == 1 and sz == 1:
        return y
    yd = torch.zeros(b, co, *N, device=y.device)
    yd[:, :, ::sx, ::sy, ::sz][:, :, :y.shape[2], :y.shape[3], :y.shape[4]] = y
    return yd


def offsets3d(N: Tuple[int, int, int], nw: int):
    nx, ny, nz = N
    r = torch.arange(-(nw//2), nw//2+1)
    offs = [i + j*nx + k*nx*ny for k in r for j in r for i in r]
    return offs


def offsets2d(N: Tuple[int, int], nw: int):
    nx, ny = N
    r = torch.arange(-(nw//2), nw//2+1)
    offs = [i + j*nx for j in r for i in r]
    return offs
