import torch
from typing import Tuple

__all__ = ['dilate2d', 'dilate3d', 'offsets2d', 'offsets3d']


@torch.jit.script
def dilate2d(y, co: int, N: Tuple[int, int], b: int, stride: Tuple[int, int]):
    sx, sy = stride
    if sx == 1 and sy == 1:
        return y
    yd = torch.zeros(b, co, *N, device=y.device)
    yd[:, :, ::sx, ::sy] = y
    return yd


@torch.jit.script
def dilate3d(y, co: int, N: Tuple[int, int, int], b: int, stride: Tuple[int, int, int]):
    sx, sy, sz = stride
    if sx == 1 and sy == 1 and sz == 1:
        return y
    yd = torch.zeros(b, co, *N, device=y.device)
    yd[:, :, ::sx, ::sy, ::sz] = y
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
