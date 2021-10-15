import torch
import contextlib

from typing import Tuple

__all__ = ['convert_net', 'update_ps', 'dilate2d', 'dilate3d', 'offsets2d', 'offsets3d',
           'random_seed_torch', 'update_mode']


def convert_net(module, name='net', ps=16, xmode='gaussian', mode='all', maxc=32001):
    """Recursively replaces all nn.Conv2d by XConv."""
    from .modules import Xconv2D, Xconv3D, BLRReLU
    # iterate through immediate child modules. Note, the recursion is
    # done by our code no need to use named_modules()
    for child_name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d) and mode in ['all', 'conv']:
            if child.in_channels < maxc and child.out_channels < maxc:
                b = child.bias is not None
                newconv = Xconv2D(child.in_channels, child.out_channels,
                                  child.kernel_size, ps=ps, mode=xmode,
                                  stride=child.stride, padding=child.padding, bias=b)
                newconv.weight = child.weight
                newconv.bias = child.bias
                setattr(module, child_name, newconv)
        elif isinstance(child, torch.nn.Conv3d) and mode in ['all', 'conv']:
            if child.in_channels < maxc and child.out_channels < maxc:
                b = child.bias is not None
                newconv = Xconv3D(child.in_channels, child.out_channels,
                                  child.kernel_size, ps=ps, stride=child.stride,
                                  padding=child.padding, bias=b, mode=xmode)
                newconv.weight = child.weight
                newconv.bias = child.bias
                setattr(module, child_name, newconv)
        elif isinstance(child, torch.nn.ReLU) and mode in ['all', 'relu']:
            setattr(module, child_name, BLRReLU(r=ps, inplace=child.inplace))
        else:
            convert_net(child, child_name, ps=ps, xmode=xmode, mode=mode, maxc=maxc)


def update_ps(module, ps):
    from .modules import Xconv2D, Xconv3D
    for child_name, child in module.named_children():
        if isinstance(child, Xconv2D) or isinstance(child, Xconv3D):
            child.ps = ps
        else:
            update_ps(child, ps)

def update_mode(module, mode):
    from .modules import Xconv2D, Xconv3D
    for child_name, child in module.named_children():
        if isinstance(child, Xconv2D) or isinstance(child, Xconv3D):
            child.mode = mode
        else:
            update_mode(child, mode)


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


@contextlib.contextmanager
def random_seed_torch(seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)
