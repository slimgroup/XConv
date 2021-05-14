import torch

from typing import List


@torch.jit.script
def back_probe_f(N: int, ci: int, co: int, b: int, ps: int, nw: int,
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
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)
    e = torch.randn(N, ps, device=eX.device)
    Ye = grad_output.view(b, co, -1)
    eYXe = torch.zeros(ps, co, ci, device=eX.device)

    for i, o in enumerate(offs):
        ev = e if N == 1 else e.roll(-int(o), dims=0)
        eY = Ye.matmul(ev).permute((2, 1, 0)).contiguous()
        torch.bmm(eY, eX, out=eYXe)
        grad_weight[:, :, i] += eYXe.sum(0)

    return grad_weight / ps


@torch.jit.script
def back_probe_a(N: int, ci: int, co: int, b: int, ps: int, nw: int,
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
    e = torch.randn(ci, N, ps, device=eX.device)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.t(), Ye).view(ps, co, N)
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)

    for i, o in enumerate(offs):
        ev = e if N==1 else e.roll(-int(o), dims=1)
        grad_weight[:, :, i] = torch.einsum('bjk, lkb -> jl', LRe, ev)
    return grad_weight / ps


@torch.jit.script
def fwd_probe_f(ps: int, b: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.

    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor

    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.view(b, ci, -1)
    e = torch.randn(N, ps, device=X.device)
    eX = Xv.matmul(e)
    return eX.permute((2, 0, 1)).contiguous()


@torch.jit.script
def fwd_probe_a(ps: int, b: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.
    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor
    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.reshape(b, -1)
    e = torch.randn(ci, N, ps, device=X.device).view(ci*N, ps)
    return torch.mm(Xv, e)


back_probe = {'all': back_probe_a, 'features': back_probe_f}
fwd_probe = {'all': fwd_probe_a, 'features': fwd_probe_f}