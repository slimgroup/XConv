import torch

from typing import List


@torch.jit.script
def rand_with_zeros(N: int, ps:int, indices, X):
    """
    Random matrix (normal distribution) with mask.

    Arguments:
        N (int): Number of pixels.
        ps (int): Number of probing vectors.
        scale (int): Scaling factor due to pseudo-orthogonalization overlap.
        indices (Tensor): Non zero indices, same for all pixels.
        X (Tensor): Input tensor (untouched, used for device detection).
    """
    n = indices.shape[0]
    a = torch.zeros(N, ps, device=X.device)
    a[:, indices] = torch.randn(N, n, device=X.device)
    return torch.sqrt(ps/n)*a


@torch.jit.script
def draw_r(ps:int, ci: int, N: int, X):
    """
    Draws a simple random multi-channel probing matrix.

    Arguments:
        ps (int): Number of probing vectors.
        ci (int): Number of input channels.
        N (int): Number of pixels.
        X (tensor): Input tensor (untouched, used for device detection).
    """
    return torch.randn(ci*N, ps, device=X.device)


@torch.jit.script
def draw_o(ps:int, ci: int, N: int, X):
    """
    Draws a pseudo block-orthogonal probing matrix. For a large enough number of probing vector (ps > 8 * ci)
    this matrix will be exactly block orthogonal

    Arguments:
        ps (int): Number of probing vectors.
        ci (int): Number of input channels.
        N (int): Number of pixels.
        X (tensor): Input tensor (untouched, used for device detection).
    """
    if ps // ci > 8:
        n = ps // ci
        inds = torch.split(torch.randperm(ps, dtype=torch.long), n)
    else:
        n = 8
        inds = torch.split(torch.randperm(n*ci, dtype=torch.long) % ps, n)
    e = torch.cat([rand_with_zeros(N, ps, inds[i], X) for i in range(ci)], dim=0)
    return e


@torch.jit.script
def _shift2d_A(e, sx: int, sy: int, nx: int, ny: int, circular: bool):
    """
    Per-axis 2D shift of a probing matrix: out[x, y] = e[x + sx, y + sy].

    The trace estimator needs the probing vectors shifted exactly the way
    the convolution shifts the image, i.e. once per spatial axis. Zero
    filling outside the image (circular=False) reproduces a zero-padded
    convolution; wrapping each axis (circular=True) reproduces a circular
    one. Rolling the flattened pixel axis by a single scalar offset wraps
    every image row into the next, which is neither boundary, and biases
    the filter gradient whenever the filter is wider than one tap.

    Arguments:
        e (Tensor): Probing matrix, shape (nx*ny, ps).
        sx (int): Shift along the first spatial axis.
        sy (int): Shift along the second spatial axis.
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        circular (bool): Wrap at the boundary instead of filling with zeros.

    Returns:
        Shifted probing matrix, shape (nx*ny, ps).
    """
    ps = e.shape[1]
    g = e.reshape(nx, ny, ps)
    if circular:
        return torch.roll(g, [-sx, -sy], [0, 1]).reshape(nx * ny, ps)
    out = torch.zeros_like(g)
    xlo = -sx if sx < 0 else 0
    xhi = nx - sx if sx > 0 else nx
    ylo = -sy if sy < 0 else 0
    yhi = ny - sy if sy > 0 else ny
    if xhi > xlo and yhi > ylo:
        out[xlo:xhi, ylo:yhi] = g[xlo + sx:xhi + sx, ylo + sy:yhi + sy]
    return out.reshape(nx * ny, ps)


@torch.jit.script
def _shift2d_B(e, sx: int, sy: int, ci: int, nx: int, ny: int,
               circular: bool):
    """
    Per-axis 2D shift of a multi-channel probing matrix (ci, nx*ny, ps).

    Channel-wise analogue of :func:`_shift2d_A`; see it for the boundary
    convention and for why a flattened roll is biased.

    Arguments:
        e (Tensor): Probing matrix, shape (ci, nx*ny, ps).
        sx (int): Shift along the first spatial axis.
        sy (int): Shift along the second spatial axis.
        ci (int): Number of input channels.
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        circular (bool): Wrap at the boundary instead of filling with zeros.

    Returns:
        Shifted probing matrix, shape (ci, nx*ny, ps).
    """
    ps = e.shape[2]
    g = e.reshape(ci, nx, ny, ps)
    if circular:
        return torch.roll(g, [-sx, -sy], [1, 2]).reshape(ci, nx * ny, ps)
    out = torch.zeros_like(g)
    xlo = -sx if sx < 0 else 0
    xhi = nx - sx if sx > 0 else nx
    ylo = -sy if sy < 0 else 0
    yhi = ny - sy if sy > 0 else ny
    if xhi > xlo and yhi > ylo:
        out[:, xlo:xhi, ylo:yhi] = g[:, xlo + sx:xhi + sx, ylo + sy:yhi + sy]
    return out.reshape(ci, nx * ny, ps)


@torch.jit.script
def _shift3d_A(e, sx: int, sy: int, sz: int, nx: int, ny: int, nz: int,
               circular: bool):
    """
    Per-axis 3D shift of a probing matrix, the analogue of _shift2d_A.

    Arguments:
        e (Tensor): Probing matrix, shape (nx*ny*nz, ps).
        sx (int): Shift along the first spatial axis.
        sy (int): Shift along the second spatial axis.
        sz (int): Shift along the third spatial axis.
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        nz (int): Third spatial dimension.
        circular (bool): Wrap at the boundary instead of filling with zeros.

    Returns:
        Shifted probing matrix, shape (nx*ny*nz, ps).
    """
    ps = e.shape[1]
    n = nx * ny * nz
    g = e.reshape(nx, ny, nz, ps)
    if circular:
        return torch.roll(g, [-sx, -sy, -sz], [0, 1, 2]).reshape(n, ps)
    out = torch.zeros_like(g)
    xlo = -sx if sx < 0 else 0
    xhi = nx - sx if sx > 0 else nx
    ylo = -sy if sy < 0 else 0
    yhi = ny - sy if sy > 0 else ny
    zlo = -sz if sz < 0 else 0
    zhi = nz - sz if sz > 0 else nz
    if xhi > xlo and yhi > ylo and zhi > zlo:
        out[xlo:xhi, ylo:yhi, zlo:zhi] = g[xlo + sx:xhi + sx,
                                           ylo + sy:yhi + sy,
                                           zlo + sz:zhi + sz]
    return out.reshape(n, ps)


@torch.jit.script
def _shift3d_B(e, sx: int, sy: int, sz: int, ci: int, nx: int, ny: int,
               nz: int, circular: bool):
    """
    Per-axis 3D shift of a multi-channel probing matrix (ci, nx*ny*nz, ps).

    Channel-wise analogue of :func:`_shift3d_A`.

    Arguments:
        e (Tensor): Probing matrix, shape (ci, nx*ny*nz, ps).
        sx (int): Shift along the first spatial axis.
        sy (int): Shift along the second spatial axis.
        sz (int): Shift along the third spatial axis.
        ci (int): Number of input channels.
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        nz (int): Third spatial dimension.
        circular (bool): Wrap at the boundary instead of filling with zeros.

    Returns:
        Shifted probing matrix, shape (ci, nx*ny*nz, ps).
    """
    ps = e.shape[2]
    n = nx * ny * nz
    g = e.reshape(ci, nx, ny, nz, ps)
    if circular:
        return torch.roll(g, [-sx, -sy, -sz], [1, 2, 3]).reshape(ci, n, ps)
    out = torch.zeros_like(g)
    xlo = -sx if sx < 0 else 0
    xhi = nx - sx if sx > 0 else nx
    ylo = -sy if sy < 0 else 0
    yhi = ny - sy if sy > 0 else ny
    zlo = -sz if sz < 0 else 0
    zhi = nz - sz if sz > 0 else nz
    if xhi > xlo and yhi > ylo and zhi > zlo:
        out[:, xlo:xhi, ylo:yhi, zlo:zhi] = g[:, xlo + sx:xhi + sx,
                                              ylo + sy:yhi + sy,
                                              zlo + sz:zhi + sz]
    return out.reshape(ci, n, ps)


@torch.jit.script
def _tap_width(nw: int, twod: bool, threed: bool):
    """
    Filter width K from the number of taps: K = nw**(1/2) (2D) or nw**(1/3).

    Arguments:
        nw (int): Number of filter taps (K*K in 2D, K*K*K in 3D).
        twod (bool): Layer is 2D.
        threed (bool): Layer is 3D.

    Returns:
        Filter width K, or 1 when the layer is neither 2D nor 3D.
    """
    if twod:
        return int(round(float(nw) ** 0.5))
    if threed:
        return int(round(float(nw) ** (1.0 / 3.0)))
    return 1


@torch.jit.script
def back_probe_f(N: int, ci: int, co: int, b: int, ps: int, nw: int,
                 offs: List[int], grad_output, eX, nx: int, ny: int, nz: int,
                 circular: bool, twod: bool, threed: bool):
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
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        nz (int): Third spatial dimension, unused for a 2D layer.
        circular (bool): Wrap the probe shift at the boundary (circular
            padding) instead of filling with zeros (zero padding).
        twod (bool): Layer is 2D.
        threed (bool): Layer is 3D.

    Returns:
        gradient w.r.t convolution filter
    """
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)
    e = torch.randn(N, ps, device=eX.device)
    Ye = grad_output.view(b, co, -1)
    eYXe = torch.zeros(ps, co, ci, device=eX.device)
    K = _tap_width(nw, twod, threed)

    for i, o in enumerate(offs):
        if twod:
            ev = _shift2d_A(e, i // K - K // 2, i % K - K // 2, nx, ny,
                            circular)
        elif threed:
            ev = _shift3d_A(e, i // (K * K) - K // 2, (i // K) % K - K // 2,
                            i % K - K // 2, nx, ny, nz, circular)
        else:
            ev = e if N == 1 else e.roll(-int(o), dims=0)
        eY = Ye.matmul(ev).permute((2, 1, 0)).contiguous()
        torch.bmm(eY, eX, out=eYXe)
        grad_weight[:, :, i] += eYXe.sum(0)

    return grad_weight / ps


@torch.jit.script
def back_probe_a(N: int, ci: int, co: int, b: int, ps: int, nw: int,
                 offs: List[int], grad_output, eX, nx: int, ny: int, nz: int,
                 circular: bool, twod: bool, threed: bool):
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
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        nz (int): Third spatial dimension, unused for a 2D layer.
        circular (bool): Wrap the probe shift at the boundary (circular
            padding) instead of filling with zeros (zero padding).
        twod (bool): Layer is 2D.
        threed (bool): Layer is 3D.
    Returns:
        gradient w.r.t convolution filter
    """
    # Redraw e
    e = draw_r(ps, ci, N, eX).view(ci, N, ps)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.t(), Ye).view(ps, co, N)
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)
    K = _tap_width(nw, twod, threed)

    for i, o in enumerate(offs):
        if twod:
            ev = _shift2d_B(e, i // K - K // 2, i % K - K // 2, ci, nx, ny,
                            circular)
        elif threed:
            ev = _shift3d_B(e, i // (K * K) - K // 2, (i // K) % K - K // 2,
                            i % K - K // 2, ci, nx, ny, nz, circular)
        else:
            ev = e if N==1 else e.roll(-int(o), dims=1)
        grad_weight[:, :, i] = torch.einsum('bjk, lkb -> jl', LRe, ev)
    return grad_weight / ps


@torch.jit.script
def back_probe_o(N: int, ci: int, co: int, b: int, ps: int, nw: int,
                 offs: List[int], grad_output, eX, nx: int, ny: int, nz: int,
                 circular: bool, twod: bool, threed: bool):
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
        nx (int): First spatial dimension.
        ny (int): Second spatial dimension.
        nz (int): Third spatial dimension, unused for a 2D layer.
        circular (bool): Wrap the probe shift at the boundary (circular
            padding) instead of filling with zeros (zero padding).
        twod (bool): Layer is 2D.
        threed (bool): Layer is 3D.
    Returns:
        gradient w.r.t convolution filter
    """
    # Redraw e
    e = draw_o(ps, ci, N, eX).view(ci, N, ps)

    # Y' X e
    Ye = grad_output.view(b, -1)
    LRe = torch.mm(eX.t(), Ye).view(ps, co, N)
    # Init gradient
    grad_weight = torch.zeros(co, ci, nw, device=eX.device)
    K = _tap_width(nw, twod, threed)

    for i, o in enumerate(offs):
        if twod:
            ev = _shift2d_B(e, i // K - K // 2, i % K - K // 2, ci, nx, ny,
                            circular)
        elif threed:
            ev = _shift3d_B(e, i // (K * K) - K // 2, (i // K) % K - K // 2,
                            i % K - K // 2, ci, nx, ny, nz, circular)
        else:
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
    e = draw_r(ps, ci, N, X)

    return torch.mm(Xv, e.reshape(ci*N, ps))


@torch.jit.script
def fwd_probe_o(ps: int, b: int, ci: int, N: int, X):
    """
    Forward pass of probing-based convolution filter gradient.
    Arguments:
        ps (int): Number of probing vectors
        X (Tensor): Layer's input Tensor
    Returns:
        eX (Tensor): Probed input tensor to be saved for backward pass
    """
    Xv = X.reshape(b, -1)
    e = draw_o(ps, ci, N, X)

    return torch.mm(Xv, e.reshape(ci*N, ps))


# Access dictionaries
back_probe = {'gaussian': back_probe_a, 'orthogonal': back_probe_o, 'independent': back_probe_f}
fwd_probe = {'gaussian': fwd_probe_a, 'orthogonal': fwd_probe_o, 'independent': fwd_probe_f}
draw_e = {'orthogonal': draw_o, 'gaussian': draw_r}