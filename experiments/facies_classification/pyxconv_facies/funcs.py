import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple

from pyxconv.probe import *
from pyxconv.utils import dilate2d, dilate3d, offsets2d, offsets3d, random_seed_torch
from .utils import align_grad_output_to_input_spatial


def _fwd_probe_grouped(mode, ps, b, ci, co, groups, nx, ny, input):
    """Probe input for XConv; for groups>1, probe each group separately."""
    if groups == 1:
        return fwd_probe[mode](ps, b, ci, nx * ny, input)
    ci_pg = ci // groups
    co_pg = co // groups
    if ci_pg * groups != ci or co_pg * groups != co:
        raise ValueError(
            f"in_channels={ci} and out_channels={co} must be divisible by groups={groups}"
        )
    parts = []
    for g in range(groups):
        sl = slice(g * ci_pg, (g + 1) * ci_pg)
        parts.append(fwd_probe[mode](ps, b, ci_pg, nx * ny, input[:, sl]))
    return torch.cat(parts, dim=2)


def _back_probe_grouped(
    mode, ps, b, ci, co, groups, nx, ny, nw_k, offs, delta, eX, seed, circular
):
    """Weight gradient for grouped XConv; matches shape (co, ci//groups, k, k).

    ``circular`` selects the boundary handling of the per-axis probe shift so the
    estimator matches the layer's padding_mode (zero-padded vs circular conv).
    This path is 2D, so ``twod=True`` is passed to ``back_probe``.
    """
    n_off = len(offs)
    # 2D path: twod=True, threed=False; nz is a dummy (unused on the 2D branch).
    if groups == 1:
        with random_seed_torch(int(seed)):
            return back_probe[mode](
                nx * ny, ci, co, b, ps, n_off, offs, delta, eX,
                nx, ny, 0, circular, True, False
            )
    ci_pg = ci // groups
    co_pg = co // groups
    dw = torch.zeros(co, ci_pg, n_off, device=eX.device, dtype=eX.dtype)
    with random_seed_torch(int(seed)):
        for g in range(groups):
            sl_in = slice(g * ci_pg, (g + 1) * ci_pg)
            sl_out = slice(g * co_pg, (g + 1) * co_pg)
            dw_g = back_probe[mode](
                nx * ny,
                ci_pg,
                co_pg,
                b,
                ps,
                n_off,
                offs,
                delta[:, sl_out],
                eX[:, :, sl_in],
                nx,
                ny,
                0,
                circular,
                True,
                False,
            )
            dw[sl_out] = dw_g
    return dw


class Xconv2D(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        input, # (B, C, H, W) eg. (10, 3, 2048, 2048)
        weight, # (num_channels, C, H, W) eg. (96, 3, 7, 7)
        ps=8, # 64
        mode='all', # independent
        bias=None, # int, 96
        stride=1, # (t, t) eg. (2, 2) 
        padding=0, # (t, t) eg. (0, 0)
        dilation=1, # (t, t) eg. (1, 1)
        groups=1,
        padding_mode='zeros',
    ):
        seed = torch.randint(100000, (1,))
        
        # b = 11, ci = 3, nx = 2048, ny = 2048
        b, ci, nx, ny = input.shape
        co = weight.shape[0]
        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = _fwd_probe_grouped(
                    mode, ps, b, ci, co, groups, nx, ny, input
                )

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.padding = padding
        ctx.mode = mode
        ctx.ps = ps
        ctx.padding_mode = padding_mode

        if padding_mode not in ('zeros', 'circular'):
            raise NotImplementedError(
                f"XConv supports padding_mode 'zeros' or 'circular', got "
                f"'{padding_mode}'. The trace-probe weight gradient can represent "
                "only zero or circular boundaries."
            )

        with torch.autograd.grad_mode.no_grad():
            # The forward must use the same boundary as the (padding-aware)
            # backward gradient estimator, otherwise the estimate is biased.
            if padding_mode == 'circular':
                ph, pw = _pair(padding)
                Y = F.conv2d(
                    F.pad(input, (pw, pw, ph, ph), mode='circular'),
                    weight, bias=bias, stride=stride, padding=0, groups=groups,
                )
            else:
                # (b, co, nx', ny') eg. (10, 96, 1021, 1021)
                Y = F.conv2d(
                    input, weight, bias=bias, stride=stride,
                    padding=padding, groups=groups,
                )

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        
        # grad_output: (B, co, nx', ny') eg. (10, 1000, 127, 127)
        # eX: (ps, b, ci) eg. (64, 10, 512)
        # weight: (co, b, nw, nw) eg. (1000, 512, 1, 1)
        # bias: (co) eg. (1000)
        eX, seed, weight, bias = ctx.saved_tensors

        # The probe shift derives the per-tap kernel offset assuming a square
        # kernel (K = sqrt(num_taps)); fail loudly otherwise rather than mis-shift.
        if weight.shape[2] != weight.shape[3]:
            raise NotImplementedError(
                "Xconv2D supports square kernels only "
                f"(got {weight.shape[2]}x{weight.shape[3]})."
            )

        dw = None
        if ctx.needs_input_grad[1]:
            nw_k = weight.shape[2]

            # b = 10, ci = 512, nx = 127, ny = 127
            b, ci, nx, ny = ctx.xshape
            co = grad_output.shape[1] # 1000

            offs = offsets2d((nx, ny), nw_k)

            # (B, co, nx', ny') eg. (10, 1000, 127, 127)
            delta = dilate2d(grad_output, co, (nx, ny), b, ctx.stride)
            with random_seed_torch(int(seed)):
                with torch.autograd.grad_mode.no_grad():
                    dw = _back_probe_grouped(
                        ctx.mode,
                        ctx.ps,
                        b,
                        ci,
                        co,
                        ctx.groups,
                        nx,
                        ny,
                        nw_k,
                        offs,
                        delta,
                        eX,
                        seed,
                        ctx.padding_mode == 'circular',
                    )
                ci_pg = ci // ctx.groups
                kh, kw = weight.shape[2], weight.shape[3]
                n_w = kh * kw
                if dw.shape[2] > n_w:
                    dw = dw[:, :, :n_w]
                dw = dw.reshape(co, ci_pg, kh, kw)

        dx = None
        if ctx.needs_input_grad[0]:
            # conv2d_input assumes zero padding; for circular the weight gradient
            # is still correct, but the input gradient would need the circular-pad
            # adjoint. Fail loudly rather than return a silently-wrong dx.
            if ctx.padding_mode == 'circular':
                raise NotImplementedError(
                    "Xconv2D input-gradient for padding_mode='circular' is not "
                    "implemented (the weight gradient is correct). Use "
                    "padding_mode='zeros'."
                )

            # (b, ci, nx', ny'), eg. (10, 512, 127, 127)
            dx = torch.nn.grad.conv2d_input(
                ctx.xshape,
                weight,
                grad_output,
                stride=ctx.stride,
                padding=ctx.padding,
                dilation=ctx.dilation,
                groups=ctx.groups
            )

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            
            # (co) eg. (1000)
            db = grad_output.sum((0, 2, 3))
            

        return dx, dw, None, None, db, None, None, None, None, None


class XconvTranspose2D(torch.autograd.Function):
    """
    Probing-based weight gradients for ConvTranspose2d.

    Input is probed like Xconv2D. Weight layout is (in_channels, out_channels, kH, kW);
    the probe accumulation is transposed to match PyTorch's transpose weight shape.
    """

    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        ps=8,
        mode='independent',
        bias=None,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
    ):
        seed = torch.randint(100000, (1,))
        b, ci, nx, ny = input.shape
        co = weight.shape[1] * groups

        with random_seed_torch(int(seed)):
            with torch.autograd.grad_mode.no_grad():
                eX = _fwd_probe_grouped(
                    mode, ps, b, ci, co, groups, nx, ny, input
                )

        ctx.xshape = input.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.mode = mode
        ctx.ps = ps

        stride_pair = stride if isinstance(stride, tuple) else (stride, stride)
        padding_pair = padding if isinstance(padding, tuple) else (padding, padding)
        output_padding_pair = (
            output_padding
            if isinstance(output_padding, tuple)
            else (output_padding, output_padding)
        )

        with torch.autograd.grad_mode.no_grad():
            Y = F.conv_transpose2d(
                input,
                weight,
                bias=bias,
                stride=stride_pair,
                padding=padding_pair,
                output_padding=output_padding_pair,
                dilation=dilation,
                groups=groups,
            )

        ctx.save_for_backward(eX, seed, weight, bias, input)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias, input = ctx.saved_tensors

        stride_pair = ctx.stride if isinstance(ctx.stride, tuple) else (ctx.stride, ctx.stride)
        padding_pair = ctx.padding if isinstance(ctx.padding, tuple) else (ctx.padding, ctx.padding)
        dilation_pair = (
            ctx.dilation if isinstance(ctx.dilation, tuple) else (ctx.dilation, ctx.dilation)
        )
        output_padding_pair = (
            ctx.output_padding
            if isinstance(ctx.output_padding, tuple)
            else (ctx.output_padding, ctx.output_padding)
        )

        dw = None
        if ctx.needs_input_grad[1]:
            if weight.shape[2] != weight.shape[3]:
                raise NotImplementedError(
                    "XconvTranspose2D supports square kernels only "
                    f"(got {weight.shape[2]}x{weight.shape[3]})."
                )
            nw_k = weight.shape[2]
            b, ci, nx, ny = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets2d((nx, ny), nw_k)
            delta = align_grad_output_to_input_spatial(
                grad_output, (nx, ny), stride_pair
            ).contiguous()

            with random_seed_torch(int(seed)):
                with torch.autograd.grad_mode.no_grad():
                    dw_probe = _back_probe_grouped(
                        ctx.mode,
                        ctx.ps,
                        b,
                        ci,
                        co,
                        ctx.groups,
                        nx,
                        ny,
                        nw_k,
                        offs,
                        delta,
                        eX,
                        seed,
                        False,
                    )
                ci_pg = ci // ctx.groups
                co_pg = co // ctx.groups
                kh, kw = weight.shape[2], weight.shape[3]
                n_w = kh * kw
                if dw_probe.shape[2] > n_w:
                    dw_probe = dw_probe[:, :, :n_w]
                if ctx.groups == 1:
                    dw = (
                        dw_probe.reshape(co, ci, kh, kw)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                    )
                else:
                    dw = torch.zeros(
                        ci, co_pg, kh, kw, device=dw_probe.device, dtype=dw_probe.dtype
                    )
                    for g in range(ctx.groups):
                        sl_out = slice(g * co_pg, (g + 1) * co_pg)
                        sl_in = slice(g * ci_pg, (g + 1) * ci_pg)
                        dw_g = (
                            dw_probe[sl_out]
                            .reshape(co_pg, ci_pg, kh, kw)
                            .permute(1, 0, 2, 3)
                        )
                        dw[sl_in] = dw_g

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.ops.aten.convolution_backward.default(
                grad_output,
                input,
                weight,
                None,
                list(stride_pair),
                list(padding_pair),
                list(dilation_pair),
                True,
                list(output_padding_pair),
                ctx.groups,
                [True, False, False],
            )[0]

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            db = grad_output.sum((0, 2, 3))

        return dx, dw, None, None, db, None, None, None, None, None


class Xconv3D(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, ps=8, mode='all', bias=None, stride=1,
                padding=0, dilation=1, groups=1, padding_mode='zeros'):
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
        ctx.padding_mode = padding_mode

        if padding_mode not in ('zeros', 'circular'):
            raise NotImplementedError(
                f"XConv supports padding_mode 'zeros' or 'circular', got "
                f"'{padding_mode}'. The trace-probe weight gradient can represent "
                "only zero or circular boundaries."
            )

        with torch.autograd.grad_mode.no_grad():
            # The forward must use the same boundary as the (padding-aware)
            # backward gradient estimator, otherwise the estimate is biased.
            if padding_mode == 'circular':
                pd, ph, pw = _triple(padding)
                Y = F.conv3d(
                    F.pad(input, (pw, pw, ph, ph, pd, pd), mode='circular'),
                    weight, bias=bias, stride=stride, padding=0, groups=groups,
                )
            else:
                Y = F.conv3d(input, weight, bias=bias, stride=stride,
                             padding=padding, groups=groups)

        ctx.save_for_backward(eX, seed, weight, bias)

        with torch.autograd.grad_mode.no_grad():
            return Y

    @staticmethod
    def backward(ctx, grad_output):
        eX, seed, weight, bias = ctx.saved_tensors

        # The probe shift derives the per-tap kernel offset assuming a cubic
        # kernel (K = nw^(1/3)); fail loudly otherwise rather than mis-shift.
        if weight.shape[2] != weight.shape[3] or weight.shape[2] != weight.shape[4]:
            raise NotImplementedError(
                "Xconv3D supports cubic kernels only "
                f"(got {weight.shape[2]}x{weight.shape[3]}x{weight.shape[4]})."
            )

        dw = None
        if ctx.needs_input_grad[1]:
            nw = weight.shape[2]
            b, ci, nx, ny, nz = ctx.xshape
            co = grad_output.shape[1]

            offs = offsets3d((nx, ny, nz), nw)
            delta = dilate3d(grad_output, co, (nx, ny, nz), b, ctx.stride)
            with random_seed_torch(int(seed)):
                with torch.autograd.grad_mode.no_grad():
                    # 3D path: padding-aware per-axis shift (threed=True);
                    # circular selects per-axis wrap vs zero-fill.
                    dw = back_probe[ctx.mode](
                        nx * ny * nz, ci, co, b, ctx.ps,
                        nw ** 3, offs, delta, eX,
                        nx, ny, nz, ctx.padding_mode == 'circular', False, True,
                    )
            dw = dw.reshape(co, ci, nw, nw, nw)

        dx = None
        if ctx.needs_input_grad[0]:
            # conv3d_input assumes zero padding; for circular the weight gradient
            # is still correct, but the input gradient would need the circular-pad
            # adjoint. Fail loudly rather than return a silently-wrong dx.
            if ctx.padding_mode == 'circular':
                raise NotImplementedError(
                    "Xconv3D input-gradient for padding_mode='circular' is not "
                    "implemented (the weight gradient is correct). Use "
                    "padding_mode='zeros'."
                )

            dx = torch.nn.grad.conv3d_input(ctx.xshape, weight, grad_output,
                                            stride=ctx.stride, padding=ctx.padding,
                                            dilation=ctx.dilation, groups=ctx.groups)

        db = None
        if bias is not None and ctx.needs_input_grad[4]:
            db = grad_output.sum((0, 2, 3, 4))

        return dx, dw, None, None, db, None, None, None, None, None


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