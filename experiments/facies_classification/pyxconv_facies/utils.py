import torch
import contextlib

from typing import Dict, Tuple

__all__ = [
    'convert_net',
    'adaptive_convert_net',
    'adaptive_convert_facies',
    'update_ps',
    'dilate2d',
    'dilate3d',
    'align_grad_output_to_input_spatial',
    'offsets2d',
    'offsets3d',
    'random_seed_torch',
    'update_mode',
]

def adaptive_convert_net(
    module: torch, 
    sample_input: torch.Tensor, 
    ps: int =16, 
    xmode: str ='gaussian', 
    mode: str ='all', 
    maxc: int =32001
):
    """
    Recursively replaces all nn.Conv2d by SpatialGatedXConv via Two-Pass conversion.
        Pass 1: run a dry forward to record each Conv2d's input HxW.
        Pass 2: replace Conv2d -> Xconv2D ONLY when (H*W) > ps and channels < maxc.

    sample_input: sample input tensor to the network for computing activation maps
    
    """
    from .modules import Xconv2D, Xconv3D, XconvTranspose2D, BReLU
    print("Probing vector is: {}".format(ps))

    # ---------- Pass 1: collect input shapes ----------
    # Map full module name -> (H, W) (max seen if multiple calls)
    in_spatial: Dict[str, Tuple[int,int]] = {}

    # Build a stable name map for all submodules
    name_map = {m: n for n, m in module.named_modules()}  # module object -> dotted name
    
    def hook_fn(m, inputs, _output):
        if not isinstance(m, torch.nn.Conv2d):
            return
        
        # inputs is a tuple; first item is the input tensor to this module
        x = inputs[0]
        # Expect shape [B, C, H, W]
        if x.dim() >= 4:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            nm = name_map[m]
            prev = in_spatial.get(nm)
            area = H * W
            if prev is None:
                in_spatial[nm] = (H, W)
            else:
                # keep the largest area seen (handles multi-branch / multi-call)
                if H * W > prev[0] * prev[1]:
                    in_spatial[nm] = (H, W)

    # Register hooks only on Conv2d
    hooks = []
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_fn))

    # Dry forward (no grad)
    module.eval()
    with torch.no_grad():
        _ = module(sample_input)

    # Clean up hooks
    for h in hooks:
        h.remove()

         # ---------- Pass 2: recursive replacement ----------
    def _convert(m: torch.nn.Module, prefix: str = ''):
        for child_name, child in list(m.named_children()):
            full_name = f'{prefix}.{child_name}' if prefix else child_name

            if isinstance(child, (Xconv2D, XconvTranspose2D)):
                _convert(child, full_name)
            elif isinstance(child, torch.nn.Conv2d) and mode in ['all', 'conv']:
                # Only replace if we observed an input size and it exceeds ps
                hw = in_spatial.get(full_name)
                if hw is not None:
                    H, W = hw
                    if (H * W) > ps and child.in_channels < maxc and child.out_channels < maxc:
                        b = child.bias is not None
                        newconv = Xconv2D(
                            child.in_channels,
                            child.out_channels,
                            child.kernel_size,
                            ps=ps,
                            mode=xmode,
                            stride=child.stride,
                            padding=child.padding,
                            bias=b,
                            padding_mode=child.padding_mode,
                        )
                        # Copy parameters (keep same tensors to preserve optimizer state if needed)
                        newconv.weight = child.weight
                        newconv.bias   = child.bias
                        setattr(m, child_name, newconv)
                    else:
                        # keep as-is, but recurse inside (in case it's a Sequential etc.)
                        _convert(child, full_name)
                else:
                    # If we never saw an input (dead branch), leave it untouched
                    _convert(child, full_name)

            elif isinstance(child, torch.nn.Conv3d) and mode in ['all', 'conv']:
                if child.in_channels < maxc and child.out_channels < maxc:
                    b = child.bias is not None
                    newconv = Xconv3D(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        ps=ps,
                        stride=child.stride,
                        padding=child.padding,
                        bias=b,
                        mode=xmode,
                        padding_mode=child.padding_mode,
                    )
                    newconv.weight = child.weight
                    newconv.bias   = child.bias
                    setattr(m, child_name, newconv)
                else:
                    _convert(child, full_name)

            elif isinstance(child, torch.nn.ReLU) and mode in ['all', 'relu']:
                from .modules import BReLU
                setattr(m, child_name, BReLU(inplace=child.inplace))

            else:
                _convert(child, full_name)

    _convert(module)
    return module


def align_grad_output_to_input_spatial(
    grad_output: torch.Tensor,
    input_hw: Tuple[int, int],
    stride,
) -> torch.Tensor:
    """
    Map grad_output spatial grid to the layer input (H, W) for probing-based weight grads.

    Mirrors dilate2d for standard conv (upsample grad to input size). For transpose
    conv with stride > 1, grad_output is often larger than the input; subsample instead.
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    nx: int = input_hw[0]
    ny: int = input_hw[1]
    b: int = grad_output.shape[0]
    co: int = grad_output.shape[1]
    nox: int = grad_output.shape[2]
    noy: int = grad_output.shape[3]
    sx: int = stride[0]
    sy: int = stride[1]

    if nox == nx and noy == ny:
        return grad_output

    if nox >= nx and noy >= ny:
        if sx == 1 and sy == 1:
            return grad_output[:, :, :nx, :ny]
        return grad_output[:, :, ::sx, ::sy][:, :, :nx, :ny]

    yd = torch.zeros(b, co, nx, ny, device=grad_output.device, dtype=grad_output.dtype)
    yd[:, :, ::sx, ::sy][:, :, :nox, :noy] = grad_output
    return yd


def _collect_conv_input_spatial(
    module: torch.nn.Module,
    sample_input: torch.Tensor,
    hook_types: Tuple[type, ...],
) -> Dict[str, Tuple[int, int]]:
    in_spatial: Dict[str, Tuple[int, int]] = {}
    name_map = {m: n for n, m in module.named_modules()}

    def hook_fn(m, inputs, _output):
        if not isinstance(m, hook_types):
            return
        x = inputs[0]
        if x.dim() >= 4:
            h, w = int(x.shape[-2]), int(x.shape[-1])
            nm = name_map[m]
            prev = in_spatial.get(nm)
            if prev is None or h * w > prev[0] * prev[1]:
                in_spatial[nm] = (h, w)

    hooks = []
    for m in module.modules():
        if isinstance(m, hook_types):
            hooks.append(m.register_forward_hook(hook_fn))

    was_training = module.training
    module.eval()
    with torch.no_grad():
        module(sample_input)
    module.train(was_training)

    for h in hooks:
        h.remove()

    return in_spatial


def adaptive_convert_facies(
    module: torch.nn.Module,
    sample_input: torch.Tensor,
    ps: int = 16,
    xmode: str = 'independent',
    mode: str = 'all',
    maxc: int = 32001,
):
    """
    Adaptive XConv for facies patch_deconvnet: Conv2d -> Xconv2D and
    ConvTranspose2d -> XconvTranspose2D when (H*W) > ps and channels < maxc.
    """
    from .modules import Xconv2D, XconvTranspose2D, BReLU

    print("Probing vector is: {}".format(ps))
    hook_types = (torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    in_spatial = _collect_conv_input_spatial(module, sample_input, hook_types)

    def _should_convert(child, full_name: str) -> bool:
        hw = in_spatial.get(full_name)
        if hw is None:
            return False
        h, w = hw
        return (
            (h * w) > ps
            and child.in_channels < maxc
            and child.out_channels < maxc
        )

    def _convert(m: torch.nn.Module, prefix: str = ''):
        for child_name, child in list(m.named_children()):
            full_name = f'{prefix}.{child_name}' if prefix else child_name

            if isinstance(child, (Xconv2D, XconvTranspose2D)):
                _convert(child, full_name)
            elif isinstance(child, torch.nn.Conv2d) and mode in ['all', 'conv']:
                if _should_convert(child, full_name):
                    b = child.bias is not None
                    newconv = Xconv2D(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        ps=ps,
                        mode=xmode,
                        stride=child.stride,
                        padding=child.padding,
                        bias=b,
                        padding_mode=child.padding_mode,
                    )
                    newconv.weight = child.weight
                    newconv.bias = child.bias
                    setattr(m, child_name, newconv)
                else:
                    _convert(child, full_name)

            elif isinstance(child, torch.nn.ConvTranspose2d) and mode in ['all', 'conv']:
                if _should_convert(child, full_name):
                    b = child.bias is not None
                    newdeconv = XconvTranspose2D(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        ps=ps,
                        mode=xmode,
                        stride=child.stride,
                        padding=child.padding,
                        output_padding=child.output_padding,
                        bias=b,
                    )
                    newdeconv.weight = child.weight
                    newdeconv.bias = child.bias
                    setattr(m, child_name, newdeconv)
                else:
                    _convert(child, full_name)

            elif isinstance(child, torch.nn.ReLU) and mode in ['all', 'relu']:
                setattr(m, child_name, BReLU(inplace=child.inplace))

            else:
                _convert(child, full_name)

    _convert(module)
    return module


def convert_net(module, name='net', ps=16, xmode='gaussian', mode='all', maxc=32001):
    """Recursively replace nn.Conv2d / nn.ConvTranspose2d by Xconv2D / XconvTranspose2D."""
    from .modules import Xconv2D, Xconv3D, XconvTranspose2D, BReLU

    # iterate through immediate child modules. Note, the recursion is
    # done by our code no need to use named_modules()
    for child_name, child in module.named_children():
        if isinstance(child, (Xconv2D, XconvTranspose2D)):
            convert_net(child, child_name, ps=ps, xmode=xmode, mode=mode, maxc=maxc)
        elif isinstance(child, torch.nn.Conv2d) and mode in ['all', 'conv']:
            if child.in_channels < maxc and child.out_channels < maxc:
                b = child.bias is not None
                newconv = Xconv2D(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    ps=ps,
                    mode=xmode,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=b,
                    padding_mode=child.padding_mode,
                )
                newconv.weight = child.weight
                newconv.bias = child.bias
                setattr(module, child_name, newconv)
        elif isinstance(child, torch.nn.ConvTranspose2d) and mode in ['all', 'conv']:
            if child.in_channels < maxc and child.out_channels < maxc:
                b = child.bias is not None
                newdeconv = XconvTranspose2D(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    ps=ps,
                    mode=xmode,
                    stride=child.stride,
                    padding=child.padding,
                    output_padding=child.output_padding,
                    bias=b,
                )
                newdeconv.weight = child.weight
                newdeconv.bias = child.bias
                setattr(module, child_name, newdeconv)
        elif isinstance(child, torch.nn.Conv3d) and mode in ['all', 'conv']:
            if child.in_channels < maxc and child.out_channels < maxc:
                b = child.bias is not None
                newconv = Xconv3D(child.in_channels, child.out_channels,
                                  child.kernel_size, ps=ps, stride=child.stride,
                                  padding=child.padding, bias=b, mode=xmode,
                                  padding_mode=child.padding_mode)
                newconv.weight = child.weight
                newconv.bias = child.bias
                setattr(module, child_name, newconv)
        elif isinstance(child, torch.nn.ReLU) and mode in ['all', 'relu']:
            setattr(module, child_name, BReLU(inplace=child.inplace))
        else:
            convert_net(child, child_name, ps=ps, xmode=xmode, mode=mode, maxc=maxc)


def update_ps(module, ps):
    from .modules import Xconv2D, Xconv3D, XconvTranspose2D
    for child_name, child in module.named_children():
        if isinstance(child, (Xconv2D, Xconv3D, XconvTranspose2D)):
            child.ps = ps
        else:
            update_ps(child, ps)

def update_mode(module, mode):
    from .modules import Xconv2D, Xconv3D, XconvTranspose2D
    for child_name, child in module.named_children():
        if isinstance(child, (Xconv2D, Xconv3D, XconvTranspose2D)):
            child.mode = mode
        else:
            update_mode(child, mode)


@torch.jit.script
def dilate2d(y, co: int, N: Tuple[int, int], b: int, stride: Tuple[int, int]):
    sx, sy = stride
    if sx == 1 and sy == 1:
        return y
    yd = torch.zeros(b, co, *N, device=y.device, dtype=y.dtype)
    
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
