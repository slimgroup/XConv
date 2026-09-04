import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import torch.nn as nn
import argparse
from torch.utils.data import Subset
from core.models import get_model
from core.loader.data_loader import patch_loader
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
import core.loss
from only_comp_avg_grad_err import (
    compute_avg_grad_err,
    get_conv_param_names
)

from pyxconv.utils import adaptive_convert_net
from pyxconv_facies.modules import XconvTranspose2D
from pyxconv_facies.utils import _collect_conv_input_spatial

# Skip fc6/fc7 (4096 channels) and 512-ch layers on patch_deconvnet.
_FACIES_XCONV_MAX_CHANNELS = 512

"""
    Compute mini-batch gradients and average gradient error for the Facies
    Classification Benchmark model.

    Non-base probing vectors: pyxconv adaptive_convert_net
    (Conv2d -> Xconv2D when H*W > ps and channels < 512).

    use probing_vector >= 16 for stable probing on patch_deconvnet.

    Usage:
        sh bash_scripts/bash_compute_mini_batch_gradients_avg_grad_err.sh
"""


def apply_transpose_xconv_only(model, sample_input, probing_vector, xmode='independent'):
    """
    Locally convert eligible ConvTranspose2d -> XconvTranspose2D only.

    Same rules as pyxconv_facies adaptive_convert_facies transpose branch:
    (H*W) > ps and in/out channels < 512.
    """
    ps = int(probing_vector)
    hook_types = (nn.ConvTranspose2d,)
    in_spatial = _collect_conv_input_spatial(model, sample_input, hook_types)
    n_converted = 0

    def _should_convert(child, full_name):
        hw = in_spatial.get(full_name)
        if hw is None:
            return False
        h, w = hw
        return (
            (h * w) > ps
            and child.in_channels < _FACIES_XCONV_MAX_CHANNELS
            and child.out_channels < _FACIES_XCONV_MAX_CHANNELS
        )

    def _convert(m, prefix=''):
        nonlocal n_converted
        for child_name, child in list(m.named_children()):
            full_name = f'{prefix}.{child_name}' if prefix else child_name

            if isinstance(child, XconvTranspose2D):
                _convert(child, full_name)
            elif isinstance(child, nn.ConvTranspose2d):
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
                    n_converted += 1
                else:
                    _convert(child, full_name)
            else:
                _convert(child, full_name)

    print(f"Converting ConvTranspose2d -> XconvTranspose2D (ps={ps})")
    _convert(model)
    print(f"Converted {n_converted} ConvTranspose2d layer(s) to XconvTranspose2D")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute mini-batch gradients and average gradient error for the Facies Classification Benchmark model.'
    )
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Whether to use data augmentation (must match full-gradient run).')
    parser.add_argument('--stride', nargs='?', type=int, default=50,
                        help='Stride when sampling patches from the volume.')
    parser.add_argument('--patch_size', nargs='?', type=int, default=99,
                        help='The size of each patch.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True,
                        help='Probing vector (use "base" for the reference full gradient).')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
    parser.add_argument(
        '--subset_indices_pickle_path',
        type=str,
        required=True,
        help='Path to the pickle file containing the selected subset indices.',
    )
    parser.add_argument(
        '--full_grads_path',
        type=str,
        required=True,
        help='Path to the pickle file containing the full gradients.',
    )
    parser.add_argument(
        '--grad_dict_dir',
        type=str,
        required=True,
        help='Path to save the average gradient errors.',
    )
    parser.add_argument('--arch', nargs='?', type=str, default='patch_deconvnet',
                        help='Architecture to use.')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=False,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument(
        '--bf16_precision',
        action='store_true',
        help='Convert model to 16-bit precision.',
    )
    args = parser.parse_args()

    return args


def compute_mini_batch_gradient_avg_grad_err(
    model,
    data_loader,
    loss_fn,
    loss_reduction: str,
    full_grad_param_dict,
    device: torch.device,
    bf16_precision: bool = False
) -> float:
    """
    Computes per mini-batch gradients and their average L2 error vs the full gradient.
    """
    model.train()
    model.zero_grad()
    batch_errors = []

    conv_param_names = get_conv_param_names(model)

    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):

        model.zero_grad()

        batch_size = images.size(0)

        images = images.to(device)
        labels = labels.to(device)

        if bf16_precision:
            images = images.half()

        outputs = model(images)
        loss = loss_fn(input=outputs, target=labels, weight=None)

        if loss_reduction == 'mean':
            pass
        elif loss_reduction == 'sum':
            loss = loss / batch_size
        else:
            raise ValueError("Unsupported loss reduction method: {}".format(loss_reduction))

        loss.backward()

        batch_grads = {
            name: param.grad.clone().cpu().detach()
            for name, param in model.named_parameters()
            if param.requires_grad and param.grad is not None
        }

        batch_grad_err = compute_avg_grad_err(
            mini_batch_grads={i: batch_grads},
            full_model_grads=full_grad_param_dict,
            model_layer=None,
            conv_param_names=conv_param_names
        )

        print("batch_grad_err for {} iteration is {}".format(i, batch_grad_err))

        batch_errors.append(batch_grad_err)

    return np.mean(batch_errors)


def main(args):

    batch_size = args.batch_size
    grad_dict_dir = args.grad_dict_dir
    bf16_precision = args.bf16_precision
    full_grads_path = args.full_grads_path
    probing_vector = args.probing_vector

    if args.aug:
        data_aug = Compose(
            [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
    else:
        data_aug = None

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    with open(args.subset_indices_pickle_path, 'rb') as f:
        subset_indices = pickle.load(f)

    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)

    print("Loading full gradients from {}".format(full_grads_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = patch_loader(
        is_transform=True,
        split='train',
        stride=args.stride,
        patch_size=args.patch_size,
        augmentations=data_aug
    )
    train_subset = Subset(train_set, subset_indices)

    print("Processing batch-size: {}".format(batch_size))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size)

    n_classes = train_set.n_classes

    model = get_model(
        args.arch,
        args.pretrained,
        n_classes
    )

    loss_fn = core.loss.cross_entropy_mean

    # Match xconv_pv and compute_full_gradients.py (mean CE, no extra mini-batch scale).
    loss_reduction = "mean"

    model = model.to(device)

    if bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    print("Processing probing_vector: {}".format(probing_vector))

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    base = (probing_vector == 'base')
    if not base:
        sample_images, _ = next(iter(train_loader))
        sample_images = sample_images.to(device)
        if bf16_precision:
            sample_images = sample_images.half()
        adaptive_convert_net(
            model,
            sample_images,
            ps=probing_vector,
            xmode='independent',
            mode='conv',
            maxc=_FACIES_XCONV_MAX_CHANNELS,
        )

    print(model)

    avg_grad_err = compute_mini_batch_gradient_avg_grad_err(
        model=model,
        data_loader=train_loader,
        loss_reduction=loss_reduction,
        loss_fn=loss_fn,
        device=device,
        full_grad_param_dict=full_grad_param_dict,
        bf16_precision=bf16_precision
    )

    num_conv_layers = len(get_conv_param_names(model))
    avg_layer_grad_err = avg_grad_err / num_conv_layers if num_conv_layers else avg_grad_err
    print(f"Avg L2 gradient error (sum over conv layers): {avg_grad_err:}")
    print(f"Avg L2 gradient error (mean per conv layer): {avg_layer_grad_err:}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_grad_err, f, pickle.HIGHEST_PROTOCOL)

    print("Saved average gradient error to {}".format(avg_grad_err_file_path))


if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)
