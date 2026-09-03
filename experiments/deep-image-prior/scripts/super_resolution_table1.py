# scripts/super_resolution_table1.py

from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import sys
import json
import wandb
from pyxconv.nvidia_mem_tracker import MemoryTracker


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torchmetrics

from models import *
from models.downsampler import Downsampler
from utils.sr_utils import *
from pyxconv.utils import convert_net
from projorg import make_experiment_name

dtype = torch.cuda.FloatTensor

"""
    Comparing Deep-Image Prior performance on Super-Resolution task.

    Usage:
        sh bash_scripts/bash_super_resolution.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Super-resolution with Deep Image Prior (Table 1 / Set5 x4 style)'
    )

    parser.add_argument('--probing_vector', type=str, default='base',
                        help='"base" = Conv; digit = probing vector for XConv')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Base output directory')

    parser.add_argument('--input_image_path', type=str, required=True,
                        help='Path to HR image (e.g. SR/Set5/original/baby.png)')
    parser.add_argument('--img_name', type=str, required=True,
                        help='Image name for logging (e.g. "baby")')

    parser.add_argument('--factor', type=int, default=4,
                        help='Downscaling factor (4 for Table 1)')
    parser.add_argument('--imsize', type=int, default=-1,
                        help='Optional HR resize; -1 = no resize')
    parser.add_argument('--enforce_div32', type=str, default='CROP',
                        help='"CROP" crops HR to be divisible by 32')

    parser.add_argument('--optimizer', type=str, required=True,
                        help='Optimizer name (e.g. "adam")')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate (DIP uses 1e-2)')
    parser.add_argument('--net_type', type=str, default='skip',
                        help='Net type: skip, unet, resnet')
    parser.add_argument('--input_depth', type=int, default=32,
                        help='Input noise depth (32 in paper)')
    parser.add_argument('--method', type=str, default='noise',
                        help='Noise method')
    parser.add_argument('--pad', type=str, default='reflection',
                        help='Padding for Conv; XConv will switch to zero')

    parser.add_argument('--opt_over', type=str, default='net',
                        help='"net", "input", or "net,input"')

    parser.add_argument('--kernel_type', type=str, default='lanczos2',
                        help='Downsampler kernel type (lanczos2 in DIP SR)')
    parser.add_argument('--tv_weight', type=float, default=0.0,
                        help='TV regularization weight (usually 0 for SR)')

    parser.add_argument('--num_iter', type=int, default=None,
                        help='If None: 2000 for x4, 4000 for x8')
    parser.add_argument('--reg_noise_std', type=float, default=None,
                        help='If None: 0.03 for x4, 0.05 for x8')

    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Experiment name for logging')
    parser.add_argument('--plot', type=bool, default=True,
                        help='Save intermediate images')

    return parser.parse_args()


def setup_parameters(factor: int):
    if factor == 4:
        return 2000, 0.03
    elif factor == 8:
        return 4000, 0.05
    else:
        raise ValueError('DIP SR paper uses factors 4 and 8 only.')


def rgb_to_y(img_np):
    """
    img_np: np.array, shape (C,H,W) or (H,W,C), values in [0,1]
    returns Y channel, shape (H,W)
    """
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):  # CHW -> HWC
        img_np = np.transpose(img_np, (1, 2, 0))
    r = img_np[..., 0]
    g = img_np[..., 1]
    b = img_np[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def psnr_y(img_np, ref_np, crop=4):
    """
    PSNR on Y channel with border crop.
    img_np, ref_np: np arrays (C,H,W) or (H,W,C), in [0,1]
    """
    y = rgb_to_y(img_np)
    y_ref = rgb_to_y(ref_np)

    if crop > 0:
        y = y[crop:-crop, crop:-crop]
        y_ref = y_ref[crop:-crop, crop:-crop]

    mse = np.mean((y - y_ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def main(args):
    imsize = args.imsize
    factor = args.factor
    opt_over = args.opt_over
    kernel_type = args.kernel_type
    tv_weight = args.tv_weight
    enforce_div32 = args.enforce_div32
    PLOT = args.plot
    hr_image_path = args.input_image_path
    output_dir = args.output_dir
    probing_vector = args.probing_vector
    lr = args.lr
    optimizer = args.optimizer
    input_depth = args.input_depth
    method = args.method
    pad = args.pad
    net_type = args.net_type
    num_iter = args.num_iter
    reg_noise_std = args.reg_noise_std

    # defaults from paper if not supplied
    if num_iter is None or reg_noise_std is None:
        num_iter_def, reg_def = setup_parameters(factor=factor)
        if num_iter is None:
            num_iter = num_iter_def
        if reg_noise_std is None:
            reg_noise_std = reg_def

    args.num_iter = num_iter
    args.reg_noise_std = reg_noise_std

    experiment_name = make_experiment_name(
        args,
        ignore_arg_list=[
            "experiment_name",
            "imsize",
            "enforce_div32",
            "input_depth",
            "opt_over",
            "kernel_type",
            "tv_weight",
            "plot",
            "input_image_path",
            "net_type",
            "output_dir",
        ]
    )
    print("W&B run name:", experiment_name)
    wandb.init(
        project="super-resolution",
        name=experiment_name,
        config=args
    )

    wandb.save('*.py')
    wandb.save('bash_scripts/*')
    wandb.save('configs/*')
    wandb.save('newton_slurm_scripts/*')
    wandb.save("models/*")
    wandb.save("utils/*")
    wandb.save("scripts/*")

    output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)


    # probing_vector: base vs XConv
    if isinstance(probing_vector, str) and probing_vector.isdigit():
        probing_vector = int(probing_vector)
    base = (probing_vector == 'base')

    # === 1) Load HR, generate LR (we ignore LR inside loader later) ===
    imgs = load_LR_HR_imgs_sr(
        fname=hr_image_path,
        imsize=imsize,
        factor=factor,
        enforce_div32=enforce_div32
    )

    baseline_imgs =  get_baselines(
        imgs['LR_pil'], imgs['HR_pil']
    )
    
    

    print("HR size:", imgs['HR_pil'].size, "| LR size:", imgs['LR_pil'].size)

    # === 2) Network input noise ===
    net_input = get_noise(
        input_depth=input_depth,
        method=method,
        spatial_size=(imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])
    ).type(dtype).detach()

    # === 3) Build network ===
    pad_for_net = pad
    if not base and pad == 'reflection':
        print("XConv + reflection pad is problematic; switching pad to 'zero'.")
        pad_for_net = 'zero'

    net = get_net(
        input_depth=input_depth,
        NET_TYPE=net_type,
        pad=pad_for_net,
        skip_n33d=128,
        skip_n33u=128,
        skip_n11=4,
        num_scales=5,
        upsample_mode='bilinear'
    ).type(dtype)

    if not base:
        convert_net(
            net, 
            ps=probing_vector, 
            xmode='independent'
        )
        print("Converted net to XConv with probing vector:", probing_vector)
    else:
        print("Using base Conv network.")
    print(net)

    # === 4) Downsampler & synthetic LR ===
    mse = torch.nn.MSELoss().type(dtype)

    downsampler = Downsampler(
        n_planes=3,
        factor=factor,
        kernel_type=kernel_type,
        phase=0.5,
        preserve_size=True
    ).type(dtype)

    print("Downsampler:")
    print(downsampler)

    # Ground-truth HR tensor
    img_HR_var = np_to_torch(imgs['HR_np']).type(dtype)

    # Synthetic LR, consistent with forward model
    with torch.no_grad():
        img_LR_var = downsampler(img_HR_var).detach()

    # === 5) Training loop ===
    psnr_history = []  # [psnr_LR_rgb, psnr_HR_rgb, psnr_HR_Y]
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    psnr_fn = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).cuda()

    i = 0
    best_psnr = -1e9
    best_path = None

    def closure():
        nonlocal i, net_input, best_psnr, best_path

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()

        # PSNR RGB
        psnr_LR = psnr_fn(out_LR, img_LR_var).item()
        psnr_HR_rgb = psnr_fn(out_HR, img_HR_var).item()

        # PSNR Y (cropped by factor)
        out_HR_np = torch_to_np(out_HR)
        psnr_HR_Y = psnr_y(
            out_HR_np, 
            imgs['HR_np'], 
            crop=factor
        )

        print(f"Iteration {i:05d}")
        print(f"   LOSS       : {total_loss.item():.6f}")
        print(f"   PSNR_LR    : {psnr_LR:0.3f}")
        print(f"   PSNR_HR_RGB: {psnr_HR_rgb:0.3f}")
        print(f"   PSNR_HR_Y  : {psnr_HR_Y:0.3f}")
        print("-" * 50)


        wandb.log({
            "loss": total_loss.item(),
            "psnr_LR": psnr_LR,
            "psnr_HR_rgb": psnr_HR_rgb,
            "psnr_HR_Y": psnr_HR_Y,
            "iter": i,
        })

        psnr_history.append([psnr_LR, psnr_HR_rgb, psnr_HR_Y])

        # Save the best PSNR image.
        if psnr_HR_Y > best_psnr:
            best_psnr = psnr_HR_Y
            best_path = os.path.join(output_dir, f"best_img_super_resolution_iteration_{i:05d}_psnrY_{psnr_HR_Y:0.3f}.png")
            np_to_pil(np.clip(out_HR_np, 0, 1)).save(best_path)

        if PLOT and (i % 100 == 0 or i == num_iter - 1):
            plot_image_grid(
                [
                    imgs['HR_np'],
                    baseline_imgs['bicubic_np'],
                    np.clip(out_HR_np, 0, 1)
                ],
                factor=13,
                nrow=3
            )
            path_name = os.path.join(
                output_dir,
                f"sr_iter_{i:05d}_psnrY_{psnr_HR_Y:0.3f}.png"
            )
            plt.savefig(path_name)
            print("Saved image to", path_name)

        i += 1
        return total_loss

    # Optimize
    with MemoryTracker() as t:
        p = get_params(
            opt_over = opt_over, 
            net = net, 
            net_input = net_input
        )
        optimize(
            optimizer_type = optimizer, 
            parameters = p, 
            closure = closure, 
            LR = lr, 
            num_iter = num_iter
        )
    peak_mem = t.torch_peak/2**20
    print("Peak memory: {} MB".format(peak_mem))
    
    # Save PSNR history
    with open(os.path.join(output_dir, 'psnr_history.json'), 'w') as f:
        json.dump(psnr_history, f)

    # Extract max PSNR values
    psnr_np = np.array(psnr_history)
    last_col = psnr_np[:, -1]
    max_value = np.max(last_col)
    max_index = np.argmax(last_col)
    print("Max PSNR_Y:", max_value)
    print("At iteration:", max_index)
    wandb.log({
        "max_psnr_y": max_value,
        "max_index": max_index,
    })

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
