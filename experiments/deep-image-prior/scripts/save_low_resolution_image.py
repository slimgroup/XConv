from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from models import *
from models.downsampler import Downsampler
from utils.sr_utils import *

dtype = torch.cuda.FloatTensor

"""
    Downsample the HR image to the LR image and save it.

    Usage:
        sh bash_scripts/bash_save_low_resolution_image.sh
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downsample the HR image to the LR image and save it'
    )
    parser.add_argument('--kernel_type', type=str, default='lanczos2',
                        help='Downsampler kernel type (lanczos2 in DIP SR)')


    parser.add_argument('--input_image_path', type=str, required=True,
                        help='Path to HR image (e.g. SR/Set5/original/baby.png)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the LR image')
    parser.add_argument('--save_name', type=str, required=True,
                        help='Name to save the LR image')
    parser.add_argument('--factor', type=int, default=4,
                        help='Downscaling factor (4 for Table 1)')
    parser.add_argument('--imsize', type=int, default=-1,
                        help='Optional HR resize; -1 = no resize')
    parser.add_argument('--enforce_div32', type=str, default='CROP',
                        help='"CROP" crops HR to be divisible by 32')

    args = parser.parse_args()
    return args


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

def save_img_np(img_np, path):
    # CHW → HWC if needed
    if img_np.ndim == 3 and img_np.shape[0] in (1, 3):
        img_np = np.transpose(img_np, (1, 2, 0))

    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)

    Image.fromarray(img_uint8).save(path)

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
    hr_image_path = args.input_image_path
    save_dir = args.save_dir
    save_name = args.save_name
    kernel_type = args.kernel_type
    enforce_div32 = args.enforce_div32

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # === 1) Load HR, generate LR (we ignore LR inside loader later) ===
    imgs = load_LR_HR_imgs_sr(
        fname=hr_image_path,
        imsize=imsize,
        factor=factor,
        enforce_div32=enforce_div32
    )

    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(
        imgs['LR_pil'], imgs['HR_pil']
    )

    print("HR size:", imgs['HR_pil'].size, "| LR size:", imgs['LR_pil'].size)

    # === 3) Downsampler & synthetic LR ===
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
        # (B, C, H, W) eg. (1, 3, 72, 72)
        img_LR_var = downsampler(img_HR_var).detach()

    img_LR_np = torch_to_np(img_LR_var)  # (3,h,w) in [0,1]
    img_LR_up_np = pil_to_np(np_to_pil(img_LR_np).resize(imgs['HR_pil'].size, resample=Image.BICUBIC))

    save_img_path = os.path.join(save_dir, save_name + ".png")
    print(f"Saving LR image to {save_img_path}")
    save_img_np(img_LR_up_np, save_img_path)




if __name__ == "__main__":
    args = parse_args()
    main(args)
