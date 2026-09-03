# scripts/super_resolution_table1.py

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
import torchmetrics

from models import *
from utils.sr_utils import *

dtype = torch.cuda.FloatTensor
from super_resolution_table1 import psnr_y

"""
    Computing the baseline performance (bicubic/bilinear) for Super-Resolution
    task.

    Usage:
        sh bash_scripts/bash_compute_baseline_psnr.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Computing the PSNR for baselines: (bicubic/bilinear) in' \
        'Super-Resolution task.'
    )
    parser.add_argument('--input_image_path', type=str, required=True,
                        help='Path to HR image (e.g. SR/Set5/original/baby.png)')
    parser.add_argument('--factor', type=int, default=4,
                        help='Downscaling factor (4 for Table 1)')
    parser.add_argument('--imsize', type=int, default=-1,
                        help='Optional HR resize; -1 = no resize')
    parser.add_argument('--enforce_div32', type=str, default='CROP',
                        help='"CROP" crops HR to be divisible by 32')
    return parser.parse_args()



def main(args):
    imsize = args.imsize
    factor = args.factor
    enforce_div32 = args.enforce_div32
    hr_image_path = args.input_image_path

    if hasattr(args, "probing_vector"):
        probing_vector = args.probing_vector
    else:
        probing_vector = "base"

    # probing_vector: base vs XConv
    if isinstance(probing_vector, str) and probing_vector.isdigit():
        probing_vector = int(probing_vector)

    # === 1) Load HR, generate LR (we ignore LR inside loader later) ===
    imgs = load_LR_HR_imgs_sr(
        fname=hr_image_path,
        imsize=imsize,
        factor=factor,
        enforce_div32=enforce_div32
    )

    psnr_fn = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).cuda()

    baseline_imgs = get_baselines(
        imgs['LR_pil'], imgs['HR_pil']
    )

    imgs['bicubic_np'] = baseline_imgs['bicubic_np']
    imgs['bilinear_np'] = baseline_imgs['bilinear_np']

    # Ground-truth HR tensor
    img_HR_var = np_to_torch(imgs['HR_np']).type(dtype)

    # Bicubic tensor
    img_bicubic = np_to_torch(imgs['bicubic_np']).type(dtype)

    psnr_bicubic = psnr_fn(img_bicubic, img_HR_var).item()
    print(f"PSNR_Bicubic: {psnr_bicubic:0.3f}")

    # Bilinear tensor
    img_bilinear = np_to_torch(imgs['bilinear_np']).type(dtype)

    psnr_bilinear = psnr_fn(img_bilinear, img_HR_var).item()
    print(f"PSNR_Bilinear: {psnr_bilinear:0.3f}")

    # psnr_bicubic_y = psnr_y(
    #     imgs['bicubic_np'],
    #     imgs['HR_np'],
    #     crop=factor
    # )
    # print(f"PSNR_Bicubic_Y: {psnr_bicubic_y:0.3f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
