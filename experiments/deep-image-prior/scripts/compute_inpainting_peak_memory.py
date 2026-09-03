import os
import argparse
from utils.common_utils import *
from utils.inpainting_utils import *
from pyxconv.utils import convert_net
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from inpainting import setup_model
from utils.common_utils import optimize
from utils.common_utils import get_params
import pandas as pd
from pyxconv.nvidia_mem_tracker import MemoryTracker

dtype = torch.cuda.FloatTensor


"""
Usage:
    sh bash_scripts/bash_compute_inpainting_peak_memory.sh
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute peak memory of inpainting via the new memory tracker.'
    )
    parser.add_argument('--img_path', type=str, required=True, help='Path to image')
    parser.add_argument('--imsize', type=int, default=-1, help='Image size')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to mask')
    parser.add_argument('--dim_div_by', type=int, default=64, help='Dimension to divide by')
    parser.add_argument('--pad', type=str, required=True, help='Padding')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--net_type', type=str, default='skip_depth6', help='Net type')
    parser.add_argument('--opt_over', type=str, default='net', help='Opt over')
    parser.add_argument('--probing_vector', type=str, default='base', help='Probing vector')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    return args


def main(args):    
     
    img_path = args.img_path
    imsize = args.imsize
    mask_path = args.mask_path
    dim_div_by = args.dim_div_by
    pad = args.pad
    optimizer = args.optimizer
    net_type = args.net_type
    opt_over = args.opt_over
    probing_vector = args.probing_vector
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the image
    # (C, H, W) eg. (3, 512, 512)
    img_pil, img_np = get_image(img_path, imsize)

    # (C, H, W) eg. (1, 512, 512)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)

    # Crop the image

    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil      = crop_image(img_pil,      dim_div_by)

    img_np      = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    # probing_vector: base vs XConv
    if isinstance(probing_vector, str) and probing_vector.isdigit():
        probing_vector = int(probing_vector)
    base = (probing_vector == 'base')

    # IMPORTANT: xconv has issues with reflection padding causing dimension mismatches
    # during backpropagation. Use 'zero' padding instead when converting to xconv.
    pad_for_net = pad
    if not base and pad == 'reflection':
        print("WARNING: Reflection padding causes dimension mismatches with xconv.")
        print("Switching to 'zero' padding for xconv compatibility.")
        pad_for_net = 'zero'

    net, net_input, lr, num_iter, param_noise, show_every, figsize, reg_noise_std = setup_model(
        img_path=img_path,
        img_np=img_np,
        pad=pad_for_net,
        net_type=net_type
    )

    if not base:
        convert_net(net, ps=probing_vector, xmode='independent')
        print("Converted net to XConv with probing vector:", probing_vector)
    else:
        print("Using base Conv network.")
    print(net)


    # Compute number of parameters
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    i = 0
    num_iter = 1
    def closure():
        nonlocal i
        
        
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50
        
        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            
            
        out = net(net_input)
    
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
            
        if  i % show_every == 0:
            out_np = torch_to_np(out)
            grid = plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
            output_path = os.path.join(args.output_dir, "inpainting_iteration_%05d.png" % i)
            if grid.shape[0] == 1:
                img = grid[0]
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img, mode="L").save(output_path)
            else:
                img = grid.transpose(1, 2, 0)
                img = (img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(output_path)
            
        i += 1

        return total_loss

    p = get_params(opt_over, net, net_input)

    with MemoryTracker() as t:
        optimize(optimizer, p, closure, lr, num_iter)
    peak_mem = t.torch_peak/2**20
    print("Peak memory: {} MB".format(peak_mem))
    peak_memory_path = os.path.join(output_dir, "peak_memory.csv")
    print(f"Saving peak memory to {peak_memory_path}")
    pd.DataFrame({
        "probing_vector": [probing_vector],
        "peak_memory": [peak_mem]
    }).to_csv(peak_memory_path, index=False)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    args = parse_args()
    main(args)