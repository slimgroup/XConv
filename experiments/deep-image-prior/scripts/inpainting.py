from __future__ import print_function
import matplotlib.pyplot as plt
import os
import argparse
import wandb
import numpy as np
from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
import torch
from pyxconv.utils import convert_net
from projorg import make_experiment_name
from torchmetrics import PeakSignalNoiseRatio

from utils.inpainting_utils import *

dtype = torch.cuda.FloatTensor

"""
    Inpainting the image using the deep-image-prior

    Usage:
        sh bash_scripts/bash_inpainting.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Example script for inpainting using deep-image-prior'
    )
    parser.add_argument(
        '--probing_vector',
        type=str,
        default='base',
        help='Probing vector. If it is a digit, it will be converted to an integer. If it is "base", it will not be converted.',
    )
    parser.add_argument(
        '--img_name',
        type=str,
        help='Image name',
        required=True
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Output directory',
        required=True
    )
    parser.add_argument(
        '--imsize', 
        type=int, 
        default=-1, 
        help='Image size',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        required=True,
        help='Optimizer',
    )
    parser.add_argument(
        '--pad',
        type=str,
        default='reflection',
        help='Padding',
    )
    parser.add_argument(
        '--experiment_name',
        type=str,   
        required=True,
        help='Experiment name',
    )
    parser.add_argument(
        '--opt_over',
        type=str,
        default='net',
        help='Opt over',
    )
    parser.add_argument(
        '--img_path',
        type=str,
        help='Path to image',
        required=True
    )
    parser.add_argument(
        '--plot', 
        type=bool, 
        default=True, 
        help='Plot the results',
    )
    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        required=True,
    )
    parser.add_argument(
        '--num_iter',
        type=int,
        help='Number of optimization steps',
        required=True
    )
    parser.add_argument(
        '--mask_path',
        type=str,
        help='Path to mask',
        required=True
    )
    parser.add_argument(
        '--dim_div_by',
        type=int,
        help='Dimension to divide by',
        required=False,
        default=64,
    )
    parser.add_argument(
        '--net_type',
        type=str,
        default='skip_depth6',
        help='Net type (skip, unet, resnet)',
    )
    args = parser.parse_args()
    return args

def setup_model(
    img_path, 
    img_np,
    pad, 
    net_type
):
    if 'vase.png' in img_path:
        INPUT = 'meshgrid'
        input_depth = 2
        LR = 0.01 
        num_iter = 5001
        param_noise = False
        show_every = 50
        figsize = 5
        reg_noise_std = 0.03
        
        net = skip(
            input_depth, 
            img_np.shape[0], 
            num_channels_down = [128] * 5,
            num_channels_up   = [128] * 5,
            num_channels_skip = [0] * 5,  
            upsample_mode='nearest', 
            filter_skip_size=1, 
            filter_size_up=3, 
            filter_size_down=3,
            need_sigmoid=True, 
            need_bias=True, 
            pad=pad, 
            act_fun='LeakyReLU'
        ).type(dtype)
    
    if ('kate.png' in img_path) or ('peppers.png' in img_path):
        INPUT = 'noise'
        input_depth = 32
        LR = 0.01 
        num_iter = 6001
        param_noise = False
        show_every = 1000
        figsize = 5
        reg_noise_std = 0.03
        
        net = skip(
            input_depth, 
            img_np.shape[0], 
                num_channels_down = [128] * 5,
                num_channels_up =   [128] * 5,
                num_channels_skip =    [128] * 5,  
                filter_size_up = 3, filter_size_down = 3, 
                upsample_mode='nearest', filter_skip_size=1,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
        
    elif 'library.png' in img_path:
        
        INPUT = 'noise'
        input_depth = 1
        
        num_iter = 3001
        show_every = 50
        figsize = 8
        reg_noise_std = 0.00
        param_noise = True
        
        if 'skip' in net_type:
            
            depth = int(net_type[-1])
            net = skip(input_depth, img_np.shape[0], 
                num_channels_down = [16, 32, 64, 128, 128, 128][:depth],
                num_channels_up =   [16, 32, 64, 128, 128, 128][:depth],
                num_channels_skip =    [0, 0, 0, 0, 0, 0][:depth],  
                filter_size_up = 3,filter_size_down = 5,  filter_skip_size=1,
                upsample_mode='nearest', # downsample_mode='avg',
                need1x1_up=False,
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
            
            LR = 0.01 
            
        elif net_type == 'UNET':
            
            net = UNet(num_input_channels=input_depth, num_output_channels=3, 
                    feature_scale=8, more_layers=1, 
                    concat_x=False, upsample_mode='deconv', 
                    pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)
            
            LR = 0.001
            param_noise = False
            
        elif net_type == 'ResNet':
            
            net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')
            
            LR = 0.001
            param_noise = False
            
        else:
            assert False
            
    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    
    return net, net_input, LR, num_iter, param_noise, show_every, figsize, reg_noise_std

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
    lr = args.lr
    img_name = args.img_name

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
    
    if hasattr(args, "lr"):
        lr = args.lr
    if hasattr(args, "num_iter"):
        num_iter = args.num_iter
    else:
        args.num_iter = num_iter
    args.reg_noise_std = reg_noise_std
    
    experiment_name = make_experiment_name(
        args,
        ignore_arg_list=[
            "experiment_name",
            "imsize",
            "img_path",
            "enforce_div32",
            "input_depth",
            "opt_over",
            "kernel_type",
            "mask_path",
            "tv_weight",
            "plot",
            "input_image_path",
            "net_type",
            "output_dir",
        ]
    )

    print("W&B run name:", experiment_name)
    wandb.init(
        project="inpainting",
        name=experiment_name,
        config=args
    )

    wandb.save('*.py')
    wandb.save('bash_scripts/*')
    wandb.save('configs/*')

    output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

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

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()

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
    best_psnr = -1e9
    best_path = None

    def closure():
        nonlocal i, best_psnr, best_path
        
        
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50
        
        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            
            
        out = net(net_input)
    
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()

        psnr_score = psnr_fn(preds = out, target = img_var).item()
    
        print ('Iteration %05d    Loss %f   PSNR %3f' % (i, total_loss.item(), psnr_score), '\n', end='')
        wandb.log({
            "loss": total_loss.item(),
            "iter": i,
            "psnr_score": psnr_score
        })
        # Save the best PSNR image.
        if psnr_score > best_psnr:
            best_psnr = psnr_score
            out_np = torch_to_np(out)
            grid = plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
            best_path = os.path.join(
                output_dir, 
                f"best_inpainting_iteration_{i}_psnr_{best_psnr}.png")
            print("Saving {}".format(best_path) )
            
            if grid.shape[0] == 1:
                best_img = grid[0]
                best_img = (best_img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(best_img, mode="L").save(best_path)
            else:
                best_img = grid.transpose(1, 2, 0)
                best_img = (best_img * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(best_img).save(best_path)

        if  i % show_every == 0:
            out_np = torch_to_np(out)
            grid = plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
            output_path = os.path.join(output_dir, "inpainting_iteration_%05d.png" % i)
            
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
    optimize(optimizer, p, closure, lr, num_iter)

    out_np = torch_to_np(net(net_input))
    grid = plot_image_grid([out_np], factor=5)

    output_path = os.path.join(args.output_dir, "inpainting_iteration_%05d.png" % i)
    if grid.shape[0] == 1:
        img = grid[0]
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(output_path)
    else:
        img = grid.transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_path)

    wandb.log({
        "best_psnr": best_psnr,
        "best_psnr_path": best_path
    })
    wandb.finish()
    


if __name__ == '__main__':
    args = parse_args()
    main(args)