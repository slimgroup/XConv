import os
from typing import Any
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import argparse
from torchvision import models

from pyxconv.utils import convert_net
from diffusers import DDPMScheduler, UNet2DModel
from only_comp_avg_grad_err import get_conv_param_names, compute_avg_grad_err
from compute_full_gradients import StoredDataset

"""
    This script computes the mini-batch gradients and the average gradient error of the SIPS[UNet] model for the given image dimension.
    Usage:
        conda activate sips
        sh bash_scripts/bash_compute_mini_batch_gradients_avg_grad_err.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the mini-batch gradients and the average gradient error of SIPS[UNet] for the given image dimension.'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument(
        '--subset_size', 
        type=int, 
        default=1024, 
        help='Subset size selected for this experiment.'
    )
    parser.add_argument(
        '--num_ch', 
        type=int, 
        required=True, 
        help='Number of Channels'
    )
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True, help='Probing-vector')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--drop', type=float, default=0, metavar='PCT',
                        help='Drop rate (default: 0.0)')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument(
        '--full_grads_path', 
        type=str, 
        required=True, 
        help='Path to the pickle file containing the full gradients.'
    )
    parser.add_argument(
        '--grad_dict_dir', 
        type=str, 
        required=True, 
        help='Path to the directory containing the dict containing gradients.'
    )
    parser.add_argument(
        '--xconv_varn', 
        type=str, 
        required=True, 
        help='Type of Xconv variation (adaptive/xconv).'
    )
    parser.add_argument(
        '--seed', 
        type=int,
        default=12, 
        help='Seed'
    )
    parser.add_argument(
         '--time_emb', 
         type=str,
        default='positional', 
        help='Time Embedding'
    )
    parser.add_argument(
         '--attn_dim', 
         type=int,
        default=8, 
        help='Attention Dimension'
    )
    parser.add_argument(
         '--block_channels', 
         type=list,              
        default=[64, 128, 192], 
        help='Block Channels'       
    )
    parser.add_argument(
         '--block_nlayers', 
         type=int,
        default=2,
        help='Block NLayers'
    )   
    parser.add_argument(
         '--nt',
        type=int,
        default=1000,
        help='Number of diffusion timesteps'
    )
    parser.add_argument(
         '--beta_schedule',
        type=str,
        default='linear',
        help='Beta Schedule'
    )
    parser.add_argument(
         '--t_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "t" vectors for the corresponding image-dimension.'
    )    
    parser.add_argument(
         '--z_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "z" vectors for the corresponding image-dimension.'
    )    
    parser.add_argument(
        '--img_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "img" vectors for the corresponding image-dimension.'
    )
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
    )
    args = parser.parse_args()

    return args


def comp_mini_batch_and_avg_grad_err(
    score_model,
    data_loader,
    conv_param_names,
    full_grad_param_dict,   
    noise_scheduler,
    loss_reduction,
    device,
    bf16_precision:bool = False
):  
    """
    Computes the mini-batch gradients for the UNet[SIPs] and uses it to compute the average gradient error.

    This function performs a forward and backward pass for each mini-batch,
    accumulates the gradients, and then computes the average gradient error
    with respect to the provided full dataset gradients.

    Args:
        score_model: The neural network model (UNet[SIPs]) for which gradients are computed.
        data_loader: DataLoader providing mini-batches of data.
        conv_param_names: Set of parameter names corresponding to convolutional layers.
        full_grad_param_dict: Dictionary containing full dataset gradients for comparison.
        loss_reduction: The reduction type for the loss.
        device: The device (CPU or GPU) on which computations are performed.
        bf16_precision (bool): If true, convert the image tensor to bf-16.
    
    Returns:    
        avg_grad_err: The average L2 gradient error across all convolutional parameters.
    """
    score_model.train() 
    score_model.zero_grad()  
    batch_errors = []

    for i, (images, zs, ts) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):
            
            score_model.zero_grad() # Zero the gradients before each batch

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            images = images.to(device)

            batch_size = images.size(0)
            
            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noise = zs.to(device)

            # B eg. 5
            timesteps = ts.long().to(device)

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noisy_images = noise_scheduler.add_noise(
                images,
                noise,
                timesteps,
            )

            if bf16_precision:
                noisy_images = noisy_images.half()

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noise_pred = score_model(
                noisy_images,
                timesteps,
                return_dict=False,
            )[0]

            # loss = torch.norm(noise_pred - noise) ** 2
            # loss = loss.mean()

            # if loss_reduction == "sum":
            #     loss /= batch_size  # Scale loss to reflect mini-batch.
            # elif loss_reduction == "mean":
            #     pass # Loss is already scaled by batch size.
            # else:
            #     raise ValueError(f"Invalid loss reduction: {loss_reduction}")

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            diff = noise_pred - noise

            # per-sample loss: average over CHW for each item
            # (B,) eg. (5,)
            per_sample = diff.pow(2).flatten(1).mean(dim=1)   # shape (B,)

            # if loss_reduction == "sum":
            #     loss /= total_samples  # Scale loss to reflect full dataset
            # elif loss_reduction == "mean":
            #     loss /= batch_size
            # else:
            #     raise ValueError(f"Invalid loss reduction: {loss_reduction}")
            if loss_reduction == "mean":
                # unbiased estimate of dataset mean gradient
                pass
            elif loss_reduction == "sum":
                # unbiased estimate of dataset *sum* gradient; scale to dataset average
                loss = per_sample.sum() / batch_size
            else:
                raise ValueError("Invalid loss_reduction")


            loss.backward()
            
            batch_grads = {
                name: param.grad.clone().cpu().detach()
                for name, param in score_model.named_parameters()
                if param.requires_grad and param.grad is not None
            }

            # Get the gradient error for this batch.
            batch_grad_err = compute_avg_grad_err(
                mini_batch_grads = {i : batch_grads},
                full_model_grads = full_grad_param_dict,
                model_layer = None,
                conv_param_names= conv_param_names,
            )
            print("For batch {}, the gradient error is: {}".format(i, batch_grad_err))
            batch_errors.append(batch_grad_err)

    
    return np.mean(batch_errors)

def main(args):    

    img_size = args.img_size
    subset_size = args.subset_size
    batch_size = args.batch_size
    probing_vector = args.probing_vector
    model_name = args.model_name
    num_ch = args.num_ch
    full_grads_path = args.full_grads_path
    grad_dict_dir = args.grad_dict_dir
    t_folder = args.t_folder
    z_folder = args.z_folder
    img_folder = args.img_folder
    bf16_precision = args.bf16_precision

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    print("Reading full gradients from {}".format(full_grads_path))
    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a stored dataset that returns the Images, t's and z's.
    stored_dataset = StoredDataset(
        img_folder = img_folder,
        z_folder = z_folder,
        t_folder = t_folder,
        subset_size = subset_size,
        device = device
    )
    
    loader = torch.utils.data.DataLoader(
        stored_dataset, 
        batch_size=batch_size
    )
    
    # Initialize the model and convert it based on the probing vector
    if model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=False).to(device) # 408 MiB [includes 200 MiB allocated when CuDA is initialized]
    elif "vanillanet" in model_name:
        # Initialize the model and convert it based on the probing vector
        model = create_model(
                model_name, 
                pretrained=False,
                num_classes=args.nb_classes, 
                act_num=args.act_num,
                drop_rate=args.drop,
                deploy=False,
        )
    elif "sips_unet" in model_name:
         model = UNet2DModel(
            in_channels=num_ch,
            out_channels=1,
            sample_size=(img_size, img_size),
            time_embedding_type=args.time_emb,
            attention_head_dim=args.attn_dim,
            block_out_channels=args.block_channels,
            layers_per_block=args.block_nlayers,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
         
         noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.nt,
            beta_schedule=args.beta_schedule,
        )

    model.to(device)

    print("Processing probing_vector: {}".format(probing_vector))
    base = (probing_vector == 'base')

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    if bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    if not base: convert_net(
        model, 
        ps = probing_vector,
        xmode ='independent'
    )
    
    print(model)
        
    conv_param_names = get_conv_param_names(model)
    loss_reduction = "sum"
             
    # Compute mini-batch gradients and use it to compute the average gradient error.
    avg_err = comp_mini_batch_and_avg_grad_err(
        score_model = model,
        data_loader = loader,
        conv_param_names=conv_param_names,
        full_grad_param_dict=full_grad_param_dict,
        noise_scheduler  = noise_scheduler,
        loss_reduction = loss_reduction,
        device = device,
        bf16_precision = bf16_precision
    )

    print(f"Avg L2 gradient error: {avg_err}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    print("Saving average gradient error at: {}".format(avg_grad_err_file_path))
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_err, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)