import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Resolve the monorepo root independently of the caller's working directory.
MONOREPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if MONOREPO_ROOT not in sys.path:
    sys.path.insert(0, MONOREPO_ROOT)

import pickle
import csv
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import argparse
import pandas as pd
from torchvision import datasets, transforms


from pyxconv.nvidia_mem_tracker import MemoryTracker

from pyxconv.utils import convert_net 
from pyxconv.mem_logger import log_mem
from diffusers import UNet2DModel, DDPMScheduler

"""
Usage:
    python3 compute_peak_memory.py --root_dir /path/to/output [--num_epochs 14]

Arguments:
    --root_dir      (required) Root directory for logs and model checkpoints
    --num_epochs    (default=14) Number of training epochs
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute peak memory via the new memory tracker.'
    )
    parser.add_argument('--in_channel_size', type=int, default=1, help='Number of input channels.')
    parser.add_argument(
        '--mem_log_dir', 
        type=str, 
        required=True,
        help='Path to save the final memory heatmap'
    )
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment Variation.')
    parser.add_argument('--batch_size', type=int, default=14, help='Batch size')
    parser.add_argument('--probing_vector', type=str, help='Probing vector')
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--drop', type=float, default=0, metavar='PCT',
                        help='Drop rate (default: 0.0)')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--img_dim', default=224, type=int,
                        help='input image dimension')
    parser.add_argument(
        '--xconv_varn', 
        type=str, 
        required=True, 
        help='Type of Xconv variation (adaptive/xconv).'
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

    args = parser.parse_args()

    return args


def compute_iteration_memory(
    score_model, 
    train_loader, 
    device, 
    noise_scheduler,
    batch_size, 
    img_dim,
    in_channel_size
):
    score_model.train()

    for i, (images, _) in tqdm(enumerate(train_loader)):

        # images: (B, C, H, W) eg. (128, 1, 28, 28)
        # labels: (B) eg. (128)

        if i > 1:
            break

        B = images.shape[0]
        dummy_inps = torch.randn(batch_size, in_channel_size, img_dim, img_dim).to(device)

        images = dummy_inps.to(device)


        # Forward pass
        # (B, num_c) eg. (128, 10)
        with MemoryTracker() as t:

            noise = torch.randn(
                images.shape,
                device=device
            )

            timesteps = torch.randint(
                0,
                len(noise_scheduler),
                (B,),
                device=device,
            ).long()

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noisy_images = noise_scheduler.add_noise(
                    images,
                    noise,
                    timesteps,
            )

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noise_pred = score_model(
                noisy_images,
                timesteps,
                return_dict=False,
            )[0]

            loss = torch.norm(noise_pred - noise) ** 2

            loss.backward()

        peak_mem = t.torch_peak/2**20
        # print(peak_mem)

    return peak_mem


def main(args):    

    transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
            ])

    dataset1 = datasets.MNIST(
        '../data', 
        train=True, 
        download=True,
        transform=transform
    )

    # Define hyperparameters
    num_epochs = 1
    in_channel_size = args.in_channel_size
    img_dim = args.img_dim
    exp_name = args.exp_name
    xconv_varn = args.xconv_varn
    

    # batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    batch_sizes = [args.batch_size]
    probing_vectors = [args.probing_vector]
    # probing_vectors = ['base', 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rows = []

    # Main training loop
    for batch_size in batch_sizes:
        print("Processing batch-size: {}".format(batch_size))
        train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
        
        for probing_vector in probing_vectors:
            base = (probing_vector == 'base')
            if not base:
                probing_vector = int(probing_vector)
            print("Processing probing_vector: {}".format(probing_vector))

            # Initialize the model and convert it based on the probing vector
            model = UNet2DModel(
                in_channels=1,
                out_channels=1,
                sample_size=(img_dim, img_dim),
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


            if not base: 
                if "adaptive" in xconv_varn:
                    print("Inside Adaptive XConv.")
                    adaptive_convert_net(
                        model, 
                        # Need only 1 Batch-size to perform checks.
                        input_shape=(1, in_channel_size, img_dim, img_dim),
                        ps=probing_vector, 
                        xmode='independent'
                    )    
                else:
                    convert_net(
                    model, 
                    ps = probing_vector,
                    xmode ='independent'
                )
                
            print(model)

            for _ in tqdm(range(num_epochs)):

                peak_mem = compute_iteration_memory(   
                    score_model=model,
                    train_loader=train_loader,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    batch_size=batch_size,
                    img_dim=img_dim,
                    in_channel_size=in_channel_size
                )
                
                rows.append({
                    "batch_size": batch_size,
                    "probing_vector": probing_vector,
                    "peak_memory": peak_mem
                })
                
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    mem_bs_dir = "{}/{}_batch".format(args.mem_log_dir, batch_size)   
    if not os.path.exists(mem_bs_dir):
        os.makedirs(mem_bs_dir)     
    memory_csv_file = "{}/{}_batch_size_{}_probing_vector_{}_peak_memory.csv".format(
        mem_bs_dir, exp_name, batch_size, probing_vector)
    df = pd.DataFrame(rows)
    df.to_csv(
        memory_csv_file,
        columns = ['batch_size', 'probing_vector', 'peak_memory'],
        index = False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
