import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import argparse
from torchvision import datasets, transforms, models

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

from genparihaka import (
    SeismicDDPM,
    load_parihaka,
    plot_grid,
    plot_losses,
)


import json


"""
    This script computes the full gradients of the Genparihaka[UNet] model for the given
    image dimension.

Usage:
    conda activate sips
    python scripts/compute_full_gradients.py
    python scripts/compute_full_gradients.py --config configs/xconv_configs/runs/full_grad_runs/full_grad_128_run_1.json

    All run settings come from configs/ddpm_128.json and the gradient JSON passed to
    --config (default: configs/xconv_configs/full_grad_128.json). Only --config is accepted
    on the command line. For multi-run sweeps, use the per-run files under
    configs/xconv_configs/runs/full_grad_runs/ (see bash_scripts/bash_seq_comp_avg_grad_err.sh).
"""

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDPM_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "ddpm_128.json")
FULL_GRAD_DEFAULT_CONFIG = os.path.join(
    _PROJECT_ROOT, "configs", "xconv_configs", "full_grad_128.json"
)

_REQUIRED_CONFIG_KEYS = (
    "img_size",
    "num_ch",
    "batch_size",
    "model_name",
    "base_grad_dict_dir",
    "t_folder",
    "z_folder",
    "img_folder",
)


def load_config_defaults(config_path: str) -> dict:
    path = os.path.expanduser(config_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return json.load(f)


def normalize_config_keys(config: dict) -> dict:
    """Align ddpm_128.json keys with this script's argparse names."""
    out = dict(config)
    if "image_size" in out and "img_size" not in out:
        out["img_size"] = out["image_size"]
    if "batchsize" in out and "batch_size" not in out:
        out["batch_size"] = out["batchsize"]
    if "dropout" in out and "drop" not in out:
        out["drop"] = out["dropout"]
    return out


def load_merged_defaults(grad_config_path: str | None) -> dict:
    """Load configs/ddpm_128.json, then overlay an optional gradient-run config."""
    paths = [DDPM_DEFAULT_CONFIG]
    if grad_config_path:
        paths.append(grad_config_path)
    merged = {}
    for path in paths:
        merged.update(normalize_config_keys(load_config_defaults(path)))
    return merged


def _cfg(defaults: dict, key: str, fallback):
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    return fallback


def parse_block_channels(value) -> list:
    if value is None:
        raise ValueError("block_channels must be set in configs/ddpm_128.json")
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [
            int(part)
            for part in value.replace(" ", "").split(",")
            if part
        ]
    return [int(value)]


def normalize_args(args) -> None:
    """Map config keys to SeismicDDPM fields and coerce list-like config values."""
    if getattr(args, "img_size", None) is None:
        args.img_size = args.image_size
    args.image_size = args.img_size

    if getattr(args, "dropout", None) is None:
        args.dropout = args.drop
    if getattr(args, "drop", None) is None:
        args.drop = args.dropout

    args.block_channels = parse_block_channels(args.block_channels)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the full gradients of SIPS[UNet] for the given image dimension."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=FULL_GRAD_DEFAULT_CONFIG,
        help=(
            "Gradient-run JSON (default: configs/xconv_configs/full_grad_128.json), "
            "merged on top of configs/ddpm_128.json."
        ),
    )
    cli_args = parser.parse_args()

    defaults = load_merged_defaults(cli_args.config)

    missing = [
        key for key in _REQUIRED_CONFIG_KEYS if _cfg(defaults, key, None) is None
    ]
    if missing:
        parser.error(
            "Missing required config keys (set in configs/full_grad_*.json): "
            + ", ".join(missing)
        )

    args = argparse.Namespace(**defaults)
    args.config = cli_args.config
    normalize_args(args)
    return args

_num = re.compile(r'(\d+)')

def natural_key(fname: str) -> int:
    m = _num.search(fname)
    if not m:
        raise ValueError(f"No integer found in {fname}")
    return int(m.group(1))


class StoredDataset(torch.utils.data.Dataset):
    """
        A dataset that returns the stored Images, z's and t's.
    """
    def __init__(
          self, 
          img_folder, 
          z_folder,
          t_folder,
          subset_size,
          device
    ):
        self.img_folder = img_folder
        self.z_folder = z_folder
        self.t_folder = t_folder
        self.device = device
        self.subset_size = subset_size
        self.num_files = subset_size

        assert self.num_files == self.subset_size

        # Sort for deterministic order
        # List only the relevant files, then sort numerically by the embedded integer
        img_files = [f for f in os.listdir(img_folder) if f.endswith('.pt') and f.startswith('img_')]
        img_files.sort(key=natural_key)  # <- numeric sort
        self.img_files = img_files[:subset_size]
        print(self.img_files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        # Extract image index from filename
        file_name = self.img_files[idx]
        base_name = os.path.splitext(file_name)[0]  # "image_716973"
        file_idx = base_name.split("_")[-1]         # "716973"

        # (C, H, W) eg. (3, 1024, 1024)
        img_path = os.path.join(self.img_folder, "img_{}.pt".format(file_idx))
        img = torch.load(img_path)

        # t: (1) eg. (1)

        t_path = os.path.join(self.t_folder, f"t_{file_idx}.pt")
        t = torch.load(t_path)

        z_path = os.path.join(self.z_folder, f"z_{file_idx}.pt")
        z = torch.load(z_path)

        return img.squeeze(0), z.squeeze(0), t.squeeze(0)


def comp_full_gradient_dict(
    score_model, 
    data_loader, 
    noise_scheduler,
    loss_reduction,
    device
):
    """
    Computes the full dataset gradient for the UNet[GenParihaka] by accumulating gradients 
    over all mini-batches.

    This function performs a forward and backward pass for each mini-batch,
    scales each batch loss appropriately so the accumulated gradients reflect
    the true average gradient over the entire dataset.

    Args:
        score_model (torch.nn.Module): The model whose gradients are to be computed.
        data_loader (DataLoader): DataLoader for the full dataset.
        loss_reduction (str): The reduction type for the loss.
        noise_scheduler (DDPMScheduler): The noise scheduler used in the diffusion process.
        device (torch.device or str): Device to run computation on (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary mapping parameter names to their accumulated gradient tensors.
              Gradients are detached from the graph and cloned.
    """

    score_model.train()
    score_model.zero_grad() # Erase gradients before performing this computation.
    grad_dict = {}

    total_samples = len(data_loader.dataset)
    
    for images, zs, ts in tqdm(data_loader, desc="Computing Full Gradient"):
        
            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            images = images.to(device)
            
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

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noise_pred = score_model(
                noisy_images,
                timesteps,
                return_dict=False
            )[0]

            # loss = torch.norm(noise_pred - noise) ** 2
            # loss = loss.mean()

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
                loss = per_sample.mean()
            elif loss_reduction == "sum":
                # unbiased estimate of dataset *sum* gradient; scale to dataset average
                loss = per_sample.sum() / total_samples
            else:
                raise ValueError("Invalid loss_reduction")


            loss.backward()  # Accumulates gradients

    # Extract and clone gradients
    for name, param in score_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_dict[name] = param.grad.clone().cpu().detach()

    return grad_dict

def main(args):    

    subset_size = args.subset_size
    batch_size = args.batch_size
    base_grad_dict_dir = args.base_grad_dict_dir
    t_folder = args.t_folder
    z_folder = args.z_folder
    img_folder = args.img_folder
    seed = args.seed

    print(f"Computing full gradients for batch size: {batch_size}")

    if not os.path.exists(base_grad_dict_dir):
        os.makedirs(base_grad_dict_dir)

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
    seismic_model = SeismicDDPM(args).to(device)
    score_model = seismic_model.model
    noise_scheduler = seismic_model.noise_scheduler

    score_model.to(device)

    trainable_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    loss_reduction = "sum"
             
    #  Compute the full gradients and store the corresponding model weights in a dictionary.      
    full_grad_param_dict = comp_full_gradient_dict(
        score_model = score_model,
        data_loader = loader,
        noise_scheduler=noise_scheduler,
        loss_reduction = loss_reduction,
        device = device
    )

    final_path = os.path.join(base_grad_dict_dir, "full_grad_param_dict.pkl")    
    print("Saving full gradient dictionary at: {}".format(final_path))        
    with open(final_path, "wb") as f:
        pickle.dump(full_grad_param_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)