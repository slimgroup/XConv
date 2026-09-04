import os
import re
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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


from genparihaka import (
    SeismicDDPM,
)
from only_comp_avg_grad_err import (
    compute_avg_grad_err,
    get_conv_param_names
)
from pyxconv.utils import convert_net


import json


"""
    This script computes the mini_batch gradients and the average gradient error of the 
    Genparihaka[UNet] model for the given image dimension.

Usage:
    conda activate Genparaihaka
    python scripts/compute_mini_batch_gradients_avg_grad_err.py
    python scripts/compute_mini_batch_gradients_avg_grad_err.py --config configs/mini_batch_grad_256.json

    All run settings come from configs/ddpm_128.json and configs/mini_batch_grad_128.json
    (or the path passed to --config). Only --config is accepted on the command line.
"""

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDPM_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "ddpm_128.json")

# Change the config file here.
mini_batch_GRAD_DEFAULT_CONFIG = os.path.join(_PROJECT_ROOT, "configs", "xconv_configs/mini_batch_grad_128_pv2048.json")

print("Using {}".format(mini_batch_GRAD_DEFAULT_CONFIG))

_REQUIRED_CONFIG_KEYS = (
    "img_size",
    "num_ch",
    "batch_size",
    "model_name",
    "grad_dict_dir",
    "full_grads_path",
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
            "Compute the mini_batch gradients of Genparaihaka[UNet] for the given image dimension."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=mini_batch_GRAD_DEFAULT_CONFIG,
        help=(
            "Gradient-run JSON (default: configs/mini_batch_grad_128.json), merged on top "
            "of configs/ddpm_128.json."
        ),
    )
    cli_args = parser.parse_args()

    defaults = load_merged_defaults(cli_args.config)

    missing = [
        key for key in _REQUIRED_CONFIG_KEYS if _cfg(defaults, key, None) is None
    ]
    if missing:
        parser.error(
            "Missing required config keys (set in configs/mini_batch_grad_*.json): "
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


def comp_mini_batch_and_avg_grad_err(
    score_model,
    data_loader,
    conv_param_names,
    full_grad_param_dict,   
    noise_scheduler,
    loss_reduction,
    device
):  
    """
    Computes the mini-batch gradients for the UNet[Genparaihaka] and uses it to compute the average 
    gradient error.

    This function performs a forward and backward pass for each mini-batch,
    accumulates the gradients, and then computes the average gradient error
    with respect to the provided full dataset gradients.

    Args:
        score_model: The neural network model (UNet[Genparaihaka]) for which gradients are computed.
        data_loader: DataLoader providing mini-batches of data.
        conv_param_names: Set of parameter names corresponding to convolutional layers.
        full_grad_param_dict: Dictionary containing full dataset gradients for comparison.
        loss_reduction: The reduction type for the loss.
        device: The device (CPU or GPU) on which computations are performed.

    
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

    subset_size = args.subset_size
    batch_size = args.batch_size
    probing_vector = args.probing_vector
    grad_dict_dir = args.grad_dict_dir
    full_grads_path = args.full_grads_path
    t_folder = args.t_folder
    z_folder = args.z_folder
    img_folder = args.img_folder

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    # Load the full gradient parameter dictionary
    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)

    print("Loading full gradients from {}".format(full_grads_path))
    print("Computing mini-batch gradients and average gradient error for batch size: {}".format(batch_size))

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

    print("Processing probing_vector: {}".format(probing_vector))

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    base = (probing_vector == 'base')

    if not base: convert_net(
        score_model, 
        ps = probing_vector,
        xmode ='independent'
    )

    conv_param_names = get_conv_param_names(score_model)

    # Compute mini-batch gradients and use it to compute the average gradient error.
    avg_err = comp_mini_batch_and_avg_grad_err(
        score_model = score_model,
        data_loader = loader,
        conv_param_names=conv_param_names,
        full_grad_param_dict=full_grad_param_dict,
        noise_scheduler  = noise_scheduler,
        loss_reduction = loss_reduction,
        device = device
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