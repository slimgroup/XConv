import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

import sys
import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import losses
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
from train import create_model
from dataset import Dataset
from albumentations import (
    RandomRotate90,
    Resize, 
    Flip, 
    Normalize
)
from albumentations.core.composition import Compose
from glob import glob
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import argparse
from torchvision import datasets, transforms, models

import json


"""
    Compute Full Gradient for the given image dimension for TriConvUNeXt model.

    Usage:
        sh bash_scripts/bash_comp_full_gradient.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Full gradient error for the given image dimension for the TriConvUNeXt model.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        required=True, 
        help='Batch-size'
    )

    parser.add_argument(
        '--probing_vector', 
        type=str, 
        required=True, 
        help='Probing-vector'
    )
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        help='Model name'
    )
    parser.add_argument(
        '--grad_dict_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where gradient dictionaries will be saved.'
    )

    parser.add_argument(
        '--img_dim', 
        type=int, 
        required=True, 
        help='Image dimension'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help='Dataset'
    )
    parser.add_argument(
        '--loss',
        default='BCEDiceLoss',
        choices=LOSS_NAMES,
        help='loss: ' +
        ' | '.join(LOSS_NAMES) +
        ' (default: BCEDiceLoss)'
    )
    parser.add_argument(
        '--num_classes', 
        type=int, 
        required=True, 
        help='Number of classes'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        required=True, 
        help='Seed to control randomness.'
    )
    parser.add_argument(
        '--run_num', 
        type=int, 
        required=True, 
        help='Run number.'
    )
    args = parser.parse_args()

    return args


def compute_full_gradient(
    model, 
    loss_fn,
    data_loader : torch.utils.data.DataLoader, 
    loss_reduction: str,
    device: torch.device
) -> dict:
    """
    Computes the full dataset gradient by accumulating gradients over all mini-batches.

    This function performs a forward and backward pass for each mini-batch,
    scales each batch loss appropriately so the accumulated gradients reflect
    the true average gradient over the entire dataset.

    Args:
        model (torch.nn.Module): The model whose gradients are to be computed.
        data_loader (DataLoader): DataLoader for the full dataset.
        loss_reduction (str): Reduction method used in the loss function ('mean' or 'sum').
        loss_fn (torch.nn.Module): Loss function to use for computing the gradient.
        device (torch.device or str): Device to run computation on (e.g., 'cuda' or 'cpu').
    Returns:
        dict: A dictionary mapping parameter names to their accumulated gradient tensors.
              Gradients are detached from the graph and cloned.
    """

    model.train()
    model.zero_grad() # Erase gradients before performing this computation.
    grad_dict = {}
    total_samples = len(data_loader.dataset)
    
    for images, target, _ in tqdm(data_loader, desc="Computing Full Gradient"):
        
        # images: (B, C, H, W) eg. (128, 3, 256, 256)
        # target: (B, H, W, C) eg. (128, 1, 256, 256)

        batch_size = images.size(0)
        
        images = images.to(device)

        target = target.to(device)

        # Forward pass
        # (B, num_classes, H, W) eg. (128, 1, 256, 256)
        outputs = model(images)

        # (1, ) 
        loss = loss_fn(outputs, target)

        if loss_reduction == 'mean':
            # Scale the loss by the batch size over dataset size
            loss = loss * batch_size / total_samples
        elif loss_reduction == 'sum':
            # Scale the loss by 1 over dataset size
            loss = loss / total_samples
        else:
            raise ValueError("Unsupported loss reduction method: {}".format(loss_reduction))

        # Compute scaled loss
        loss.backward()  # Accumulates gradients

    # Extract and clone gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_dict[name] = param.grad.clone().cpu().detach()

    return grad_dict

def main(args):    

    img_dim = args.img_dim
    batch_size = args.batch_size
    model_name = args.model_name
    dataset = args.dataset
    probing_vector = args.probing_vector
    grad_dict_dir = args.grad_dict_dir
    run_num = args.run_num
    num_classes = args.num_classes
    loss_fn = args.loss

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the custom GLAS dataset
    train_img_ids = sorted(glob(os.path.join(dataset, 'train', 'images', '*')))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(img_dim, img_dim),
        Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(dataset, 'train','images'),
        mask_dir=os.path.join(dataset, 'train','masks'),
        img_ext='.bmp',
        mask_ext='.bmp',
        num_classes=num_classes,
        transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # define loss function (criterion)
    if loss_fn == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[loss_fn]().to(device) # BCEDiceLoss


    model = create_model(
        num_classes=num_classes, 
        probing_vector=probing_vector,
        arch=model_name
    )
    
    model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    loss_reduction = 'mean'

    # Compute full gradient        
    full_grad_param_dict = compute_full_gradient(
        model = model,
        data_loader = train_loader,
        loss_fn = criterion,
        loss_reduction = loss_reduction,
        device = device
    )

    final_path = grad_dict_dir + "/" + "full_grad_param_dict.pkl"
    print("Saving full gradient to {}".format(final_path))            
    with open(final_path, "wb") as f:
        pickle.dump(full_grad_param_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)