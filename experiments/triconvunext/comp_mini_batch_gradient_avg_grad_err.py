import os
import json

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import torch.nn as nn

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
import argparse
from only_comp_avg_grad_err import (
    compute_avg_grad_err,
    get_conv_param_names
)

"""
    Compute Mini-batch Gradient for the given image dimension for UNeXt model.

    Usage:
        sh bash_scripts/bash_comp_mini_batch_gradient_avg_grad_err.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the mini-batch gradients and the average gradient error for the given image dimension for the UNeXt model.'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True, help='Probing-vector')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes.')
    parser.add_argument(
        '--loss',
        default='BCEDiceLoss',
        choices=LOSS_NAMES,
        help='loss: ' +
        ' | '.join(LOSS_NAMES) +
        ' (default: BCEDiceLoss)'
    )
    parser.add_argument('--dataset', type=str, required=True, help='Dataset')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
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
        help='Path to save the average gradient errors.'
    )
    args = parser.parse_args()

    return args
            

def compute_mini_batch_gradient_avg_grad_err(
    model, 
    data_loader, 
    loss_fn,
    loss_reduction: str,
    full_grad_param_dict, 
    device: torch.device
  ) -> float:
    """
    Computes and stores parameter gradients for each mini-batch in the dataset.

    For every batch in the DataLoader, this function performs a forward and backward pass,
    and stores a dictionary of parameter gradients for that batch.

    Args:
        model (torch.nn.Module): The model whose gradients are to be computed.
        data_loader (DataLoader): DataLoader yielding mini-batches of (images, labels).
        loss_reduction (str): Reduction method used in the loss function ('mean' or 'sum').
        loss_fn (torch.nn.Module): Loss function to use for computing the gradient.
        full_grad_param_dict (dict): Dictionary of full dataset gradients for comparison.
        device (torch.device or str): Device on which computations should be run.
    Returns:
        avg_grad_err (float): Average L2 error between mini-batch gradients and full dataset gradient
    """

    model.train() 
    model.zero_grad()  
    batch_errors = []

    conv_param_names = get_conv_param_names(model)
    
    for i, (images, target, _) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):

            model.zero_grad() # Zero the gradients before each batch

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
                pass
            elif loss_reduction == 'sum':
                # Scale the loss by 1 over dataset size
                loss = loss / batch_size
            else:
                raise ValueError("Unsupported loss reduction method: {}".format(loss_reduction))

            loss.backward()
            
            batch_grads = {
                name: param.grad.clone().cpu().detach()
                for name, param in model.named_parameters()
                if param.requires_grad and param.grad is not None
            }

            # For each mini-batch, compute the average gradient error across all conv layers
            batch_grad_err = compute_avg_grad_err(
                mini_batch_grads = {i : batch_grads},
                full_model_grads = full_grad_param_dict,
                model_layer = None,
                conv_param_names= conv_param_names
            )            

            print("batch_grad_err for {} iteration is {}".format(i, batch_grad_err))

            batch_errors.append(batch_grad_err)

    return np.mean(batch_errors)

def main(args):    

    batch_size = args.batch_size
    img_size = args.img_size
    probing_vector = args.probing_vector
    model_name = args.model_name
    full_grads_path = args.full_grads_path
    grad_dict_dir = args.grad_dict_dir
    dataset = args.dataset
    num_classes = args.num_classes
    loss_fn = args.loss

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    # Load the full gradient parameter dictionary
    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)

    print("Loading full gradients from {}".format(full_grads_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the custom GLAS dataset
    train_img_ids = sorted(glob(os.path.join(dataset, 'train', 'images', '*')))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]

    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(img_size, img_size),
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
    model = model.to(device)

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    loss_reduction = 'mean'

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")


    # Compute the average gradient errors.        
    avg_grad_err = compute_mini_batch_gradient_avg_grad_err(
        model = model,
        data_loader = train_loader,
        loss_fn = criterion,
        loss_reduction = loss_reduction,
        device = device,
        full_grad_param_dict = full_grad_param_dict
    )

    print(f"Avg L2 gradient error: {avg_grad_err:}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_grad_err, f, pickle.HIGHEST_PROTOCOL)
    
    print("Saved average gradient error to {}".format(avg_grad_err_file_path))

if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)