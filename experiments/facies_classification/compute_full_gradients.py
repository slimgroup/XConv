import os

from torch.utils.data import Subset
from core.models import get_model
import core.loss
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
from core.augmentations import (
    Compose, RandomHorizontallyFlip, RandomRotate, AddNoise)
import argparse
from core.loader.data_loader import patch_loader


"""
    Compute Full Gradient for the given image dimension for Facies Classification Benchmark model.

    Usage:
        sh bash_scripts/bash_compute_full_gradients.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Full gradient error for the given image dimension for the Facies Classification Benchmark model.'
    )
    parser.add_argument('--aug', nargs='?', type=bool, default=False,
                        help='Whether to use data augmentation.')
    parser.add_argument('--stride', nargs='?', type=int, default=50,
                        help='The vertical and horizontal stride when we are sampling patches from the volume.' +
                             'The smaller the better, but the slower the training is.')
    parser.add_argument('--patch_size', nargs='?', type=int, default=99,
                        help='The size of each patch')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument(
        '--grad_dict_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where gradient dictionaries will be saved.'
    )
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
    parser.add_argument('--arch', nargs='?', type=str, default='patch_deconvnet',
                        help='Architecture to use [\'patch_deconvnet, path_deconvnet_skip, section_deconvnet, section_deconvnet_skip\']')
    parser.add_argument('--pretrained', nargs='?', type=bool, default=False,
                        help='Pretrained models not supported. Keep as False for now.')
    parser.add_argument(
        '--subset_indices_pickle_path', 
        type=str, 
        required=True, 
        help='Path to the pickle file containing the selected subset indices.'
    )
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
    )
    args = parser.parse_args()

    return args


def compute_full_gradient(
    model, 
    loss_fn,
    data_loader : torch.utils.data.DataLoader, 
    loss_reduction: str,
    device: torch.device,
    bf16_precision:bool = False
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
        bf16_precision (bool): If true, convert the image tensor to bf-16.

    Returns:
        dict: A dictionary mapping parameter names to their accumulated gradient tensors.
              Gradients are detached from the graph and cloned.
    """

    model.train()
    model.zero_grad() # Erase gradients before performing this computation.
    grad_dict = {}
    total_samples = len(data_loader.dataset)

    for images, labels in tqdm(data_loader, desc="Computing Full Gradient"):
        
        batch_size = images.size(0)
        
        # (B, C, H, W) eg. (128, 3, 28, 28)
        images = images.to(device)
        labels = labels.to(device)

        if bf16_precision:
            images = images.half()

        # Forward pass
        # (B, num_classes) eg. (128, 1000)
        outputs = model(images)

        # (B, num_classes) eg. (128, 1000)
        loss = loss_fn(input=outputs, target=labels, weight=None)        # loss = outputs.sum()

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

    batch_size = args.batch_size
    subset_indices_pickle_path = args.subset_indices_pickle_path
    grad_dict_dir = args.grad_dict_dir
    bf16_precision = args.bf16_precision

    if args.aug:
            data_aug = Compose(
                [RandomRotate(10), RandomHorizontallyFlip(), AddNoise()])
    else:
        data_aug = None

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    with open(subset_indices_pickle_path, 'rb') as f:
        subset_indices = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = patch_loader(
        is_transform=True,
        split='train',
        stride=args.stride, # 50
        patch_size=args.patch_size, # 99
        augmentations=data_aug
    )
    train_subset = Subset(train_set, subset_indices)
    
    # Main processing loop
    print("Processing batch-size: {}".format(batch_size))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size)

    # 6 classes    
    n_classes = train_set.n_classes

    model = get_model(
        args.arch, 
        args.pretrained, 
        n_classes
    )   

    loss_fn = core.loss.cross_entropy_mean

    # Match xconv_pv: mean CE with per-batch scaling for the full-dataset gradient.
    loss_reduction = "mean"
    
    model = model.to(device)

    if bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    print(model)

    # Compute full gradient        
    full_grad_param_dict = compute_full_gradient(
        model = model,
        data_loader = train_loader,
        loss_fn = loss_fn,
        loss_reduction = loss_reduction,
        device = device,
        bf16_precision = bf16_precision
    )

    final_path = grad_dict_dir + "/" + "full_grad_param_dict.pkl"
    print("Saving full gradient to {}".format(final_path))            
    with open(final_path, "wb") as f:
        pickle.dump(full_grad_param_dict, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    main(args)