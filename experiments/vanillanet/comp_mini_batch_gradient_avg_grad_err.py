import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Anti-fragmentation. PYTORCH_CUDA_ALLOC_CONF is the legacy name (vanillanet_env);
# torch>=2.9 (sips) deprecated it in favour of PYTORCH_ALLOC_CONF, so set BOTH
# before torch is imported. Methodology-neutral.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

from pyxconv.utils import convert_net, adaptive_convert_net


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
import models.vanillanet
from comp_full_gradient import StoredDataset
from timm.models import create_model




from only_comp_avg_grad_err import (
    compute_avg_grad_err,
    get_conv_param_names
)


"""
    Compute Mini-batch Gradient and Average Gradient Error for the given image dimension for VanillaNet model.

    Usage:
        sh bash_scripts/bash_comp_mini_batch_gradient_avg_grad_err.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Mini-batch Gradient and Average Gradient Error for the given image dimension for the VanillaNet model.'
    )
    parser.add_argument(
        '--num_ch', 
        type=int, 
        required=True, 
        help='Number of Channels'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True, help='Probing-vector')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--xconv_varn', type=str, required=True, help='XConv variant name')

    parser.add_argument(
        '--grad_dict_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where gradient dictionaries will be saved.'
    )
    parser.add_argument('--deploy', type=str2bool, default=False)
    parser.add_argument(
        '--drop', 
        type=float, 
        default=0, 
        metavar='PCT',
        help='Drop rate (default: 0.0)'
    )
    parser.add_argument('--act_num', default=3, type=int)

    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
    parser.add_argument(
        '--init_seed',
        type=int,
        default=0,
        help='Fixed seed for model initialization. Applied immediately before '
             'model creation (and adaptive_convert_net, which reuses the same '
             'weight tensors) so the init weights are byte-identical across runs '
             'and identical to the reference (comp_full_gradient.py); --seed '
             '(run_num) then drives only the post-init probing draws.'
    )
    parser.add_argument(
        '--subset_indices_path',
        type=str,
        default=None,
        help='Accepted for parity with the bash launcher (the StoredDataset '
             'reads pre-generated img/label pairs directly, so this is unused).'
    )
    parser.add_argument(
        '--full_grads_path',
        type=str, 
        required=True, 
        help='Path to the pickle file containing the full gradients.'
    )
    parser.add_argument(
        '--img_folder', 
        type=str, 
        required=True, 
        help='Path to the folder containing images.'
    )
    parser.add_argument(
        '--label_folder', 
        type=str, 
        required=True, 
        help='Path to the folder containing labels.'
    )
    parser.add_argument(
        '--subset_size', 
        type=int, 
        required=True, 
        help='Size of the subset.'
    )
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
    )

    args = parser.parse_args()

    return args

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def compute_mini_batch_gradient_avg_grad_err(
    model, 
    data_loader, 
    loss_fn,
    loss_reduction: str,
    full_grad_param_dict, 
    device: torch.device,
    bf16_precision:bool = False
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
        bf16_precision (bool): If true, convert the image tensor to bf-16.

    Returns:
        avg_grad_err (float): Average L2 error between mini-batch gradients and full dataset gradient
    """

    # Pure train mode, matching the reference (comp_full_gradient.py) and the
    # paper's VanillaNet AGE experiment. NOTE: do NOT force BatchNorm/Dropout to
    # eval here -- on the untrained VanillaNet, eval-ing its 28 BatchNorm layers
    # collapses the forward to ~1e-13 (vanishing gradients -> degenerate AGE and
    # an artifactual "XConv < Conv"). The shared fixed --init_seed gives both
    # processes the same init theta, so AGE still isolates the estimator error;
    # the per-run XConv probing draws (torch.randint in Xconv2D.forward from the
    # run-seeded global RNG) remain the intended +/- sigma source.
    model.train()
    model.zero_grad()
    batch_errors = []

    conv_param_names = get_conv_param_names(model)
    
    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):

            model.zero_grad() # Zero the gradients before each batch

            batch_size = images.size(0)

            # Skip size-<2 batches BEFORE the forward pass. BatchNorm in train
            # mode cannot compute batch statistics from a single sample and
            # raises "Expected more than 1 value per channel" once a 1x1-spatial
            # BN layer sees batch=1 (happens for a trailing minibatch when
            # subset_size % batch_size == 1, e.g. img=512/batch=9). Dropping one
            # size-1 sample out of thousands is negligible for the AGE (a mean
            # over many minibatches) and keeps the count correct because such a
            # batch never gets appended to batch_errors below.
            if batch_size < 2:
                continue

            # (B, C, H, W) eg. (128, 3, 28, 28)
            images = images.to(device)
            labels = labels.to(device)

            if bf16_precision:
                images = images.half()

            # Forward pass
            # (B, num_classes) eg. (128, 1000)
            outputs = model(images)

            # (B, num_classes) eg. (128, 1000)
            loss = loss_fn(outputs, labels)
            # loss = outputs.sum()

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

    img_size = args.img_size
    batch_size = args.batch_size
    probing_vector = args.probing_vector
    model_name = args.model_name
    xconv_varn = args.xconv_varn
    full_grads_path = args.full_grads_path
    grad_dict_dir = args.grad_dict_dir
    num_ch = args.num_ch
    img_folder = args.img_folder
    label_folder = args.label_folder
    subset_size = args.subset_size
    bf16_precision = args.bf16_precision

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    # Load the full gradient parameter dictionary
    with open(full_grads_path, "rb") as f:
        full_grad_param_dict = pickle.load(f)

    print("Loading full gradients from {}".format(full_grads_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stored_dataset = StoredDataset(
        img_folder = img_folder,
        label_folder = label_folder,
        subset_size = subset_size,
        device = device
    )   

    # Main processing loop
    print("Processing batch-size: {}".format(batch_size))
    train_loader = torch.utils.data.DataLoader(
        stored_dataset, 
        batch_size=batch_size
    )
    
    # for probing_vector in probing_vectors:
    print("Processing probing_vector: {}".format(probing_vector))
    base = (probing_vector == 'base')

    print("Model: {}".format(model_name))

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    # Fix the model INIT so the weights are byte-identical across runs and
    # identical to the reference (comp_full_gradient.py). This makes AGE isolate
    # the estimator (sampling + probing) error at a single representative theta,
    # rather than being dominated by random per-run init gradient-scale variance.
    # The scope also covers adaptive_convert_net below, whose Pass-1 dry forward
    # (run in eval(), so BN uses fixed running stats) is then deterministic.
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed(args.init_seed)

    # Initialize the model and convert it based on the probing vector
    if "squeezenet1_0" in model_name:
        model = models.squeezenet1_0(pretrained=False) # 408 MiB [includes 200 MiB allocated when CuDA is initialized]
    elif "vanillanet" in model_name:
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=args.nb_classes,
            act_num=args.act_num,
            drop_rate=args.drop,
            deploy=args.deploy,
        )

    model = model.to(device)

    if bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    if not base:
        if "adaptive" in xconv_varn:
            # Adaptive XConv = the paper's "Adaptive XConv" for VanillaNet: a
            # two-pass conversion (dry forward records each conv's input H*W, then
            # replaces only convs where H*W > ps and channels < maxc), reusing the
            # SAME weight tensors -> same fixed init as the base reference. NOT
            # convert_net(mode='all'). Keep it adaptive.
            if bf16_precision:
                dummy = torch.randn(1, num_ch, img_size, img_size, dtype=torch.float16).to(device)
            else:
                dummy = torch.randn(1, num_ch, img_size, img_size).to(device)
            model = adaptive_convert_net(
                model,
                sample_input=dummy,
                ps=probing_vector,                 # threshold
                xmode='independent',   # or 'gaussian', etc.
                mode='all',
                maxc=32001
            )
        else:
            convert_net(
                model,
                ps = probing_vector,
                xmode ='independent'
            )

    # Re-seed AFTER model creation / conversion with the per-run seed (run_num)
    # so the per-layer XConv probe seeds (drawn via torch.randint in
    # Xconv2D.forward from the global RNG) vary per run -- the intended +/- sigma
    # source at fixed theta. The DataLoader here is not shuffled, so the data
    # order is identical across runs.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(model)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_reduction = loss_fn.reduction


    # Compute the average gradient errors.        
    avg_grad_err = compute_mini_batch_gradient_avg_grad_err(
        model = model,
        data_loader = train_loader,
        loss_reduction = loss_reduction,
        loss_fn = loss_fn,
        full_grad_param_dict = full_grad_param_dict,
        device = device,
        bf16_precision = bf16_precision
    )

    print(f"Avg L2 gradient error: {avg_grad_err:}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_grad_err, f, pickle.HIGHEST_PROTOCOL)
    
    print("Saved average gradient error to {}".format(avg_grad_err_file_path))

if __name__ == "__main__":

    args = parse_args()

    # NOTE: seeding is done inside main() in a specific order -- the fixed
    # --init_seed immediately before model creation / adaptive_convert_net, then
    # --seed (run_num) re-applied after. Do NOT seed here, or the init seed would
    # be overridden out of order.
    main(args)