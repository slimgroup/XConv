import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# PYTORCH_CUDA_ALLOC_CONF is deprecated on torch>=2.9; set the new name too so
# expandable_segments (anti-fragmentation) actually takes effect. Methodology-neutral.
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from torchvision import models
from comp_full_gradient import StoredDataset
from avg_grad_err import (
    compute_avg_grad_err,
    get_conv_param_names
)
from pyxconv.utils import convert_net


"""
    Compute Mini-batch Gradient for the given image dimension for SqueezeNet model.

    Usage:
        sh scripts/drivers/bash_comp_mini_batch_gradient_avg_grad_err.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the mini-batch gradients and the average gradient error for the given image dimension for the SqueezeNet model.'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True, help='Probing-vector')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--num_ch', type=int, required=True, help='Number of input channels.')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
    parser.add_argument(
        '--init_seed',
        type=int,
        default=0,
        help='Fixed seed for model initialization. Applied immediately before '
             'model creation (and convert_net, which reuses the same weight '
             'tensors) so the init weights are byte-identical across runs and '
             'identical to the reference (comp_full_gradient.py); --seed '
             '(run_num) then drives only the post-init sampling/probing draws.'
    )
    parser.add_argument(
        '--subset_indices_path', 
        type=str, 
        required=True, 
        help='Path to the json file containing the selected subset indices.'
    )
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
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
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
    args = parser.parse_args()

    return args
            

def compute_mini_batch_gradient_avg_grad_err(
    model, 
    data_loader, 
    loss_fn,
    loss_reduction: str,
    full_grad_param_dict, 
    device: torch.device,
    bf16_precision: bool = False
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
    # Match the reference (comp_full_gradient.py): neutralize stochastic Dropout
    # so the only post-init per-run randomness left is the XConv probing draws
    # (drawn via torch.randint in Xconv2D.forward from the run-seeded global RNG).
    # This makes AGE isolate the estimator error against the SAME fixed-theta
    # path the reference computes. SqueezeNet has no BatchNorm, so eval-ing only
    # Dropout leaves every other layer in train mode.
    for _m in model.modules():
        if isinstance(_m, nn.modules.dropout._DropoutNd):
            _m.eval()
    model.zero_grad()
    batch_errors = []

    conv_param_names = get_conv_param_names(model)
    
    for i, (images, labels) in enumerate(tqdm(data_loader, desc="Computing Mini-Batch Gradients")):

            model.zero_grad() # Zero the gradients before each batch

            batch_size = images.size(0)

            # (B, C, H, W) eg. (128, 3, 28, 28)
            images = images.to(device)

            # (B, num_classes) eg. (128, 1000)
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
    subset_json_path = args.subset_indices_path
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
        batch_size = batch_size
    )

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    loss_reduction = loss_fn.reduction
    
    # for probing_vector in probing_vectors:
    print("Processing probing_vector: {}".format(probing_vector))

    print("Model: {}".format(model_name))

    if probing_vector.isdigit():
        probing_vector = int(probing_vector)

    base = (probing_vector == 'base')

    # Fix the model INIT so the weights are byte-identical across runs and
    # identical to the reference (comp_full_gradient.py). This makes AGE isolate
    # the estimator (sampling + probing) error at a single representative theta,
    # rather than being dominated by random per-run init gradient-scale variance.
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed(args.init_seed)

    # Initialize the model and convert it based on the probing vector
    if "squeezenet1_0" in model_name:
        model = models.squeezenet1_0(pretrained=False) # 408 MiB [includes 200 MiB allocated when CuDA is initialized]

    if bf16_precision:
        model = model.half()


    model = model.to(device)

    if args.bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    # convert_net swaps Conv2d -> Xconv2D in place, reusing the SAME weight/bias
    # tensors (newconv.weight = child.weight), so the probed method starts from
    # the SAME fixed init as the base reference. Done inside the init-seed scope
    # so any RNG the new modules touch is reproducible too.
    if not base: convert_net(
        model,
        ps = probing_vector,
        xmode ='independent'
    )

    # Re-seed AFTER model creation / conversion with the per-run seed (run_num)
    # so the per-layer XConv probe seeds (drawn via torch.randint in
    # Xconv2D.forward from the global RNG) -- and the DataLoader shuffle if any
    # -- vary per run, which is the intended +/- sigma source at fixed theta.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(model)

    # Compute the average gradient errors.        
    avg_grad_err = compute_mini_batch_gradient_avg_grad_err(
        model = model,
        data_loader = train_loader,
        loss_reduction = loss_reduction,
        loss_fn = loss_fn,
        device = device,
        full_grad_param_dict = full_grad_param_dict,
        bf16_precision = bf16_precision
    )

    print(f"Avg L2 gradient error: {avg_grad_err:}")

    avg_grad_err_file_path = os.path.join(grad_dict_dir, "avg_grad_err.pkl")
    with open(avg_grad_err_file_path, "wb") as f:
        pickle.dump(avg_grad_err, f, pickle.HIGHEST_PROTOCOL)
    
    print("Saved average gradient error to {}".format(avg_grad_err_file_path))

if __name__ == "__main__":

    args = parse_args()

    # NOTE: seeding is done inside main() in a specific order: the fixed
    # --init_seed is applied immediately before model creation / convert_net,
    # then --seed (run_num) is re-applied after. Do NOT seed here, or the init
    # seed below would be overridden out of order.
    main(args)