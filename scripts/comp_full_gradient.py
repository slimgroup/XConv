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

import argparse
from torchvision import models



"""
    Compute Full Gradient for the given image dimension for SqueezeNet model.

    Usage:
        sh scripts/drivers/bash_comp_full_gradient.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Full gradient error for the given image dimension for the SqueezeNet model.'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--probing_vector', type=str, required=True, help='Probing-vector')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument(
        '--grad_dict_dir', 
        type=str, 
        required=True, 
        help='Path to the directory where gradient dictionaries will be saved.'
    )
    parser.add_argument('--num_ch', type=int, required=True, help='Number of input channels.')
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--seed', type=int, required=True, help='Seed to control randomness.')
    parser.add_argument(
        '--init_seed',
        type=int,
        default=0,
        help='Fixed seed for model initialization. Applied immediately before '
             'model creation so the init weights are byte-identical across runs '
             'and across the reference/method scripts; --seed (run_num) then '
             'drives only the post-init sampling/probing randomness.'
    )
    parser.add_argument(
        '--subset_indices_path', 
        type=str, 
        required=True, 
        help='Path to the json file containing the selected subset indices.'
    )
    parser.add_argument(
        '--img_folder', 
        type=str, 
        required=True, 
        help='Path to the folder containing images.'
    )
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
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

class StoredDataset(torch.utils.data.Dataset):
    """
        A dataset that returns the stored Images and labels.
    """
    def __init__(
          self, 
          img_folder, 
          label_folder, 
          subset_size,
          device
    ):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.device = device
        self.subset_size = subset_size
        self.num_files = subset_size

        assert self.num_files == self.subset_size

        assert self.num_files == len(os.listdir(self.img_folder))
        assert self.num_files == len(os.listdir(self.label_folder))

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, f"img_{idx}.pt")
        label_path = os.path.join(self.label_folder, f"label_{idx}.pt")

        img = torch.load(img_path, map_location="cpu")
        label = torch.load(label_path, map_location="cpu")

        return img.squeeze(0), label.squeeze(0)


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
    # Neutralize stochastic Dropout so the REFERENCE full gradient is a single,
    # fixed-theta gradient (identical across runs), not a per-run-noisy one.
    # SqueezeNet's only post-init randomness in .train() is classifier Dropout
    # (it has no BatchNorm), and that mask is driven by the per-run --seed; with
    # the squared Eq. 9 metric it would otherwise leak into AGE as non-estimator
    # noise. Eval-ing ONLY Dropout leaves every other layer in train mode.
    for _m in model.modules():
        if isinstance(_m, nn.modules.dropout._DropoutNd):
            _m.eval()
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
        loss = loss_fn(outputs, labels)
        # loss = outputs.sum()

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

    img_size = args.img_size
    batch_size = args.batch_size
    model_name = args.model_name
    subset_json_path = args.subset_indices_path
    grad_dict_dir = args.grad_dict_dir
    num_ch = args.num_ch
    img_folder = args.img_folder
    label_folder = args.label_folder
    subset_size = args.subset_size
    bf16_precision = args.bf16_precision

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

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

    # Fix the model INIT so the weights are byte-identical across runs and
    # across the reference (this script) and the method
    # (comp_mini_batch_gradient.py). This makes AGE isolate the
    # estimator (sampling + probing) error at a single representative theta,
    # rather than being dominated by random per-run init gradient-scale variance.
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.cuda.manual_seed(args.init_seed)

    if "squeezenet1_0" in model_name:
        model = models.squeezenet1_0(pretrained=False) # 408 MiB [includes 200 MiB allocated when CuDA is initialized]

    # Re-seed AFTER model creation with the per-run seed (run_num) so all
    # post-init randomness (the base reference is deterministic here; for the
    # probed method this is the DataLoader / XConv probing draws) still varies
    # per run while the init stays fixed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    # NOTE: seeding is done inside main() in a specific order: the fixed
    # --init_seed is applied immediately before model creation, then --seed
    # (run_num) is re-applied after creation. Do NOT seed here, or the init
    # seed below would be overridden out of order.
    main(args)