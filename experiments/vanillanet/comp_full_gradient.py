import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Anti-fragmentation. PYTORCH_CUDA_ALLOC_CONF is the legacy name (vanillanet_env);
# torch>=2.9 (sips) deprecated it in favour of PYTORCH_ALLOC_CONF, so set BOTH
# before torch is imported. Methodology-neutral.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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
import models.vanillanet
from timm.models import create_model


"""
    Compute Full Gradient for the given image dimension for VanillaNet model.

    Usage:
        sh bash_scripts/bash_comp_full_gradient.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute Full gradient error for the given image dimension for the VanillaNet model.'
    )
    parser.add_argument(
        '--num_ch', 
        type=int, 
        required=True, 
        help='Number of Channels'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
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
             'model creation so the init weights are byte-identical across runs '
             'and across the reference (this script) and the method '
             '(comp_mini_batch_gradient_avg_grad_err.py); --seed (run_num) then '
             'drives only the post-init sampling randomness.'
    )
    parser.add_argument(
        '--subset_indices_path',
        type=str,
        default=None,
        help='Accepted for parity with the bash launcher (the StoredDataset '
             'reads pre-generated img/label pairs directly, so this is unused).'
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

    # Pure train mode, as in the paper's VanillaNet AGE experiment. NOTE: do NOT
    # force BatchNorm/Dropout to eval here. VanillaNet is untrained at this point,
    # and putting its 28 BatchNorm layers into eval (fixed running stats, which
    # are still at init) collapses the forward to ~1e-13 -> vanishing gradients,
    # a degenerate AGE, and an artifactual "XConv < Conv" ordering. The fixed
    # --init_seed already makes the init theta shared by this reference and the
    # method, so AGE still isolates the estimator error at a single theta.
    model.train()
    model.zero_grad() # Erase gradients before performing this computation.
    grad_dict = {}
    total_samples = len(data_loader.dataset)

    for images, labels in tqdm(data_loader, desc="Computing Full Gradient"):

        batch_size = images.size(0)

        # Skip size-<2 batches BEFORE the forward pass. BatchNorm in train mode
        # raises "Expected more than 1 value per channel" once a 1x1-spatial BN
        # layer sees batch=1 (a trailing minibatch when subset_size % batch_size
        # == 1, e.g. img=512/batch=9). Because the loss is scaled by
        # batch_size/total_samples, omitting one size-1 sample from the
        # accumulated full-dataset gradient is negligible for the reference.
        if batch_size < 2:
            continue

        # (B, C, H, W) eg. (128, 3, 28, 28)
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        # (B, num_classes) eg. (128, 1000)
        if bf16_precision:
            images = images.half()

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

    batch_size = args.batch_size
    model_name = args.model_name
    grad_dict_dir = args.grad_dict_dir
    img_folder = args.img_folder
    label_folder = args.label_folder
    subset_size = args.subset_size
    bf16_precision = args.bf16_precision

    if not os.path.exists(grad_dict_dir):
        os.makedirs(grad_dict_dir)

    # # Resize image to test the memory consumption of the method.
    # transform=transforms.Compose([
    #         transforms.Grayscale(num_output_channels=num_ch),
    #         transforms.Resize((img_size, img_size)),
    #         transforms.ToTensor(),
    #         #transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    # # Create a subset of "subset_size" 
    # train_dataset = datasets.MNIST(
    #     '../data', 
    #     train=True, 
    #     download=True,
    #     transform=transform
    # )
    
    # with open(subset_json_path) as f:
    #     subset_indices = json.load(f)
    
    # print("Reading from {}".format(subset_json_path))
    # subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

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
    # (comp_mini_batch_gradient_avg_grad_err.py). This makes AGE isolate the
    # estimator (sampling + probing) error at a single representative theta,
    # rather than being dominated by random per-run init gradient-scale variance.
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

    # Re-seed AFTER model creation with the per-run seed (run_num) so the only
    # post-init randomness (here just the deterministic DataLoader order; the
    # reference has no stochastic layers once BN/Dropout are eval-ed in
    # compute_full_gradient) still varies per run while the init stays fixed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    if bf16_precision:
        model = model.half()

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

    # NOTE: seeding is done inside main() in a specific order -- the fixed
    # --init_seed immediately before model creation, then --seed (run_num) after
    # creation. Do NOT seed here, or the init seed would be overridden out of
    # order.
    main(args)