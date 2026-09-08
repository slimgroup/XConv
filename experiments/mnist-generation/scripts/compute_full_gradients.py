import os
import re
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import pickle
import numpy as np
import torch.utils
import torch.utils.data
from tqdm import tqdm
import torch

import argparse
from torchvision import models

from diffusers import DDPMScheduler, UNet2DModel


"""
    This script computes the full gradients of the SIPS[UNet] model for the given image dimension.

Usage:
    conda activate sips
    sh bash_scripts/bash_compute_full_gradients.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the full gradients of SIPS[UNet] for the given image dimension.'
    )
    parser.add_argument('--img_size', type=int, required=True, help='Image dimension')
    parser.add_argument(
        '--subset_size', 
        type=int, 
        default=1024, 
        help='Subset size selected for this experiment.'
    )
    parser.add_argument(
        '--num_ch', 
        type=int, 
        required=True, 
        help='Number of Channels'
    )
    parser.add_argument('--batch_size', type=int, required=True, help='Batch-size')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument(
        '--base_grad_dict_dir', 
        type=str, 
        required=True,  
        help='Base gradient dictionary directory to save the full gradients.'
    )
    parser.add_argument('--act_num', default=3, type=int)
    parser.add_argument('--drop', type=float, default=0, metavar='PCT',
                        help='Drop rate (default: 0.0)')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument(
        '--seed', 
        type=int,
        default=12, 
        help='Seed'
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
    parser.add_argument(
         '--t_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "t" vectors for the corresponding image-dimension.'
    )    
    parser.add_argument(
         '--z_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "z" vectors for the corresponding image-dimension.'
    )    
    parser.add_argument(
        '--img_folder',
        type=str,
        required=True,
        help='Folder containing pre-computed "img" vectors for the corresponding image-dimension.'
    )
    parser.add_argument(
        '--bf16_precision', 
        action='store_true', 
        help = "Convert model to 16-bit precision."
    )
    args = parser.parse_args()

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


def comp_full_gradient_dict(
    score_model, 
    data_loader, 
    noise_scheduler,
    loss_reduction,
    device,
    bf16_precision:bool = False
):
    """
    Computes the full dataset gradient for the UNet[SIPs] by accumulating gradients over all mini-batches.

    This function performs a forward and backward pass for each mini-batch,
    scales each batch loss appropriately so the accumulated gradients reflect
    the true average gradient over the entire dataset.

    Args:
        score_model (torch.nn.Module): The model whose gradients are to be computed.
        data_loader (DataLoader): DataLoader for the full dataset.
        loss_reduction (str): The reduction type for the loss.
        noise_scheduler (DDPMScheduler): The noise scheduler used in the diffusion process.
        device (torch.device or str): Device to run computation on (e.g., 'cuda' or 'cpu').
        bf16_precision (bool): If true, convert the image tensor to bf-16.

    Returns:
        dict: A dictionary mapping parameter names to their accumulated gradient tensors.
              Gradients are detached from the graph and cloned.
    """

    score_model.train()
    score_model.zero_grad() # Erase gradients before performing this computation.
    grad_dict = {}

    total_samples = len(data_loader.dataset)
    
    for images, zs, ts in tqdm(data_loader, desc="Computing Full Gradient"):
        
            _ = images.size(0)

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

            if bf16_precision:
                noisy_images = noisy_images.half()

            # (B, C, H, W) eg. (5, 1, 1024, 1024)
            noise_pred = score_model(
                noisy_images,
                timesteps,
                return_dict=False,
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

    img_size = args.img_size
    subset_size = args.subset_size
    batch_size = args.batch_size
    model_name = args.model_name
    num_ch = args.num_ch
    base_grad_dict_dir = args.base_grad_dict_dir
    t_folder = args.t_folder
    z_folder = args.z_folder
    img_folder = args.img_folder
    bf16_precision = args.bf16_precision

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
    if model_name == "squeezenet1_0":
        model = models.squeezenet1_0(pretrained=False).to(device) # 408 MiB [includes 200 MiB allocated when CuDA is initialized]
    elif "vanillanet" in model_name:
        # Initialize the model and convert it based on the probing vector
        model = create_model(
                model_name, 
                pretrained=False,
                num_classes=args.nb_classes, 
                act_num=args.act_num,
                drop_rate=args.drop,
                deploy=False,
        )
    elif "sips_unet" in model_name:
         model = UNet2DModel(
            in_channels=num_ch,
            out_channels=1,
            sample_size=(img_size, img_size),
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

    if bf16_precision:
        model = model.half()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"#Params \n{trainable_params / 1e6:.1f}M")

    loss_reduction = "sum"
             
    #  Compute the full gradients and store the corresponding model weights in a dictionary.      
    full_grad_param_dict = comp_full_gradient_dict(
        score_model = model,
        data_loader = loader,
        noise_scheduler=noise_scheduler,
        loss_reduction = loss_reduction,
        device = device,
        bf16_precision = bf16_precision
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