import argparse
import torch
import os
from tqdm import tqdm

"""
    This script creates (img, z, t) pairs for a subset of the dataset based on the image dimension.

    Usage:
         sh bash_scripts/bash_create_img_z_t_dataset.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Create (img, z, t) pairs for a subset of the dataset.")
    parser.add_argument(
        "--img_dim", 
        type=int, 
        required=True,
        help="Image dimension (H, W)."
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=1024, 
        required = True, 
        help="Size of the subset to create (z, t) pairs for."
    )
    parser.add_argument(
        "--num_ch",
        type=int,
        default=3,
        help="Number of channels in the images."
    )
    parser.add_argument(
        "--t_folder", 
        type=str, 
        default="t_folder",
        required = True, 
        help="Path to the folder containing t values."
    )    
    parser.add_argument(
        "--z_folder", 
        type=str, 
        default="z_folder",
        required = True, 
        help="Path to the folder containing z values."
    )
    parser.add_argument(
        "--img_folder", 
        type=str, 
        default="img_folder",
        required = True, 
        help="Path to the folder containing images."
    )
    return parser.parse_args()

def main(args):
    img_dim = args.img_dim
    subset_size = args.subset_size
    t_folder = args.t_folder
    z_folder = args.z_folder
    num_ch = args.num_ch
    img_folder = args.img_folder

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    
    if not os.path.exists(t_folder):
        os.makedirs(t_folder)
    
    if not os.path.exists(z_folder):
        os.makedirs(z_folder)
    
    print(f"Creating (img, z, t) pairs for image shape: (1, {num_ch}, {img_dim}, {img_dim})")
    sde_t_val = 1.0  # Example t value
    eps_val = 1e-5  # Small epsilon value
    
    # Add your logic here to create (img, z, t) pairs based on the image dimension.
    for idx in tqdm(range(subset_size), desc="Creating (img, z, t) pairs"):
        img = torch.randn(1, num_ch, img_dim, img_dim)  # Example img value
        t = torch.randn(1)*(sde_t_val - eps_val) + eps_val  # Example t value
        z = torch.randn(1, num_ch, img_dim, img_dim)  # Example z value
        torch.save(img, os.path.join(img_folder, f"img_{idx}.pt"))
        torch.save(t, os.path.join(t_folder, f"t_{idx}.pt"))
        torch.save(z, os.path.join(z_folder, f"z_{idx}.pt"))


if __name__ == "__main__":
    args = parse_args()
    main(args)