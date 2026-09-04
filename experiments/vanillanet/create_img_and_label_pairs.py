import argparse
import torch
import os
from tqdm import tqdm

"""
    This script creates (img, label) pairs for a dataset based on the image dimension.

    Usage:
         sh bash_scripts/bash_create_img_and_label_pairs.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Create (img, label) pairs for a dataset for a given image dimension.")
    parser.add_argument(
        "--img_dim", 
        type=int, 
        required=True,
        help="Image dimension (H, W)."
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=4096, 
        required = True, 
        help="Size of the subset to create (img, label) pairs for."
    )
    parser.add_argument(
        "--num_ch",
        type=int,
        default=3,
        help="Number of channels in the images."
    )
    parser.add_argument(
        "--img_folder", 
        type=str, 
        default="img_folder",
        required = True, 
        help="Path to the folder containing images."
    )    
    parser.add_argument(
        "--label_folder", 
        type=str, 
        default="label_folder",
        required = True, 
        help="Path to the folder containing labels."
    )
    return parser.parse_args()

def main(args):
    img_dim = args.img_dim
    subset_size = args.subset_size
    img_folder = args.img_folder
    label_folder = args.label_folder
    num_ch = args.num_ch
    
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
    print(f"Creating (img, label) pairs for image shape: (1, {num_ch}, {img_dim}, {img_dim})")
    
    for idx in tqdm(range(subset_size), desc="Creating (img, label) pairs"):
        img = torch.randn(1, num_ch, img_dim, img_dim)
        label = torch.randint(0, 10, (1,))

        torch.save(img, os.path.join(img_folder, f"img_{idx}.pt"))
        torch.save(label, os.path.join(label_folder, f"label_{idx}.pt"))


if __name__ == "__main__":
    args = parse_args()
    main(args)