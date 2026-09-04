import argparse
import os
import sys
from PIL import Image
import torchvision.transforms.functional as TF

"""
    Overlay mask on true image.

    Usage:
        sh bash_scripts/bash_overlay_mask_true_image.sh

"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Overlay mask on true image'
    )
    parser.add_argument(
        '--mask_path',
        type=str,
        required=True,
        help='Path to mask',
    )
    parser.add_argument(
        '--true_image_path',
        type=str,
        required=True,
        help='Path to true image',
    )
    parser.add_argument(
        '--save_name',
        type=str,
        required=True,
        help='Name to save the results',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save the results',
    )
    args = parser.parse_args()
    return args

def load_image(path):

    img = Image.open(path).convert('RGB')

    # (C, H, W) eg. (3, 512, 512)
    img = TF.to_tensor(img)

    return img

def load_mask(path):

    mask = Image.open(path).convert('L')

    # (C, H, W) eg. (1, 512, 512)
    mask = TF.to_tensor(mask)

    mask = (mask > 0.5).float()

    return mask

def main(args):
    mask_path = args.mask_path
    true_image_path = args.true_image_path
    save_dir = args.save_dir
    save_name = args.save_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading mask from: ", mask_path)
    mask = load_mask(mask_path)
    
    print("Loading true image from: ", true_image_path)
    true_image = load_image(true_image_path)

    # (C, H, W) eg. (3, 512, 512)
    corrupted_image = true_image*mask

    save_path = os.path.join(save_dir, save_name + "_corrupted_image.png")
    print("Saving corrupted image to: ", save_path)
    out = TF.to_pil_image(corrupted_image.clamp(0,1))
    out.save(save_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
