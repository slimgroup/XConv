import os
import argparse
import torchvision.transforms.functional as F
from torchmetrics import PeakSignalNoiseRatio
from PIL import Image 

"""
    Compute the Peak-Signal-to-Noise Ratio (PSNR) metric between a image generated via DIP and the 
    original image.

    Usage:
        sh bash_scripts/bash_compute_psnr_dip_outputs.sh
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute the PSNR of a generated vs an original image.'
    )
    parser.add_argument(
        '--orig_img_path',
        type=str,
        required=True,
        help='Path to the original image.',
    )
    parser.add_argument(
        '--gen_img_path',
        type=str,
        help='Path to the image generated from the DIP.',
        required=True
    )
    args = parser.parse_args()
    return args

def main(args):
    orig_img_path = args.orig_img_path
    gen_img_path = args.gen_img_path

    assert os.path.exists(orig_img_path) and os.path.exists(gen_img_path)
    print("Reading the original image from {}".format(orig_img_path))
    print("Reading the generated image from {}".format(gen_img_path))

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()
    orig_img = Image.open(orig_img_path).convert("RGB")
    gen_img = Image.open(gen_img_path).convert("RGB")

    # Convert image to tensor to convert range to [0, 1]
    orig_img_tensor = F.to_tensor(orig_img)
    gen_img_tensor = F.to_tensor(gen_img)

    # 3. Add a batch dimension (TorchMetrics expects [B, C, H, W])
    orig_img_tensor = orig_img_tensor.unsqueeze(0)
    gen_img_tensor = gen_img_tensor.unsqueeze(0)

    # 4. Compute PSNR (Using 1.0 because torchvision.transforms scales to 0-1)
    psnr_score = psnr_fn(preds = gen_img_tensor, target = orig_img_tensor)
    print(f"PSNR: {psnr_score.item():.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)