import os
import argparse
import cv2
import numpy as np
from PIL import Image

"""
    This script saves a mask as a black background with white foreground.

    Usage:
        sh bash_scripts/bash_save_mask_black_white.sh

"""

def parse_args():
    parser = argparse.ArgumentParser("Save mask as black background with white foreground")
    parser.add_argument(
        "--mask_path", type=str, required=True,
        help="Path to the mask file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to the output directory"
    )
    return parser.parse_args()

def save_mask_bw(
    mask, 
    out_path, 
    invert=False,
    new_size=(256, 256)
):
    """
    mask: HxW or 1xHxW, values in {0,1} or [0,1] or {0,255}
    Saves a black background with white foreground (like typical segmentation figs).
    """

    # (H, W, 1) eg. (522, 775, 1)
    m = np.asarray(mask)

    # (H, W, 1) -> (H, W) eg. (522, 775)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]

    # Convert to binary mask (only 0/1 values).
    binary_mask = (m > 0).astype(np.uint8)   # 0/1

    if invert:
        binary_mask = 1 - binary_mask

    # Convert to 0/255 grayscale image so that it can be saved as PNG.
    img = (binary_mask * 255).astype(np.uint8)  # 0/255

    # Convert to PIL Image
    pil_img = Image.fromarray(img, mode="L")
    
    resized_image = pil_img.resize(new_size, Image.NEAREST)

    resized_image.save(out_path, format="PNG")

    print(f"Saved mask to: {out_path}")

    return

def main(args):
    mask_path = args.mask_path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving mask as black background with white foreground: {mask_path}")

    # (H, W, 1) eg. (522, 775, 1)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]

    save_mask_bw(
        mask=mask, 
        out_path=os.path.join(
            output_dir, 
            os.path.basename(mask_path).replace(".bmp", "_bw.png")
        ), 
        invert=False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
