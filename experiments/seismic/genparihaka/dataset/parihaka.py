"""Parihaka seismic dataset loader.

Loads ``data/seismic/training-pairs.h5`` (5282 velocity model patches,
single channel, 256x256) and produces train / validation tensors plus the
:class:`Normalizer` used to z-score them. The HDF5 file is auto-downloaded
from Dropbox on first call if it is not already on disk.
"""

import os
import subprocess
from typing import Tuple

import h5py
import torch
import torch.nn.functional as F
from projorg import datadir

from ..utils.normalizer import Normalizer

DATASET_FILENAME = "training-pairs.h5"
DATASET_KEY = "dm"
DATASET_SIZE = 5282
NATIVE_RESOLUTION = 256
DATASET_URL = (
    "https://www.dropbox.com/scl/fi/0dmnhlxk4jso10gr3oua9/"
    "training-pairs.h5?rlkey=2sdyqgs79jqoc7vjh1qrcnwx7&dl=1"
)


def _download_dataset(path: str) -> None:
    """Download the HDF5 file from Dropbox to ``path`` via wget."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Parihaka dataset not found at {path}.")
    print(f"Downloading from Dropbox ({DATASET_URL})...")
    result = subprocess.run(
        ["wget", "--no-check-certificate", "-O", path, DATASET_URL],
        check=False,
    )
    if result.returncode != 0 or not os.path.isfile(path):
        if os.path.isfile(path):
            os.remove(path)
        raise RuntimeError(
            f"Failed to download the Parihaka dataset to {path}. "
            f"Download it manually from {DATASET_URL} and place it there."
        )


def load_parihaka(
    image_size: int = 128,
    num_train: int = 5000,
    num_val: int = 282,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Normalizer]:
    """Load and preprocess the Parihaka seismic patches.

    Pipeline: load HDF5 -> permute spatial axes -> optional bilinear
    downsample -> per-pixel z-score with :class:`Normalizer`.

    Args:
        image_size: Target spatial resolution. Must be ``<= 256``; if smaller
            than the native 256, samples are bilinearly downsampled.
        num_train: Number of training samples.
        num_val: Number of validation samples drawn from the remainder.
        seed: Seed for the train/val permutation.

    Returns:
        ``(x_train, x_val, normalizer)``. Tensors have shape
        ``[num_train, 1, image_size, image_size]`` and
        ``[num_val, 1, image_size, image_size]`` respectively, in z-score
        space. The normalizer holds the per-pixel mean and std needed to
        invert the normalization for plotting.
    """
    if num_train + num_val > DATASET_SIZE:
        raise ValueError(
            f"num_train + num_val = {num_train + num_val} exceeds dataset size "
            f"{DATASET_SIZE}."
        )
    if image_size > NATIVE_RESOLUTION:
        raise ValueError(
            f"image_size={image_size} exceeds native {NATIVE_RESOLUTION}."
        )

    path = os.path.join(datadir("seismic"), DATASET_FILENAME)
    if not os.path.isfile(path):
        _download_dataset(path)

    with h5py.File(path, "r") as f:
        x = torch.from_numpy(f[DATASET_KEY][:])  # (5282, 1, 256, 256)
    x = x.permute(0, 1, 3, 2).contiguous().float()

    if image_size < NATIVE_RESOLUTION:
        x = F.interpolate(
            x, size=image_size, mode="bilinear", align_corners=False,
        )

    normalizer = Normalizer(x)
    x = normalizer.normalize(x)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    x_train = x[perm[:num_train]]
    x_val = x[perm[num_train : num_train + num_val]]
    return x_train, x_val, normalizer
