"""Matplotlib helpers for sample grids and loss curves.

Samples are expected to be already-denormalized tensors of shape
``[N, 1, H, W]`` on CPU. No per-sample renormalization happens here -- the
caller is responsible for inverting any preprocessing before plotting.
"""

import os
from typing import Optional, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Color range for velocity model patches (matches the Parihaka dataset
# distribution; values outside this range are clipped at plot time).
VMIN_DEFAULT = -1.5e3
VMAX_DEFAULT = 1.5e3


def plot_image(
    image: torch.Tensor,
    path: str,
    *,
    vmin: float = VMIN_DEFAULT,
    vmax: float = VMAX_DEFAULT,
    cmap: str = "Greys",
    interpolation: str = "lanczos",
) -> None:
    """Save a single single-channel image to ``path``.

    Args:
        image: Tensor of shape ``[1, H, W]`` or ``[H, W]``, already denormalized.
        path: Output PNG path; parent directory is created if missing.
        vmin, vmax: Color scale (same defaults as :func:`plot_grid`).
        cmap, interpolation: Matplotlib imshow settings.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = image[0].cpu().numpy() if image.ndim == 3 else image.cpu().numpy()
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect=1,
        interpolation=interpolation,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_grid(
    images: torch.Tensor,
    path: str,
    *,
    nrow: int = 4,
    title: Optional[str] = None,
    vmin: float = VMIN_DEFAULT,
    vmax: float = VMAX_DEFAULT,
    cmap: str = "Greys",
    interpolation: str = "lanczos",
) -> None:
    """Save a grid of single-channel images to ``path``.

    Args:
        images: Tensor of shape ``[N, 1, H, W]``, already denormalized.
        path: Output PNG path; parent directory is created if missing.
        nrow: Number of columns in the grid.
        title: Optional figure title.
        vmin, vmax: Shared color scale across all panels.
        cmap, interpolation: Matplotlib imshow settings.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = images.shape[0]
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(
        nrows,
        ncol,
        figsize=(2.5 * ncol, 2.5 * nrows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    if nrows == 1:
        axes = np.atleast_2d(axes)
    for i in range(nrows):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                ax.imshow(
                    images[idx, 0].cpu().numpy(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect=1,
                    interpolation=interpolation,
                )
            ax.set_xticks([])
            ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=14)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_losses(
    train_obj: Sequence[float],
    val_obj: Sequence[float],
    val_every: int,
    path: str,
    *,
    title: str = "Training loss",
    ylabel: str = "Loss",
) -> None:
    """Plot training and validation loss curves vs. epoch.

    The training history is assumed to have one entry per minibatch and is
    spread linearly across epochs. The validation history is assumed to have
    one entry every ``val_every`` epochs starting at epoch 0.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(7, 4))
    if train_obj:
        n_train = len(train_obj)
        last_epoch = max(1, (len(val_obj) - 1) * val_every) if val_obj else n_train
        x_train = np.linspace(0, last_epoch, n_train)
        plt.plot(x_train, train_obj, color="orange", alpha=1.0, label="training")
    if val_obj:
        x_val = np.arange(len(val_obj)) * val_every
        plt.plot(x_val, val_obj, color="k", alpha=0.8, label="validation")
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
