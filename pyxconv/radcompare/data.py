"""Data for the comparison.

AGE needs real CIFAR-10 minibatches (the gradient must be meaningful); peak
memory follows ``scripts/compute_peak_memory.py`` and uses random inputs sized
to the target batch/resolution. The CIFAR subset uses a fixed permutation and
``shuffle=False`` so every method sees the identical minibatch partition --
the comparison must isolate the gradient estimator, not the data order.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from projorg import datadir

__all__ = [
    "cifar10_subset_loader",
    "random_dataset_loader",
    "random_regression_loader",
    "LazyRegressionDataset",
    "random_inputs",
    "CIFAR_MEAN",
    "CIFAR_STD",
]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def cifar10_subset_loader(
    subset_size: int,
    batch_size: int,
    seed: int = 0,
    train: bool = True,
) -> DataLoader:
    """Fixed CIFAR-10 subset, fixed (unshuffled) minibatch partition."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    dataset = datasets.CIFAR10(
        datadir("cifar10"), train=train, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=False)


def random_dataset_loader(
    subset_size: int,
    batch_size: int,
    image_dim: int,
    seed: int = 0,
    channels: int = 3,
    num_classes: int = 10,
) -> DataLoader:
    """A fixed random (images, labels) dataset for the image-dimension AGE sweep.

    Matches the repo's AGE protocol, which stores random (image, label) pairs per
    image dimension. Deterministic given ``seed`` and fixed (unshuffled), so every
    method sees the identical minibatch partition; data stays on CPU and the AGE
    code moves each batch to the device.
    """
    generator = torch.Generator().manual_seed(seed)
    images = torch.randn(
        subset_size, channels, image_dim, image_dim, generator=generator
    )
    labels = torch.randint(0, num_classes, (subset_size,), generator=generator)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def random_regression_loader(
    subset_size: int,
    batch_size: int,
    image_dim: int,
    channels: int = 3,
    out_channels: int = 3,
    seed: int = 0,
) -> DataLoader:
    """Fixed random (input_image, target_image) pairs for the U-Net AGE sweep
    (MSE regression). Deterministic, unshuffled (same partition for all methods)."""
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        subset_size, channels, image_dim, image_dim, generator=generator
    )
    targets = torch.randn(
        subset_size, out_channels, image_dim, image_dim, generator=generator
    )
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


class LazyRegressionDataset(Dataset):
    """On-demand random (input_image, target_image) pairs for the U-Net AGE sweep.

    Sample ``i`` is generated lazily in ``__getitem__`` from a per-index seed
    ``dataset_seed + i``, so it is a PURE FUNCTION of ``(i, dataset_seed)`` and
    is byte-identical no matter the access order (sequential reference pass vs.
    shuffled minibatch pass). The full ``(subset_size, C, H, W)`` tensor is never
    materialized -- only the few samples in the live batches occupy host RAM --
    which is what keeps host memory bounded at large image dimensions where an
    eager dataset would need tens of GB. Mirrors ``create_img_and_label_pairs.py``
    (one deterministic random tensor per index) but in-memory/on-demand, with no
    disk writes. Tensors are float32; the fp16 path casts at the call site.
    """

    def __init__(
        self,
        subset_size: int,
        image_dim: int,
        channels: int = 3,
        out_channels: int = 3,
        dataset_seed: int = 0,
    ):
        self.subset_size = subset_size
        self.image_dim = image_dim
        self.channels = channels
        self.out_channels = out_channels
        self.dataset_seed = dataset_seed

    def __len__(self) -> int:
        return self.subset_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(self.dataset_seed + int(i))
        img = torch.randn(self.channels, self.image_dim, self.image_dim, generator=g)
        target = torch.randn(
            self.out_channels, self.image_dim, self.image_dim, generator=g
        )
        return img, target


def random_inputs(
    batch_size: int,
    image_dim: int = 32,
    channels: int = 3,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (images, labels) for peak-memory measurement, as in the paper's
    peak-memory scripts (memory depends on tensor shapes, not data)."""
    images = torch.randn(batch_size, channels, image_dim, image_dim, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    return images, labels
