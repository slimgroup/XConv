"""Per-pixel z-score normalizer with stored statistics."""

import torch


class Normalizer:
    """Normalize a tensor with training mean and standard deviation.

    Mean and std are computed across the batch dimension at construction
    time, so each spatial location has its own statistics.

    Attributes:
        mean: Mean over the dataset batch dimension. Shape matches a single
            example.
        std: Standard deviation over the dataset batch dimension.
        eps: Small float added to std to avoid division by zero.
    """

    def __init__(self, dataset: torch.Tensor, eps: float = 1e-5) -> None:
        self.mean = torch.mean(dataset, 0)
        self.std = torch.std(dataset, 0)
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return x * (std + self.eps) + mean
