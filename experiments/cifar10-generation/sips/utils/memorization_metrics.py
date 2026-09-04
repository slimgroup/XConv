import torch


def euclidean_mem_metric(
    x0: torch.Tensor, samples: torch.Tensor, tol: list[float]
) -> torch.Tensor:
    pairwise_dist = torch.norm(x0[:, None, :] - samples[None, :, :], dim=2)

    num_memorized_samples = (pairwise_dist < tol).any(dim=0).sum().item()
    return num_memorized_samples / x0.shape[0]


def ratio_mem_metric(
    x_train: torch.Tensor,
    samples: torch.Tensor,
    ratio: float = 1 / 9,
    batch_size: int = 1000,
) -> float:
    """Nearest-neighbor ratio memorization metric.

    For each generated sample, computes d1/d2 where d1 is the distance to the
    closest training sample and d2 is the distance to the second closest. A
    sample is considered memorized when d1 <= ratio * d2.

    Args:
        x_train: Flattened training data (N_train, D).
        samples: Flattened generated samples (N_gen, D).
        ratio: Threshold ratio (default 1/9).
        batch_size: Process training data in chunks to avoid OOM.

    Returns:
        Fraction of generated samples considered memorized.
    """
    # Track top-2 closest distances per generated sample.
    n_gen = samples.shape[0]
    top2_dists = torch.full((n_gen, 2), float("inf"), device=samples.device)

    for i in range(0, x_train.shape[0], batch_size):
        batch = x_train[i : i + batch_size]
        dists = torch.cdist(samples, batch)  # (N_gen, batch_size)
        # Merge with running top-2.
        combined = torch.cat([top2_dists, dists], dim=1)
        top2_dists = combined.topk(k=2, dim=1, largest=False).values

    d1 = top2_dists[:, 0]
    d2 = top2_dists[:, 1]
    memorized = d1 <= ratio * d2
    return memorized.float().mean().item()