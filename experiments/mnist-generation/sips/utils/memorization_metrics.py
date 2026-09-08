import torch


def euclidean_mem_metric(
    x0: torch.Tensor, samples: torch.Tensor, tol: list[float]
) -> torch.Tensor:
    pairwise_dist = torch.norm(x0[:, None, :] - samples[None, :, :], dim=2)

    num_memorized_samples = (pairwise_dist < tol).any(dim=0).sum().item()
    return num_memorized_samples / x0.shape[0]
