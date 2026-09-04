# pylint: disable=E1102
# pylint: disable=invalid-name
from typing import List

import torch
from torch.distributions import (
    MultivariateNormal,
    Categorical,
    MixtureSameFamily,
)


def closest_component(mixture_dist, x):
    # Compute the PDF of each component at x
    component_probs = mixture_dist.component_distribution.log_prob(x)

    # Compute the weights of each component in the mixture
    weights = mixture_dist.mixture_distribution.probs

    # Compute the overall PDF for each component
    component_pdf = torch.exp(component_probs) * weights.unsqueeze(0)

    # Find the component with the highest PDF value at x
    closest_component_idx = torch.argmax(component_pdf, dim=1)

    return closest_component_idx


def setup_gmm_dist(
    input_size: int,
    mixture_weights: torch.Tensor,
    mixture_means: torch.Tensor,
    mixture_vars: torch.Tensor,
    device: torch.device,
) -> torch.distributions.mixture_same_family.MixtureSameFamily:
    """
    Set up a Gaussian mixture model distribution.

    Args:
        input_size (int): Dimensionality of the input data.
        mixture_weights (torch.Tensor): Weights of the mixture components.
        mixture_means (torch.Tensor): Means of the mixture components.
            Can be either:
            - A list of scalars (one per component) to be replicated across dimensions
            - A list of vectors (each of size input_size) for full mean specification
        mixture_vars (torch.Tensor): Variance of the mixture components.
            Can be either:
            - A list of scalars (one per component) for isotropic covariance
            - A list of vectors (each of size input_size) for diagonal covariance
        device (torch.device): Device on which the tensors are created.

    Returns:
        torch.distributions.mixture_same_family.MixtureSameFamily: Gaussian
            mixture model data distribution.
    """

    # Convert the input parameters to tensors.
    mixture_weights: torch.Tensor = torch.tensor(mixture_weights, device=device)
    # Normalize the mixture weights.
    mixture_weights = mixture_weights / mixture_weights.sum()

    # Process means: check if scalars or vectors
    processed_means = []
    for mean in mixture_means:
        if isinstance(mean, (int, float)):
            # Scalar case: replicate across all dimensions
            processed_means.append(
                torch.full((input_size,), float(mean), device=device)
            )
        else:
            # Vector case: use as-is
            mean_tensor = torch.tensor(mean, device=device)
            if mean_tensor.numel() != input_size:
                raise ValueError(
                    f"Mean vector has size {mean_tensor.numel()}, expected {input_size}"
                )
            processed_means.append(mean_tensor)

    mixture_means = torch.stack(processed_means)

    # Process variances: check if scalars or vectors
    processed_vars = []
    for var in mixture_vars:
        if isinstance(var, (int, float)):
            # Scalar case: isotropic covariance (same variance for all dimensions)
            processed_vars.append(
                torch.eye(input_size, device=device) * float(var)
            )
        else:
            # Vector case: diagonal covariance with different variances per dimension
            var_tensor = torch.tensor(var, device=device)
            if var_tensor.numel() != input_size:
                raise ValueError(
                    f"Variance vector has size {var_tensor.numel()}, expected {input_size}"
                )
            processed_vars.append(torch.diag(var_tensor))

    mixture_vars = torch.stack(processed_vars)

    # Define mixing proportions for each component
    categorial_dist = Categorical(mixture_weights)

    # Create MultivariateNormal distributions for each component in batch
    component_dists = MultivariateNormal(mixture_means, mixture_vars)

    # Construct the Gaussian mixture model.
    gmm_dist = MixtureSameFamily(categorial_dist, component_dists)

    return gmm_dist
