import os
from typing import List, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from torch.distributions import MultivariateNormal
from torchvision.utils import save_image, make_grid

from projorg import plotsdir
from ..dataset import closest_component

sns.set_style("whitegrid")
font: dict[str, Any] = {"family": "serif", "style": "normal", "size": 14}
matplotlib.rc("font", **font)
sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
matplotlib.use("Agg")
np.set_printoptions(precision=4)


def plot_nearest_neighbors(
    generated: torch.Tensor,
    train_data: torch.Tensor,
    save_path: str,
    n_show: int = 16,
    precomputed_gen_feats: Optional[torch.Tensor] = None,
    precomputed_train_feats: Optional[torch.Tensor] = None,
) -> None:
    n_show = min(n_show, generated.shape[0])

    if precomputed_gen_feats is not None and precomputed_train_feats is not None:
        gen_feats   = precomputed_gen_feats[:n_show]
        train_feats = precomputed_train_feats
    else:
        # pixel-space fallback
        gen_feats   = generated[:n_show].reshape(n_show, -1)
        train_feats = train_data.reshape(train_data.shape[0], -1)

    dists      = torch.cdist(gen_feats.float(), train_feats.float())
    nn_indices = dists.argmin(dim=1)

    pairs = []
    for i in range(n_show):
        pairs.append(generated[i])
        pairs.append(train_data[nn_indices[i]])

    save_image(torch.stack(pairs), save_path, nrow=2 * int(n_show**0.5), normalize=True)


def compute_cluster_distances(trajectories, predicted_component_labels):
    # Compute pairwise L2 distances
    def pairwise_l2_distances(x, y):
        return ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(-1).sqrt()

    # Number of clusters
    num_clusters = predicted_component_labels.max().item() + 1

    # Get unique labels excluding -1 (noise label)
    unique_labels = torch.unique(
        predicted_component_labels[predicted_component_labels != -1]
    )

    # Initialize arrays to store distances
    distances = torch.zeros(
        num_clusters, trajectories.size(2), unique_labels.size(0)
    )

    # Loop over each time step
    for t in range(trajectories.size(2)):
        # Extract trajectories at time step t
        trajectories_t = trajectories[:, :, t]

        # Loop over each cluster
        for cluster_index in range(num_clusters):
            # Select trajectories belonging to the current cluster
            cluster_mask = predicted_component_labels[:, 0] == cluster_index
            trajectories_cluster = trajectories_t[cluster_mask]

            # Compute pairwise L2 distances within the cluster
            distances_within_cluster = pairwise_l2_distances(
                trajectories_cluster, trajectories_cluster
            )

            # Compute average within-cluster distances
            within_cluster_avg_distance = distances_within_cluster.sum() / (
                trajectories_cluster.size(0)
                * (trajectories_cluster.size(0) - 1)
            )

            distances[cluster_index, t, 0] = within_cluster_avg_distance.item()

            # Loop over other clusters
            for label_index, other_label in enumerate(unique_labels):
                if other_label == cluster_index:
                    continue  # Skip current cluster

                # Select trajectories belonging to the other cluster
                other_cluster_mask = (
                    predicted_component_labels[:, 0] == other_label
                )
                trajectories_other_cluster = trajectories_t[other_cluster_mask]

                # Compute pairwise L2 distances across clusters
                distances_across_clusters = pairwise_l2_distances(
                    trajectories_cluster, trajectories_other_cluster
                )

                # Compute average across-cluster distances
                across_cluster_avg_distance = (
                    distances_across_clusters.sum()
                    / (
                        trajectories_cluster.size(0)
                        * trajectories_other_cluster.size(0)
                    )
                )

                distances[cluster_index, t, label_index] = (
                    across_cluster_avg_distance.item()
                )

    # Final array shape: (num_clusters, num_time_steps, num_other_clusters + 1)
    return distances


def plot_loss(
    args: Any,
    train_loss: List[float],
    val_loss: Optional[List[float]] = [],
    epoch=-1,
) -> None:
    """
    Plot training loss over training iterations and save the plot.

    Parameters:
        args (Any): Arguments for the experiment.
        train_loss (List[float]): List of training losses over iterations.
        val_loss (List[float]): Optional list of training losses over iterations.

    Returns:
        None
    """

    if epoch == -1:
        epoch = args.testing_epoch

    # Create a new figure for plotting
    fig = plt.figure("training logs", figsize=(7, 4))

    # Plot training (and validation) loss
    plt.plot(
        np.linspace(0, epoch + 1, (epoch + 1) * len(train_loss) // (epoch + 1)),
        train_loss,
        color="orange",
        alpha=1.0,
        label="training loss",
    )
    if val_loss:
        plt.plot(
            np.linspace(0, epoch + 1, epoch + 1),
            val_loss,
            color="k",
            alpha=0.8,
            label="validation loss",
        )

    # Format y-axis labels using scientific notation
    plt.ticklabel_format(axis="y", style="sci", useMathText=True)

    # Set plot title and labels
    plt.title("Training loss over training iterations")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Save the plot
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "log.png"),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )

    # Close the figure to release memory
    plt.close(fig)


def plot_cifar100_example(
    args: Any,
    epoch: int,
    trajectories: torch.Tensor,
    labels: torch.Tensor,
    train_loss: List[float],
    val_loss: List[float],
    real_images: torch.Tensor,
) -> None:
    """
    Plot CIFAR100 example.

    Args:
        args (Any): Arguments for plotting.
        trajectories (torch.Tensor): Training data tensor.
        labels (torch.Tensor): Validation data tensor.
        trajectories (torch.Tensor): Trajectories of samples.
        train_loss (List[float]): Training loss values.
        val_loss (List[float]): Validation loss values.
        real_images (torch.Tensor): Real images.
    """

    # Colors for each cluster.
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#ff1493",  # deep pink
        "#00ced1",  # dark turquoise
    ]

    # Plot training loss.
    plot_loss(args, train_loss, val_loss=val_loss, epoch=epoch)

    num_samples = trajectories.shape[0]

    with torch.no_grad():
        trajectories = trajectories.cpu()
        image = trajectories[..., -1]

        # Reshape, permute, and flatten the tensor.
        trajectories = (
            trajectories.view(
                num_samples,
                -1,
                args.nt,
            )
            .permute(0, 2, 1)
            .reshape(num_samples * args.nt, -1)
        )

        # Perform PCA with two components.
        u_matrix = torch.pca_lowrank(trajectories, q=2)[0]

        # Reshape the output tensor back to the desired shape.
        trajectories = u_matrix.reshape(
            num_samples,
            args.nt,
            -1,
        ).permute(0, 2, 1)

    image_grid = make_grid(
        image[:256],
        normalize=True,
        value_range=(-1, 1),
        nrow=16,
    )
    save_image(
        image_grid,
        f"{os.path.join(plotsdir(args.experiment))}/"
        f"sampled-images_{epoch:04d}.png",
    )

    # Real images.
    image_grid = make_grid(
        real_images[:256],
        normalize=True,
        value_range=(-1, 1),
        nrow=16,
    )
    save_image(
        image_grid,
        f"{os.path.join(plotsdir(args.experiment))}/"
        f"true-images_{epoch:04d}.png",
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Turn values in labels into colors. The labels could be arbitrary numbers
    # and we want to map them to values 0 to number of unique labels before
    # assigning colors.
    labels = labels.long()
    unique_labels = torch.unique(labels)

    # Replace the labels with the corresponding index in the unique_labels
    # tensor.
    for i, label in enumerate(unique_labels):
        labels[labels == label] = i

    for i, sample_trajectory in enumerate(trajectories):
        ax.plot(
            np.arange(args.nt),
            sample_trajectory[0, :].cpu(),
            sample_trajectory[1, :].cpu(),
            color=colors[labels[i].long().item() % len(colors)],
            alpha=0.1,
        )

    ylim_max_abs = max(np.abs(ax.get_ylim()))
    zlim_max_abs = max(np.abs(ax.get_zlim()))
    ax.set_ylim([-ylim_max_abs, ylim_max_abs])
    ax.set_zlim([-zlim_max_abs, zlim_max_abs])
    ax.set_xlim([0, args.nt])

    # Generating the grid
    y_grid = torch.linspace(
        *ax.get_ylim(),
        200,
    ).to(trajectories.device)
    z_grid = torch.linspace(
        *ax.get_zlim(),
        200,
    ).to(trajectories.device)

    # Creating meshgrid
    y_grid, z_grid = torch.meshgrid(y_grid, z_grid)

    latent_dist = MultivariateNormal(
        torch.zeros([2], device=trajectories.device),
        torch.eye(2, device=trajectories.device),
    )
    latent_log_prob = latent_dist.log_prob(
        torch.stack(
            [y_grid, z_grid],
            dim=-1,
        )
    ).cpu()

    # Plot the log-likelihoods.
    ax.contourf(
        latent_log_prob,
        y_grid.cpu(),
        z_grid.cpu(),
        1000,
        zdir="x",
        offset=0,
        vmin=-30,
        alpha=0.7,
    )

    # Adjust the first value (X-axis) to change the aspect ratio
    ax.set_box_aspect([3, 1, 1])

    # Remove grid lines
    ax.grid(False)
    # Change the size and font of tick labels on the X-axis
    ax.tick_params(axis="x", labelsize=12)
    ax.set_yticks([])
    ax.set_zticks([])

    # Set plot title and labels
    ax.set_title(
        "Trajectory of samples during reverse diffusion", y=1.0, pad=-14
    )
    ax.set_xlabel("Diffusion timestep", labelpad=20)

    # Rotate the 3D plot
    ax.view_init(elev=30, azim=245)

    plt.savefig(
        os.path.join(plotsdir(args.experiment), "trajectories_angle-0.png"),
        format="png",
        bbox_inches="tight",
        dpi=500,
    )

    # Rotate the 3D plot
    ax.view_init(elev=30, azim=295)

    plt.savefig(
        os.path.join(plotsdir(args.experiment), "trajectories_angle-1.png"),
        format="png",
        bbox_inches="tight",
        dpi=500,
    )

    plt.close(fig)


def plot_gmm_example(
    args: Any,
    trajectories: torch.Tensor,
    train_loss: List[float],
    val_loss: List[float],
    mem_metric: dict[str, List[float]],
    data_dist: torch.distributions.mixture_same_family.MixtureSameFamily,
) -> None:
    """
    Plot Gaussian example.

    Args:
        args (Any): Arguments for plotting.
        trajectories (torch.Tensor): Training data tensor.
        x_val (torch.Tensor): Validation data tensor.
        trajectories (torch.Tensor): Trajectories of samples.
        train_loss (List[float]): Training loss values.
        val_loss (List[float]): Validation loss values.
        mem_metric (dict[str, List[float]]): Memorization metric values.
        data_dist (torch.distributions.mixture_same_family.MixtureSameFamily):
            Data distribution.
    """

    # Plot training loss.
    plot_loss(args, train_loss, val_loss=val_loss)

    # Plot memorization metric.
    fig = plt.figure("memorization metric", figsize=(7, 4))
    for key, values in mem_metric.items():
        # Take into account that the metric is computed every arg.mem_metric_freq
        plt.plot(
            np.linspace(
                0,
                args.mem_metric_freq * (len(values) - 1),
                len(values),
            ),
            values,
            label=key,
        )
    plt.title("Memorization metric over training iterations")
    plt.ylabel("Memorization metric")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(
        os.path.join(plotsdir(args.experiment), "mem_metric.png"),
        format="png",
        bbox_inches="tight",
        dpi=400,
        pad_inches=0.02,
    )
    # Colors for each cluster.
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#ff1493",  # deep pink
        "#00ced1",  # dark turquoise
    ]

    if args.input_size == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for sample_trajectory in trajectories:
            ax.plot(
                sample_trajectory.detach().cpu().numpy()[0],
                color=colors[
                    closest_component(
                        data_dist, sample_trajectory[..., -1]
                    ).item()
                ],
                alpha=0.1,
            )

        ax.set_title("Trajectory of approximated samples")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Sample value")
        ax.set_xlim([0, args.nt])
        plt.savefig(
            os.path.join(
                plotsdir(args.experiment),
                "trajectories.png",
            ),
            format="png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close(fig)

    elif args.input_size == 2:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        predicted_component_labels = torch.stack(
            [
                closest_component(data_dist, sample_trajectory[..., -1])
                for sample_trajectory in trajectories
            ],
            dim=0,
        )

        distances = compute_cluster_distances(
            trajectories, predicted_component_labels
        )

        for sample_trajectory in trajectories:
            ax.plot(
                np.arange(args.nt),
                sample_trajectory[0, :].cpu(),
                sample_trajectory[1, :].cpu(),
                color=colors[
                    closest_component(
                        data_dist, sample_trajectory[..., -1]
                    ).item()
                ],
                alpha=0.1,
            )

        ylim_max_abs = max(np.abs(ax.get_ylim()))
        zlim_max_abs = max(np.abs(ax.get_zlim()))
        ax.set_ylim([-ylim_max_abs, ylim_max_abs])
        ax.set_zlim([-zlim_max_abs, zlim_max_abs])
        ax.set_xlim([0, args.nt])

        # Generating the grid
        y_grid = torch.linspace(
            *ax.get_ylim(),
            200,
        ).to(trajectories.device)
        z_grid = torch.linspace(
            *ax.get_zlim(),
            200,
        ).to(trajectories.device)

        # Creating meshgrid
        y_grid, z_grid = torch.meshgrid(y_grid, z_grid, indexing="ij")

        # Compute data log-likelihood.
        data_log_prob = data_dist.log_prob(
            torch.stack(
                [y_grid, z_grid],
                dim=-1,
            )
        ).cpu()
        # Compute the log-likelihood of the latent distribution.
        latent_dist = MultivariateNormal(
            torch.zeros([args.input_size], device=trajectories.device),
            torch.eye(args.input_size, device=trajectories.device),
        )
        latent_log_prob = latent_dist.log_prob(
            torch.stack(
                [y_grid, z_grid],
                dim=-1,
            )
        ).cpu()

        # Plot the log-likelihoods.
        ax.contourf(
            data_log_prob,
            y_grid.cpu(),
            z_grid.cpu(),
            500,
            zdir="x",
            offset=args.nt,
            vmin=-30,
        )
        ax.contourf(
            latent_log_prob,
            y_grid.cpu(),
            z_grid.cpu(),
            1000,
            zdir="x",
            offset=0,
            vmin=-30,
            alpha=0.7,
        )

        # Adjust the first value (X-axis) to change the aspect ratio
        ax.set_box_aspect([3, 1, 1])

        # Remove grid lines
        ax.grid(False)
        # Change the size and font of tick labels on the X-axis
        ax.tick_params(axis="x", labelsize=12)
        ax.set_yticks([])
        ax.set_zticks([])

        # Set plot title and labels
        ax.set_title(
            "Trajectory of samples during reverse diffusion", y=1.0, pad=-14
        )
        ax.set_xlabel("Diffusion timestep", labelpad=20)

        # Rotate the 3D plot
        ax.view_init(elev=30, azim=245)

        plt.savefig(
            os.path.join(plotsdir(args.experiment), "trajectories_angle-0.png"),
            format="png",
            bbox_inches="tight",
            dpi=500,
        )

        # Rotate the 3D plot
        ax.view_init(elev=30, azim=295)

        plt.savefig(
            os.path.join(plotsdir(args.experiment), "trajectories_angle-1.png"),
            format="png",
            bbox_inches="tight",
            dpi=500,
        )

        plt.close(fig)
