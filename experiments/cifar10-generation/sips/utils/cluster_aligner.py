import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MiniBatchKMeans

from sips.dataset import setup_gmm_dist
from sips.utils import plotsdir


def pairwise_distances(C: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between cluster centers."""
    return np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)


class _OrthogonalClusterer:
    """Drop-in replacement for MiniBatchKMeans that partitions space using
    random orthogonal directions.  Works reliably in high dimensions where
    K-means on isotropic Gaussian noise degenerates.

    Samples are assigned to the cluster whose direction has the largest
    projection (i.e., argmax of X @ directions.T).  By symmetry of the
    Gaussian distribution, each cluster receives roughly 1/k of the samples.
    """

    def __init__(self, n_clusters: int, n_features: int) -> None:
        Q, _ = np.linalg.qr(np.random.randn(n_features, n_clusters))
        self.cluster_centers_ = Q[:, :n_clusters].T  # (n_clusters, n_features)
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray) -> "_OrthogonalClusterer":
        self.labels_ = self.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        projections = X @ self.cluster_centers_.T  # (n_samples, n_clusters)
        return np.argmax(projections, axis=1)


class ClusterAligner:
    """
    Generates and organizes noise samples into clusters using KMeans
    clustering, with support for dual-dataset clustering and
    Gromov-Wasserstein alignment.

    Capabilities:
    - Fit KMeans to data and noise datasets
    - Predict cluster assignments for new samples
    - Generate synthetic noise samples organized into clusters
    - Align clusters between datasets using optimal transport
    """

    def __init__(
        self, n_clusters: int = 5, noise_grouping: str = "auto"
    ) -> None:
        """
        Initialize the ClusterAligner.

        Args:
            n_clusters: Number of clusters for KMeans. Defaults to 5.
            noise_grouping: How to partition noise samples into clusters.
                "auto" (default) uses orthogonal directions when
                n_features >= n_clusters, else K-means.
                "kmeans" always uses MiniBatchKMeans.
                "orthogonal" always uses orthogonal directions.
        """
        if noise_grouping not in ("auto", "kmeans", "orthogonal"):
            raise ValueError(
                f"noise_grouping must be 'auto', 'kmeans', or 'orthogonal', "
                f"got '{noise_grouping}'"
            )
        self.n_clusters = n_clusters
        self.noise_grouping = noise_grouping

        # KMeans models for both datasets
        self.data_kmeans: Optional[KMeans] = None
        self.noise_kmeans: Optional[KMeans] = None

        # Cluster labels and weights
        self.data_labels: Optional[np.ndarray] = None
        self.noise_labels: Optional[np.ndarray] = None
        self.data_weights: Optional[np.ndarray] = None
        self.noise_weights: Optional[np.ndarray] = None

        # Cluster alignment mapping (data_cluster_id ->
        # noise_cluster_id)
        self.cluster_mapping: Optional[Dict[int, int]] = None

    def find_clusters(
        self,
        input_data: Optional[np.ndarray] = None,
        input_noise: Optional[np.ndarray] = None,
    ) -> None:
        """
        Fit KMeans models to the provided data and/or noise.

        Args:
            input_data: Data to cluster, shape (n_samples, n_features)
            input_noise: Noise to cluster, shape (n_samples, n_features)

        Raises:
            ValueError: If both input_data and input_noise are None
        """
        if input_data is None and input_noise is None:
            raise ValueError(
                "At least one of input_data or input_noise must be provided"
            )

        if input_data is not None:
            self.data_kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters, batch_size=1024
            )
            self.data_kmeans.fit(input_data)
            self.data_labels = self.data_kmeans.predict(input_data)
            self.data_weights = np.bincount(
                self.data_labels, minlength=self.n_clusters
            ) / len(self.data_labels)

        if input_noise is not None:
            n_features = input_noise.shape[1]
            use_orthogonal = (
                self.noise_grouping == "orthogonal"
                or (
                    self.noise_grouping == "auto"
                    and n_features >= self.n_clusters
                )
            )
            if use_orthogonal:
                self.noise_kmeans = _OrthogonalClusterer(
                    self.n_clusters, n_features
                )
            else:
                self.noise_kmeans = MiniBatchKMeans(
                    n_clusters=self.n_clusters, batch_size=1024
                )
            self.noise_kmeans.fit(input_noise)
            self.noise_labels = self.noise_kmeans.predict(input_noise)
            self.noise_weights = np.bincount(
                self.noise_labels, minlength=self.n_clusters
            ) / len(self.noise_labels)

    def predict_clusters(
        self, input_array: np.ndarray, dataset_type: str = "data"
    ) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Args:
            input_array: Data to predict, shape (n_samples, n_features)
            dataset_type: Which KMeans to use - "data" or "noise"

        Returns:
            Array of cluster labels for each sample

        Raises:
            ValueError: If the specified KMeans model hasn't been fitted
        """
        if dataset_type == "data":
            if self.data_kmeans is None:
                raise ValueError(
                    "Data KMeans not fitted. Call find_clusters with input_data first."
                )
            return self.data_kmeans.predict(input_array)
        elif dataset_type == "noise":
            if self.noise_kmeans is None:
                raise ValueError(
                    "Noise KMeans not fitted. Call find_clusters with input_noise first."
                )
            return self.noise_kmeans.predict(input_array)
        else:
            raise ValueError("dataset_type must be either 'data' or 'noise'")

    def _gw_cluster_match(
        self, regularization: float = 1e-2, max_iterations: int = 10000
    ) -> Dict[int, int]:
        """
        Match data clusters to noise clusters using Gromov-Wasserstein
        distance.

        Args:
            regularization: Regularization parameter for entropic GW
            max_iterations: Maximum iterations for optimization

        Returns:
            Mapping from data cluster IDs to noise cluster IDs

        Raises:
            ValueError: If both KMeans models haven't been fitted
        """
        if self.data_kmeans is None or self.noise_kmeans is None:
            raise ValueError(
                "Both data and noise KMeans must be fitted before alignment"
            )

        # Get cluster centers and weights
        data_centers = self.data_kmeans.cluster_centers_
        noise_centers = self.noise_kmeans.cluster_centers_
        data_weights = self.data_weights
        noise_weights = self.noise_weights

        # Compute normalized pairwise distance matrices
        data_distances = pairwise_distances(data_centers)
        noise_distances = pairwise_distances(noise_centers)
        data_distances = data_distances / (1e-8 + data_distances.max())
        noise_distances = noise_distances / (1e-8 + noise_distances.max())

        # Compute entropic Gromov-Wasserstein optimal transport plan
        transport_plan = ot.gromov.entropic_gromov_wasserstein(
            data_distances,
            noise_distances,
            data_weights,
            noise_weights,
            loss_fun="square_loss",
            epsilon=regularization,
            max_iter=max_iterations,
        )

        # Extract one-to-one mapping using Hungarian algorithm
        data_indices, noise_indices = linear_sum_assignment(-transport_plan)

        return {
            int(data_idx): int(noise_idx)
            for data_idx, noise_idx in zip(data_indices, noise_indices)
        }

    def align_clusters(
        self, regularization: float = 5e-2, max_iterations: int = 1000
    ) -> Dict[int, int]:
        """
        Align data clusters with noise clusters using Gromov-Wasserstein
        distance.

        Args:
            regularization: Regularization parameter for entropic GW
            max_iterations: Maximum iterations for optimization

        Returns:
            Mapping from data cluster IDs to noise cluster IDs
        """
        self.cluster_mapping = self._gw_cluster_match(
            regularization=regularization, max_iterations=max_iterations
        )
        return self.cluster_mapping

    def get_aligned_labels(
        self, labels: np.ndarray, direction: str = "noise_to_data"
    ) -> np.ndarray:
        """
        Apply cluster alignment mapping to transform labels.

        Args:
            labels: Original cluster labels direction: Mapping direction
            - "noise_to_data" or "data_to_noise"

        Returns:
            Labels aligned according to the mapping

        Raises:
            ValueError: If clusters haven't been aligned or invalid
            direction
        """
        if self.cluster_mapping is None:
            raise ValueError(
                "Clusters must be aligned first. Call align_clusters()."
            )

        if direction == "noise_to_data":
            # Invert mapping: data->noise becomes noise->data
            inverse_mapping = {v: k for k, v in self.cluster_mapping.items()}
            return np.array([inverse_mapping[label] for label in labels])
        elif direction == "data_to_noise":
            # Use direct mapping: data->noise
            return np.array([self.cluster_mapping[label] for label in labels])
        else:
            raise ValueError(
                "direction must be either 'noise_to_data' or 'data_to_noise'"
            )

    def sample_noise(
        self,
        data_labels: np.ndarray,
        rigidness: float = 8e-1,
    ) -> np.ndarray:
        """
        Generate synthetic noise samples matching the cluster
        distribution of target labels.

        For each sample, with probability ``rigidness`` the noise is
        standard Gaussian drawn from the Voronoi cell of the corresponding
        noise cluster.  With probability ``1 - rigidness`` the noise is
        unconstrained standard Gaussian.

        Args:
            data_labels: Data cluster labels defining desired distribution
            rigidness: Controls probability of sampling from cluster vs
                gaussian noise. Higher values favor cluster sampling
                (0.0 to 1.0).
        Returns:
            Noise samples ordered to match data_labels sequence, shape
            (len(data_labels), n_features)
        Raises:
            ValueError: If noise KMeans hasn't been fitted
        """
        if self.noise_kmeans is None:
            raise ValueError(
                "Noise KMeans not fitted. Call find_clusters with input_noise first."
            )

        # Perform alignment if not already done
        if self.cluster_mapping is None:
            print("Cluster alignment not found. Performing alignment...")
            self.align_clusters()

        n_samples = len(data_labels)
        n_features = self.noise_kmeans.cluster_centers_.shape[1]

        # Start with random Gaussian noise for all positions
        all_noises = np.random.randn(n_samples, n_features).astype(np.float32)

        # Determine how many samples get cluster-aligned noise
        n_cluster_samples = int(np.ceil(n_samples * rigidness))
        if n_cluster_samples == 0:
            return all_noises

        # Pick which positions get cluster-aligned noise
        cluster_positions = np.random.choice(
            n_samples, size=n_cluster_samples, replace=False
        )

        # Map data labels -> noise cluster labels for selected positions
        target_noise_labels = self.get_aligned_labels(
            data_labels[cluster_positions], direction="data_to_noise"
        )

        # Count how many samples are needed per noise cluster
        unique_nc, nc_counts = np.unique(
            target_noise_labels, return_counts=True
        )
        needed = dict(zip(unique_nc, nc_counts))
        max_needed = nc_counts.max()

        # Build a reservoir of classified noise samples per cluster.
        # Size the pool so each cluster gets ~max_needed samples
        # (each cluster receives pool_size / k on average).
        pool_by_cluster: Dict[int, np.ndarray] = {}
        pool_size = self.n_clusters * max_needed * 2
        noise_pool = np.random.randn(pool_size, n_features)
        pool_labels = self.noise_kmeans.predict(noise_pool)
        for c in unique_nc:
            pool_by_cluster[c] = noise_pool[pool_labels == c]

        # Top up any cluster that is still short (rare with balanced
        # orthogonal clusters, but handles worst-case imbalance).
        while any(
            len(pool_by_cluster[c]) < needed[c] for c in unique_nc
        ):
            extra = np.random.randn(pool_size, n_features)
            extra_labels = self.noise_kmeans.predict(extra)
            for c in unique_nc:
                if len(pool_by_cluster[c]) < needed[c]:
                    pool_by_cluster[c] = np.concatenate(
                        [pool_by_cluster[c], extra[extra_labels == c]]
                    )

        # Assign cluster-aligned noise to each selected position
        cluster_counters: Dict[int, int] = {c: 0 for c in unique_nc}
        for i, pos in enumerate(cluster_positions):
            nc = target_noise_labels[i]
            idx = cluster_counters[nc]
            all_noises[pos] = pool_by_cluster[nc][idx]
            cluster_counters[nc] = idx + 1

        return all_noises

    @property
    def data_centers(self) -> Optional[np.ndarray]:
        """Get data cluster centers."""
        return (
            self.data_kmeans.cluster_centers_
            if self.data_kmeans is not None
            else None
        )

    @property
    def noise_centers(self) -> Optional[np.ndarray]:
        """Get noise cluster centers."""
        return (
            self.noise_kmeans.cluster_centers_
            if self.noise_kmeans is not None
            else None
        )


class RawAligner:
    """
    Aligns individual noise samples to data samples using
    Gromov-Wasserstein optimal transport on raw pairwise distances.

    Unlike ClusterAligner, this operates directly on samples without
    any clustering — each data point is matched to a noise point.
    """

    def __init__(self, data_samples: np.ndarray) -> None:
        """
        Initialize the RawAligner with data samples.

        Args:
            data_samples: Data samples, shape (n_samples, n_features).
        """
        self.data_samples = data_samples
        self.n_samples = data_samples.shape[0]
        self.n_features = data_samples.shape[1]

        # Precompute normalized pairwise distances for data.
        self.data_distances = pairwise_distances(data_samples)
        self.data_distances = self.data_distances / (
            1e-8 + self.data_distances.max()
        )

        # Populated by align_samples().
        self.sample_mapping: Optional[Dict[int, int]] = None

    def _gw_sample_match(
        self,
        noise_samples: np.ndarray,
        regularization: float = 1e-2,
        max_iterations: int = 10000,
    ) -> Dict[int, int]:
        """
        Match data samples to noise samples using Gromov-Wasserstein
        distance on raw pairwise distance matrices.

        Args:
            noise_samples: Noise samples, shape (n_samples, n_features).
            regularization: Regularization parameter for entropic GW.
            max_iterations: Maximum iterations for optimization.

        Returns:
            Mapping from data sample index to noise sample index.
        """
        # Compute normalized pairwise distances for noise.
        noise_distances = pairwise_distances(noise_samples)
        noise_distances = noise_distances / (
            1e-8 + noise_distances.max()
        )

        # Uniform weights over samples.
        data_weights = np.ones(self.n_samples) / self.n_samples
        noise_weights = np.ones(self.n_samples) / self.n_samples

        # Compute entropic Gromov-Wasserstein optimal transport plan.
        transport_plan = ot.gromov.entropic_gromov_wasserstein(
            self.data_distances,
            noise_distances,
            data_weights,
            noise_weights,
            loss_fun="square_loss",
            epsilon=regularization,
            max_iter=max_iterations,
        )

        # Extract one-to-one mapping using Hungarian algorithm.
        data_indices, noise_indices = linear_sum_assignment(-transport_plan)

        return {
            int(d): int(n)
            for d, n in zip(data_indices, noise_indices)
        }

    def align_samples(
        self,
        noise_samples: np.ndarray,
        regularization: float = 5e-2,
        max_iterations: int = 1000,
    ) -> Dict[int, int]:
        """
        Align data samples with noise samples using Gromov-Wasserstein
        distance.

        Args:
            noise_samples: Noise samples, shape (n_samples, n_features).
            regularization: Regularization parameter for entropic GW.
            max_iterations: Maximum iterations for optimization.

        Returns:
            Mapping from data sample index to noise sample index.
        """
        self.sample_mapping = self._gw_sample_match(
            noise_samples,
            regularization=regularization,
            max_iterations=max_iterations,
        )
        return self.sample_mapping

    def sample_noise(
        self,
        rigidness: float = 1.0,
        n_features: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate noise samples aligned to data samples.

        For each data sample, with probability ``rigidness`` a fresh
        Gaussian noise sample is drawn and permuted according to the
        GW alignment.  With probability ``1 - rigidness`` an
        unconstrained standard Gaussian noise is used instead.

        Args:
            rigidness: Controls probability of using aligned noise vs
                random Gaussian (0.0 to 1.0).
            n_features: Dimensionality of the noise. Defaults to the
                data dimensionality (``self.n_features``).

        Returns:
            Noise samples, shape (n_samples, n_features).
        """
        if n_features is None:
            n_features = self.n_features

        # Draw fresh noise and compute a new alignment.
        noise_samples = np.random.randn(
            self.n_samples, n_features
        ).astype(np.float32)

        # Alignment is based on the data pairwise distances (already
        # computed in __init__), so we only need to compute the noise
        # pairwise distances for the GW matching.  When n_features
        # differs from self.n_features, the alignment still works — it
        # just matches the distance structure in the noise space.
        mapping = self.align_samples(noise_samples)

        # Permute noise according to the GW mapping.
        aligned_noise = np.empty_like(noise_samples)
        for data_idx, noise_idx in mapping.items():
            aligned_noise[data_idx] = noise_samples[noise_idx]

        if rigidness >= 1.0:
            return aligned_noise

        # Mix: each position independently uses aligned or random noise.
        random_noise = np.random.randn(
            self.n_samples, n_features
        ).astype(np.float32)
        use_aligned = (
            np.random.rand(self.n_samples) < rigidness
        )[:, None]
        return np.where(use_aligned, aligned_noise, random_noise)


def create_gmm_parameters() -> Tuple[
    List[float], List[List[float]], List[List[float]]
]:
    """Define standard GMM parameters for testing."""
    cluster_weights = np.random.dirichlet(alpha=[1.0] * 10).tolist()
    cluster_means = np.random.uniform(-10, 10, size=(10, 2)).tolist()
    cluster_variances = np.random.uniform(0.5, 1.5, size=(10, 2)).tolist()
    return cluster_weights, cluster_means, cluster_variances


def visualize_alignment(
    cluster_aligner: ClusterAligner,
    plot_data: np.ndarray,
    plot_data_labels: np.ndarray,
    plot_noise_samples: np.ndarray,
    rigidness: float,
    noise_grouping: str = "kmeans",
) -> None:
    """Create visualization plots for cluster alignment."""
    n_clusters = cluster_aligner.n_clusters
    colors = [f"C{i}" for i in range(n_clusters)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot data clusters
    for cluster_id in range(n_clusters):
        cluster_mask = plot_data_labels == cluster_id
        if np.any(cluster_mask):
            axes[0].scatter(
                plot_data[cluster_mask, 0],
                plot_data[cluster_mask, 1],
                color=colors[cluster_id],
                alpha=0.5,
                s=15,
                label=f"Data cluster {cluster_id}",
            )

        # Plot cluster centers
        if cluster_aligner.data_centers is not None:
            axes[0].scatter(
                cluster_aligner.data_centers[cluster_id, 0],
                cluster_aligner.data_centers[cluster_id, 1],
                color="black",
                marker="x",
                s=50,
                linewidth=3,
            )

    # Extend y-axis for legend space
    y_min, y_max = axes[0].get_ylim()
    y_range = y_max - y_min

    x_min, x_max = axes[0].get_xlim()
    x_range = x_max - x_min
    axes[0].set_xlim(
        min(x_min, y_min) - 0.1 * min(x_range, y_range),
        max(x_max, y_max) + 0.1 * min(x_range, y_range),
    )
    axes[0].set_ylim(
        min(x_min, y_min), max(x_max, y_max) + 0.2 * min(x_range, y_range)
    )

    axes[0].set_title("Data clusters (GMM)", fontsize=14)
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncols=3, loc="upper left", fontsize=10)

    # Plot aligned noise samples
    for cluster_id in range(n_clusters):
        cluster_mask = plot_data_labels == cluster_id
        if np.any(cluster_mask) and cluster_aligner.cluster_mapping:
            mapped_noise_cluster = cluster_aligner.cluster_mapping[cluster_id]
            axes[1].scatter(
                plot_noise_samples[cluster_mask, 0],
                plot_noise_samples[cluster_mask, 1],
                color=colors[cluster_id],
                alpha=0.5,
                s=15,
                label=f"Noise cluster {mapped_noise_cluster}",
            )

        # Plot noise cluster centers
        if (
            cluster_aligner.noise_centers is not None
            and cluster_aligner.cluster_mapping
            and cluster_id in cluster_aligner.cluster_mapping
        ):
            mapped_noise_cluster = cluster_aligner.cluster_mapping[cluster_id]
            axes[1].scatter(
                cluster_aligner.noise_centers[mapped_noise_cluster, 0],
                cluster_aligner.noise_centers[mapped_noise_cluster, 1],
                color="black",
                marker="x",
                s=50,
                linewidth=3,
            )

    # Extend y-axis for legend space
    y_min, y_max = axes[1].get_ylim()
    y_range = y_max - y_min
    axes[1].set_ylim(y_min, y_max + 0.2 * y_range)
    axes[1].set_xlim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    axes[1].set_title(
        f"Aligned noise ({noise_grouping}), rigidness {rigidness:.2f}",
        fontsize=14,
    )
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncols=3, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plotsdir("."),
            f"cluster_alignment_{noise_grouping}_rigidness-{rigidness:.2f}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def visualize_raw_alignment(
    data_samples: np.ndarray,
    noise_samples: np.ndarray,
    rigidness: float,
    sample_colors: np.ndarray,
) -> None:
    """Create visualization plots for raw sample alignment.

    Each data point is colored by ``sample_colors`` (a continuous scalar
    per sample, e.g. position along a Swiss roll).  The matched noise
    point receives the same color, so smooth color gradients in the
    noise panel indicate neighborhood-preserving alignment.

    Args:
        data_samples: Data samples, shape (n_samples, n_features).
        noise_samples: Aligned noise samples, shape (n_samples, n_features).
        rigidness: Rigidness value used when generating the noise.
        sample_colors: Scalar array of length n_samples used for coloring.
    """
    cmap = "Spectral"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc0 = axes[0].scatter(
        data_samples[:, 0],
        data_samples[:, 1],
        c=sample_colors,
        cmap=cmap,
        alpha=0.7,
        s=15,
    )
    axes[0].set_title("Data samples (Swiss roll)", fontsize=14)
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(sc0, ax=axes[0], label="position along roll")

    sc1 = axes[1].scatter(
        noise_samples[:, 0],
        noise_samples[:, 1],
        c=sample_colors,
        cmap=cmap,
        alpha=0.7,
        s=15,
    )
    axes[1].set_title(
        f"Aligned noise (raw), rigidness {rigidness:.2f}",
        fontsize=14,
    )
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(sc1, ax=axes[1], label="position along roll")

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plotsdir("."),
            f"raw_alignment_rigidness-{rigidness:.2f}.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    """
    Demonstrate ClusterAligner and RawAligner functionality with GMM data.

    Generates figures for both noise grouping methods (kmeans and orthogonal)
    at rigidness values of 1.0 and 0.8, followed by RawAligner demo.
    """

    # Configuration
    n_clusters = 10
    n_training_samples = 50000
    n_plot_samples = 2000  # Smaller for cleaner visualization

    # Create GMM distribution
    cluster_weights, cluster_means, cluster_variances = create_gmm_parameters()

    gmm_distribution = setup_gmm_dist(
        2, cluster_weights, cluster_means, cluster_variances, device="cpu"
    )

    # Generate training data (shared across both methods)
    training_data = gmm_distribution.sample((n_training_samples,)).cpu().numpy()
    training_noise = np.random.randn(n_training_samples, 2)
    plot_data = gmm_distribution.sample((n_plot_samples,)).cpu().numpy()

    n_features = training_data.shape[1]

    for grouping in ("kmeans", "orthogonal"):
        # Orthogonal needs n_features >= n_clusters; cap if needed
        k = n_clusters
        if grouping == "orthogonal" and n_features < n_clusters:
            k = n_features
            print(
                f"\nNote: orthogonal requires n_features >= n_clusters; "
                f"using {k} clusters for 2D demo "
                f"(in high-dim MNIST/CIFAR this is not an issue)"
            )

        print(f"\n--- noise_grouping = {grouping}, n_clusters = {k} ---")
        cluster_aligner = ClusterAligner(
            n_clusters=k, noise_grouping=grouping
        )
        cluster_aligner.find_clusters(
            input_data=training_data, input_noise=training_noise
        )

        cluster_mapping = cluster_aligner.align_clusters()
        print(f"Cluster mapping (data -> noise): {cluster_mapping}")

        plot_data_labels = cluster_aligner.predict_clusters(plot_data, "data")

        for rigidness in (1.0, 0.8):
            plot_noise_samples = cluster_aligner.sample_noise(
                plot_data_labels, rigidness=rigidness
            )
            visualize_alignment(
                cluster_aligner,
                plot_data,
                plot_data_labels,
                plot_noise_samples,
                rigidness,
                noise_grouping=grouping,
            )
            print(
                f"  Saved figure for rigidness={rigidness:.2f}, "
                f"grouping={grouping}"
            )

    # --- RawAligner demo ---
    # Use a Swiss roll dataset: its clear spatial structure (position
    # along the roll) makes neighborhood-preserving GW alignment easy
    # to see.  Each point is colored by its position parameter `t`.
    from sklearn.datasets import make_swiss_roll

    n_raw_samples = 500
    X, t = make_swiss_roll(n_samples=n_raw_samples, noise=0.3, random_state=42)
    raw_data = X[:, [0, 2]]  # 2D projection (x and z coordinates)
    sample_colors = t  # color by position along the roll

    print(f"\n--- RawAligner (Swiss roll), n_samples = {n_raw_samples} ---")
    raw_aligner = RawAligner(raw_data)

    for rigidness in (1.0, 0.8):
        raw_noise = raw_aligner.sample_noise(rigidness=rigidness)
        visualize_raw_alignment(
            raw_data, raw_noise, rigidness, sample_colors
        )
        print(f"  Saved figure for rigidness={rigidness:.2f}, method=raw")


if __name__ == "__main__":
    main()
