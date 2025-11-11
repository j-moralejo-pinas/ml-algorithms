"""Data generation utilities for synthetic datasets."""

import math
import random
from typing import Any

from jax import numpy as jnp


def linear_regression_dataset(
    n_features: int, n_samples: int
) -> tuple[list[list[float]], list[float]]:
    """
    Generate a synthetic dataset for linear regression.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_samples : int
        Number of samples.

    Returns
    -------
    list[list[float]]
        Input features.
    list[float]
        Target values.
    """
    coefficients = [random.gauss() for _ in range(n_features)]
    intercept = random.gauss()

    x_values = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]

    return x_values, [
        random.gauss(
            sum(x * coeff for x, coeff in zip(x_inst, coefficients, strict=True)) + intercept
        )
        for x_inst in x_values
    ]


def binary_classification_dataset(
    n_features: int, n_samples: int
) -> tuple[list[list[float]], list[float]]:
    """
    Generate a synthetic dataset for binary classification.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_samples : int
        Number of samples.

    Returns
    -------
    list[list[float]]
        Input features.
    list[float]
        Target values (0 or 1).
    """
    coefficients = [random.gauss() for _ in range(n_features)]
    intercept = random.gauss()

    x_values = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]

    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-z))

    y_vals = [
        sigmoid(sum(x * coeff for x, coeff in zip(x_inst, coefficients, strict=True)) + intercept)
        for x_inst in x_values
    ]
    min_y = min(y_vals)
    max_y = max(y_vals)

    return x_values, [1.0 if y >= random.uniform(min_y, max_y) else 0.0 for y in y_vals]


def clustering_dataset(
    n_features: int, n_samples: int, n_clusters: int, cluster_spread: float = 0.1
) -> tuple[list[list[float]], list[int]]:
    """
    Generate a synthetic dataset for clustering.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_samples : int
        Number of samples.
    n_clusters : int
        Number of clusters.
    cluster_spread : float, optional
        Spread of each cluster, by default 0.1.

    Returns
    -------
    list[list[float]]
        Input features.
    list[int]
        Cluster assignments.
    """
    cluster_centers = [
        [random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_clusters)
    ]

    x_values = []
    y_values = []

    for _ in range(n_samples):
        cluster_id = random.randint(0, n_clusters - 1)
        center = cluster_centers[cluster_id]
        point = [random.gauss(mu=coord, sigma=cluster_spread) for coord in center]
        x_values.append(point)
        y_values.append(cluster_id)

    return x_values, y_values


def constrained_clustering_dataset(
    n_features: int, n_samples: int, n_clusters: int
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Generate a synthetic dataset for constrained clustering.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_samples : int
        Number of samples.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    list[list[float]]
        Input features.
    list[list[int]]
        Constraints matrix (1 for must-link, 0 for no constraint).
    """
    cluster_centers = [
        [random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_clusters)
    ]

    x_values = [[random.uniform(-1, 1) for _ in range(n_features)] for _ in range(n_samples)]
    y_values = []

    nodes_in_cluster = {i: [] for i in range(n_clusters)}

    for idx, x in enumerate(x_values):
        # Assign each point to the nearest cluster center
        closest_center = min(
            range(n_clusters),
            key=lambda i: sum((x[j] - cluster_centers[i][j]) ** 2 for j in range(n_features)),
        )
        y_values.append(closest_center)
        nodes_in_cluster[closest_center].append(idx)
    constrains = [[0 for _ in range(n_samples)] for _ in range(n_samples)]
    for cluster_idx_i, cluster_nodes_i in nodes_in_cluster.items():
        for cluster_idx_j, cluster_nodes_j in nodes_in_cluster.items():
            if cluster_idx_i != cluster_idx_j:
                for node_i in cluster_nodes_i:
                    for node_j in cluster_nodes_j:
                        constrains[node_i][node_j] = 1

    return x_values, constrains


def generate_dataset(
    dataset_style: str,
    data_structure_type: str,
    n_features: int,
    n_samples: int,
    n_clusters: int = 3,
) -> tuple[Any, Any]:
    """
    Generate a dataset based on the specified style and data structure type.

    Parameters
    ----------
    dataset_style : str
        The style of the dataset to generate.
    data_structure_type : str
        The type of data structure to use for the dataset.
    n_features : int
        The number of features in the dataset.
    n_samples : int
        The number of samples in the dataset.
    n_clusters : int, optional
        The number of clusters (for clustering datasets), by default 3.

    Returns
    -------
    Any
        Generated dataset in the specified data structure type.
    Any
        Generated target values in the specified data structure type.

    Raises
    ------
    ValueError
        If an unsupported dataset style or data structure type is provided.
    """
    if dataset_style == "linear_regression":
        x, y = linear_regression_dataset(n_features=n_features, n_samples=n_samples)
    elif dataset_style == "binary_classification":
        x, y = binary_classification_dataset(n_features=n_features, n_samples=n_samples)
    elif dataset_style == "clustering":
        x, y = clustering_dataset(n_features=n_features, n_samples=n_samples, n_clusters=n_clusters)
    elif dataset_style == "constrained_clustering":
        x, y = constrained_clustering_dataset(
            n_features=n_features, n_samples=n_samples, n_clusters=n_clusters
        )
    else:
        raise ValueError

    if data_structure_type == "jax":
        return jnp.array(x), jnp.array(y)

    raise ValueError
