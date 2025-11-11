"""K-Means Clustering Algorithm Implementation."""

from functools import partial

import jax
from jax import Array
from jax import numpy as jnp

from ml_algorithms.unsupervised_learning.base_ul_algo import BaseULAlgo


@jax.jit
def get_ranking_distances(points: Array, queries: Array) -> Array:
    """
    Compute the squared L2 distances between points and queries.

    Parameters
    ----------
    points : Array
        Array of shape (num_points, num_features).
    queries : Array
        Array of shape (num_queries, num_features).

    Returns
    -------
    Array
        Array of shape (num_queries, num_points) with squared L2 distances.
    """
    return jnp.sum((points[None, :, :] - queries[:, None, :]) ** 2, axis=2)


@jax.jit
def get_l2_distances(points: Array, queries: Array) -> Array:
    """
    Compute the L2 distances between points and queries.

    Parameters
    ----------
    points : Array
        Array of shape (num_points, num_features).
    queries : Array
        Array of shape (num_queries, num_features).

    Returns
    -------
    Array
        Array of shape (num_queries, num_points) with L2 distances.
    """
    return jnp.sqrt(get_ranking_distances(points, queries))


@jax.jit
def get_closest_point(points: Array, queries: Array) -> Array:
    """
    Find the closest point to each query.

    Parameters
    ----------
    points : Array
        Array of shape (num_points, num_features).
    queries : Array
        Array of shape (num_queries, num_features).

    Returns
    -------
    Array
        Array of shape (num_queries,) with indices of closest points.
    """
    distances = get_ranking_distances(points, queries)
    return jnp.argmin(distances, axis=1)


def get_closest_constrained_point(points: Array, queries: Array, constrains: Array) -> Array:
    """
    Find the closest point to each query subject to constraints.

    Parameters
    ----------
    points : Array
        Array of shape (num_points, num_features).
    queries : Array
        Array of shape (num_queries, num_features).
    constrains : Array
        Array of shape (num_queries,) with constraint values.

    Returns
    -------
    Array
        Array of shape (num_queries,) with indices of closest constrained points.

    Raises
    ------
    ValueError
        If no feasible cluster assignment exists for some data points.
    """
    distances = get_ranking_distances(points, queries)
    cluster_assignments = jnp.zeros(queries.shape[0], dtype=jnp.int32)
    for idx in range(queries.shape[0]):
        if jnp.all(distances[idx] == jnp.inf):
            msg = "No feasible cluster assignment for some data points."
            raise ValueError(msg)
        cluster_assignments = cluster_assignments.at[idx].set(jnp.argmin(distances[idx]))
        distances = distances.at[:, cluster_assignments[idx]].add(constrains[idx])
    return cluster_assignments


@jax.jit
def get_within_cluster_squared_distances(centroids: Array, x: Array, assignments: Array) -> Array:
    """
    Compute within-cluster squared distances for each data point.

    Parameters
    ----------
    centroids : Array
        Array of shape (num_clusters, num_features) with cluster centroids.
    x : Array
        Array of shape (num_samples, num_features) with data points.
    assignments : Array
        Array of shape (num_samples,) with cluster assignments.

    Returns
    -------
    Array
        Array of shape (num_samples,) with squared distances from each point to its centroid.
    """
    return jnp.sum((centroids[assignments] - x) ** 2, axis=1)


@jax.jit
def get_wcsd_per_cluster(centroids: Array, x: Array, assignments: Array) -> Array:
    """
    Compute mean within-cluster squared distances per cluster.

    Parameters
    ----------
    centroids : Array
        Array of shape (num_clusters, num_features) with cluster centroids.
    x : Array
        Array of shape (num_samples, num_features) with data points.
    assignments : Array
        Array of shape (num_samples,) with cluster assignments.

    Returns
    -------
    Array
        Array of shape (num_clusters, 1) with mean within-cluster squared distances per cluster.
    """
    wcsd = get_within_cluster_squared_distances(centroids, x, assignments)
    return (
        jax.ops.segment_sum(wcsd, assignments, num_segments=centroids.shape[0])
        / (
            jax.ops.segment_sum(
                jnp.ones_like(assignments), assignments, num_segments=centroids.shape[0]
            )[:, None]
        )
    )


@partial(jax.jit, static_argnames="n_clusters")
def get_centroids(x: Array, assignments: Array, n_clusters: int, random_key: Array) -> Array:
    """
    Compute cluster centroids from assignments.

    Parameters
    ----------
    x : Array
        Array of shape (num_samples, num_features) with data points.
    assignments : Array
        Array of shape (num_samples,) with cluster assignments.
    n_clusters : int
        Number of clusters.
    random_key : Array
        JAX random key for sampling.

    Returns
    -------
    Array
        Array of shape (n_clusters, num_features) with cluster centroids.
    """
    centroids = (
        jax.ops.segment_sum(x, assignments, num_segments=n_clusters)
        / (
            jax.ops.segment_sum(jnp.ones_like(assignments), assignments, num_segments=n_clusters)[
                :, None
            ]
        )
    )
    nan_mask = jnp.isnan(centroids).any(axis=1)
    sample = jax.random.choice(random_key, x.shape[0], (n_clusters,), replace=False)
    return jnp.where(
        nan_mask[:, None],
        x[sample, :],
        centroids,
    )  # pyright: ignore[reportAttributeAccessIssue])


@jax.jit
def get_wcss_from_assignments(centroids: Array, x: Array, assignments: Array) -> Array:
    """
    Compute within-cluster sum of squares from cluster assignments.

    Parameters
    ----------
    centroids : Array
        Array of shape (num_clusters, num_features) with cluster centroids.
    x : Array
        Array of shape (num_samples, num_features) with data points.
    assignments : Array
        Array of shape (num_samples,) with cluster assignments.

    Returns
    -------
    Array
        Scalar value representing the total within-cluster sum of squares.
    """
    wcsd = get_within_cluster_squared_distances(centroids, x, assignments)
    return jnp.sum(wcsd)


@jax.jit
def get_wcss(centroids: Array, x: Array) -> Array:
    """
    Compute within-cluster sum of squares.

    Parameters
    ----------
    centroids : Array
        Array of shape (num_clusters, num_features) with cluster centroids.
    x : Array
        Array of shape (num_samples, num_features) with data points.

    Returns
    -------
    Array
        Scalar value representing the total within-cluster sum of squares.
    """
    assignments = get_closest_point(centroids, x)
    return get_wcss_from_assignments(centroids, x, assignments)


class KMeans(BaseULAlgo):
    """
    K-Means clustering algorithm implementation using JAX.

    Attributes
    ----------
    centroids : Array
        Cluster centroids of shape (n_clusters, n_features).
    key : Array
        JAX random key for reproducibility.

    Parameters
    ----------
    key : Array | None, optional
        JAX random key for reproducibility. If None, uses a default key (42).
    """

    centroids: Array

    key: Array

    def __init__(self, key: Array | None = None) -> None:
        self.key = key if key is not None else jax.random.key(42)

    def _get_random_key(self) -> Array:
        """
        Generate a new random key by splitting the current key.

        Returns
        -------
        Array
            A new JAX random key derived from the current key.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def predict(self, x: Array) -> Array:
        """
        Predict cluster assignments for data points.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.

        Returns
        -------
        Array
            Array of shape (num_samples,) with cluster assignments.
        """
        return get_closest_point(self.centroids, x)

    def fit(self, x: Array, n_clusters: int, max_steps: int) -> list[float]:
        """
        Fit K-Means model to the data.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.
        n_clusters : int
            Number of clusters to form.
        max_steps : int
            Maximum number of iterations for the algorithm.

        Returns
        -------
        list[float]
            List of within-cluster sum of squares values at each iteration.
        """
        sample = jax.random.choice(self._get_random_key(), x.shape[0], (n_clusters,), replace=False)
        self.centroids = x[sample, :]
        loss_history = []
        closest_centroids = get_closest_point(self.centroids, x)
        latest_closest_centroids = closest_centroids
        loss_history.append(get_wcss_from_assignments(self.centroids, x, closest_centroids))
        for _ in range(max_steps):
            self.centroids = get_centroids(x, closest_centroids, n_clusters, self._get_random_key())
            closest_centroids = get_closest_point(self.centroids, x)
            loss_history.append(get_wcss_from_assignments(self.centroids, x, closest_centroids))
            if jnp.all(closest_centroids == latest_closest_centroids):
                break
            latest_closest_centroids = closest_centroids
        return loss_history

    def get_wcsd_per_cluster(self, x: Array) -> Array:
        """
        Get mean within-cluster sum of squared distances per cluster.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.

        Returns
        -------
        Array
            Array of shape (n_clusters, 1) with mean within-cluster squared distances per cluster.
        """
        assignments = get_closest_point(self.centroids, x)
        return get_wcsd_per_cluster(self.centroids, x, assignments)


class COPKMeans(KMeans):
    """
    Constrained K-Means clustering algorithm implementation using JAX.

    This class extends KMeans to support constraints on cluster assignments,
    implementing the COP-KMeans (Constrained K-Means) algorithm.
    """

    def constrained_predict(self, x: Array, constrains: Array) -> Array:
        """
        Predict cluster assignments subject to constraints.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.
        constrains : Array
            Array of shape (num_samples,) with constraint values.

        Returns
        -------
        Array
            Array of shape (num_samples,) with cluster assignments respecting constraints.
        """
        return get_closest_constrained_point(self.centroids, x, constrains)

    def fit(self, x: Array, n_clusters: int, max_steps: int, constrains: Array) -> list[float]:
        """
        Fit constrained K-Means model to the data.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.
        n_clusters : int
            Number of clusters to form.
        max_steps : int
            Maximum number of iterations for the algorithm.
        constrains : Array
            Array of shape (num_samples,) with binary constraint values (0 or 1).
            Non-zero values indicate cannot-link constraints.

        Returns
        -------
        list[float]
            List of within-cluster sum of squares values at each iteration.
        """
        constrains = jnp.where(constrains, jnp.inf, 0)
        sample = jax.random.choice(self._get_random_key(), x.shape[0], (n_clusters,), replace=False)
        self.centroids = x[sample, :]
        loss_history = []
        closest_centroids = get_closest_constrained_point(self.centroids, x, constrains)
        latest_closest_centroids = closest_centroids
        loss_history.append(get_wcss_from_assignments(self.centroids, x, closest_centroids))
        for _ in range(max_steps):
            self.centroids = get_centroids(x, closest_centroids, n_clusters, self._get_random_key())
            closest_centroids = get_closest_constrained_point(self.centroids, x, constrains)
            loss_history.append(get_wcss_from_assignments(self.centroids, x, closest_centroids))
            if jnp.all(closest_centroids == latest_closest_centroids):
                break
            latest_closest_centroids = closest_centroids
        return loss_history
