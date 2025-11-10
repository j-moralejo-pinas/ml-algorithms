"""Divisive clustering algorithm implementation using JAX and K-Means."""

import heapq

from jax import Array
from jax import numpy as jnp

from ml_algorithms.unsupervised_learning.k_means import KMeans


# @jax.jit
def get_single_cluster_wcss(centroid: Array, data: Array) -> float:
    """Compute the within-cluster sum of squares (WCSS) for a single cluster."""
    return float(jnp.sum((data - centroid[None, :]) ** 2))


class ClusterNode:
    """
    Represents a node in a divisive clustering tree.

    Attributes
    ----------
    data : Array
        Data points in this cluster of shape (num_samples, num_features).
    centroid : Array
        Centroid of the cluster of shape (num_features,).
    wcss : float
        Within-cluster sum of squares for this cluster.
    left : ClusterNode | None
        Left child node after division.
    right : ClusterNode | None
        Right child node after division.
    id : int | None
        Cluster ID assigned after fitting.
    """

    data: Array
    centroid: Array
    wcss: float
    left: ClusterNode | None
    right: ClusterNode | None
    id: int | None

    def __init__(self, centroid: Array, data: Array) -> None:
        """
        Initialize a cluster node.

        Parameters
        ----------
        centroid : Array
            The centroid of the cluster.
        data : Array
            The data points in the cluster.
        """
        self.data = data
        self.centroid = centroid
        self.wcss = get_single_cluster_wcss(centroid, data)
        self.left = None
        self.right = None
        self.id = None

    def divide(
        self, max_steps: int
    ) -> tuple[tuple[float, ClusterNode], tuple[float, ClusterNode]]:
        """
        Divide this cluster into two child clusters using K-Means.

        Parameters
        ----------
        max_steps : int
            Maximum number of iterations for K-Means fitting.

        Returns
        -------
        tuple[tuple[float, ClusterNode], tuple[float, ClusterNode]]
            Two tuples containing (wcss, ClusterNode) for left and right clusters.
        """
        kmeans = KMeans()
        kmeans.fit(self.data, n_clusters=2, max_steps=max_steps)
        assignments = kmeans.predict(self.data)
        left_data = self.data[assignments == 0]
        right_data = self.data[assignments == 1]
        self.left = ClusterNode(centroid=kmeans.centroids[0], data=left_data)
        self.right = ClusterNode(centroid=kmeans.centroids[1], data=right_data)
        return (self.left.wcss, self.left), (self.right.wcss, self.right)

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node has not been divided yet, False otherwise.
        """
        return self.left is None or self.right is None

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

def merge_classes(mask_0: Array, classes_0: Array, mask_1: Array, classes_1: Array) -> Array:
    """
    Merge two sets of class assignments based on masks.

    Parameters
    ----------
    mask_0 : Array
        Boolean mask for the first set of samples.
    classes_0 : Array
        Class assignments for samples where mask_0 is True.
    mask_1 : Array
        Boolean mask for the second set of samples.
    classes_1 : Array
        Class assignments for samples where mask_1 is True.

    Returns
    -------
    Array
        Merged class assignments for all samples.
    """
    out = jnp.empty(mask_0.shape[0], dtype=classes_0.dtype)
    out = out.at[jnp.where(mask_0, size=classes_0.size)].set(classes_0)
    return out.at[jnp.where(mask_1, size=classes_1.size)].set(classes_1)

def search_leaf_node(node: ClusterNode, x: Array) -> Array:
    """
    Recursively search the cluster tree to find leaf node assignments.

    Parameters
    ----------
    node : ClusterNode
        The root node of the cluster tree to search.
    x : Array
        Array of shape (num_samples, num_features) with data points to classify.

    Returns
    -------
    Array
        Array of shape (num_samples,) with cluster IDs (leaf node assignments).
    """
    if node.id is not None:
        return jnp.full(x.shape[0], node.id)
    assert node.left is not None
    assert node.right is not None
    centroids = jnp.stack([node.left.centroid, node.right.centroid])
    closest_centroid = get_closest_point(centroids, x)

    mask_0 = closest_centroid == 0
    mask_1 = closest_centroid == 1

    left_samples = x[mask_0]
    right_samples = x[mask_1]

    new_classes_0 = search_leaf_node(node.left, left_samples)
    new_classes_1 = search_leaf_node(node.right, right_samples)

    return merge_classes(mask_0, new_classes_0, mask_1, new_classes_1)

class BisectingKMeans(KMeans):
    """
    Bisecting K-Means clustering algorithm implementation using JAX.

    This class implements the divisive hierarchical clustering algorithm,
    which recursively bisects clusters using K-Means until the desired
    number of clusters is reached.

    Attributes
    ----------
    root : ClusterNode
        Root node of the clustering tree.
    centroids : Array
        Final cluster centroids of shape (n_clusters, n_features).
    """

    root: ClusterNode
    centroids: Array

    def fit(self, x: Array, n_clusters: int, max_steps: int) -> list[float]:
        """
        Fit Bisecting K-Means model to the data.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.
        n_clusters : int
            Number of clusters to form.
        max_steps : int
            Maximum number of iterations for K-Means fitting in each bisection.

        Returns
        -------
        list[float]
            List of within-cluster sum of squares values at each bisection step.
        """
        centroid = jnp.mean(x, axis=0)
        self.root = ClusterNode(centroid=centroid, data=x)
        loss_history = [self.root.wcss]
        nodes = [(self.root.wcss, self.root)]
        while len(nodes) < n_clusters:
            node_to_divide = heapq.heappop_max(nodes)[1]
            left, right = node_to_divide.divide(max_steps=max_steps)
            heapq.heappush_max(nodes, left)
            heapq.heappush_max(nodes, right)

            loss_history.append(loss_history[-1] - node_to_divide.wcss + left[0] + right[0])

        id_counter = 0
        centroids = []
        while nodes:
            _, node = heapq.heappop_max(nodes)
            node.id = id_counter
            id_counter += 1
            centroids.append(node.centroid)

        self.centroids = jnp.stack(centroids)

        return loss_history

    def predict(self, x: Array) -> Array:
        """
        Predict cluster assignments for data points using the fitted tree.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.

        Returns
        -------
        Array
            Array of shape (num_samples,) with cluster assignments.
        """
        return search_leaf_node(self.root, x)

