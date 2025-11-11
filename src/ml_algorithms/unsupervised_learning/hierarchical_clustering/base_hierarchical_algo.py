"""Base algorithm class."""

from abc import abstractmethod
from functools import total_ordering
from typing import TYPE_CHECKING, Any

from jax import numpy as jnp

from ml_algorithms.unsupervised_learning.base_ul_algo import BaseULAlgo

if TYPE_CHECKING:
    from jax import Array


@total_ordering
class HeapItem:
    """
    Helper class for heap operations.

    Attributes
    ----------
    priority : float
        Priority value for ordering in the heap.
    item : Any
        The actual item to store in the heap.
    """

    priority: float
    item: Any

    def __init__(self, priority: float, item: Any) -> None:
        self.priority = priority
        self.item = item

    def __lt__(self, other: HeapItem) -> bool:
        """Compare priority with another HeapItem for ordering."""
        return self.priority < other.priority

    def __eq__(self, other: HeapItem) -> bool:
        """Check equality with another HeapItem."""
        return self.priority == other.priority

    def __hash__(self) -> int:
        """Compute a hash value for the HeapItem."""
        return hash((self.priority, self.item))


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
    if node.node_id is not None:
        return jnp.full(x.shape[0], node.node_id)
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


def non_vec_search_leaf_node(node: ClusterNode, x: Array, max_id: int) -> int:
    """
    Non-vectorized recursive search to find the leaf node assignment for a single data point.

    Parameters
    ----------
    node : ClusterNode
        The root node of the cluster tree to search.
    x : Array
        Array of shape (num_features,) representing a single data point.
    max_id : int
        Maximum cluster ID to consider for leaf nodes.

    Returns
    -------
    int
        Cluster ID (leaf node assignment) for the data point.
    """
    if node.left is None or node.right is None:
        return node.node_id
    search_node = (
        node.left
        if jnp.sum((node.left.centroid - x) ** 2) < jnp.sum((node.right.centroid - x) ** 2)
        else node.right
    )

    if search_node.node_id > max_id:
        return node.node_id
    return non_vec_search_leaf_node(search_node, x, max_id)


class ClusterNode:
    """
    Represents a node in a divisive clustering tree.

    Attributes
    ----------
    data : Array
        Data points in this cluster of shape (num_samples, num_features).
    centroid : Array
        Centroid of the cluster of shape (num_features,).
    left : ClusterNode | None
        Left child node after division.
    right : ClusterNode | None
        Right child node after division.
    id : int | None
        Cluster ID assigned after fitting.
    """

    data: Array
    centroid: Array
    left: ClusterNode | None
    right: ClusterNode | None
    node_id: int

    def __init__(self, centroid: Array, data: Array, node_id: int) -> None:
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
        self.left = None
        self.right = None
        self.node_id = node_id

    def is_leaf(self) -> bool:
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node has not been divided yet, False otherwise.
        """
        return self.left is None or self.right is None


class BaseHierarchicalAlgo(BaseULAlgo):
    """Abstract base class for hierarchical clustering algorithms."""

    root: ClusterNode
    centroids: Array

    @abstractmethod
    def fit(self, x: Any) -> Any:
        """
        Fit the model to the data.

        Parameters
        ----------
        x : Any
            Input features.
        y : Any
            Target values.

        Returns
        -------
        Any
            Loss value after fitting.
        """

    @abstractmethod
    def predict(self, x: Any) -> Any:
        """
        Make predictions using the fitted model.

        Parameters
        ----------
        x : Any
            Input features.

        Returns
        -------
        Any
            Predicted values.
        """

    def transform_list(self, data: list[Any]) -> Array:
        """
        Convert a Python list to a JAX array.

        Parameters
        ----------
        data : list[Any]
            Python list to convert.

        Returns
        -------
        Array
            JAX array representation of the input list.
        """
        return jnp.array(data)
