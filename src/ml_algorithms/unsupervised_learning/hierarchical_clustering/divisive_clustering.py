"""Divisive clustering algorithm implementation using JAX and K-Means."""

import heapq

from jax import Array
from jax import numpy as jnp

from ml_algorithms.unsupervised_learning.hierarchical_clustering.base_hierarchical_algo import (
    ClusterNode,
    HeapItem,
    non_vec_search_leaf_node,
)
from ml_algorithms.unsupervised_learning.k_means import KMeans


# @jax.jit
def get_single_cluster_wcss(centroid: Array, data: Array) -> float:
    """Compute the within-cluster sum of squares (WCSS) for a single cluster."""
    return float(jnp.sum((data - centroid[None, :]) ** 2))


class DivisiveClusterNode(ClusterNode):
    """
    Represents a node in a divisive clustering tree.

    Attributes
    ----------
    wcss : float
        Within-cluster sum of squares for this cluster.
    """

    wcss: float

    def __init__(self, centroid: Array, data: Array, node_id: int) -> None:
        """
        Initialize a cluster node.

        Parameters
        ----------
        centroid : Array
            The centroid of the cluster.
        data : Array
            The data points in the cluster.
        node_id : int
            The cluster ID.
        """
        self.wcss = get_single_cluster_wcss(centroid, data)
        super().__init__(centroid=centroid, data=data, node_id=node_id)

    def divide(
        self, max_steps: int, last_id: int = 0
    ) -> tuple[DivisiveClusterNode, DivisiveClusterNode]:
        """
        Divide this cluster into two child clusters using K-Means.

        Parameters
        ----------
        max_steps : int
            Maximum number of iterations for K-Means fitting.
        last_id : int
            Last used cluster ID for assigning IDs to child clusters.

        Returns
        -------
        DivisiveClusterNode
            Left child DivisiveClusterNode
        DivisiveClusterNode
            Right child DivisiveClusterNode
        """
        kmeans = KMeans()
        kmeans.fit(self.data, n_clusters=2, max_steps=max_steps)
        assignments = kmeans.predict(self.data)
        left_data = self.data[assignments == 0]
        right_data = self.data[assignments == 1]
        self.left = DivisiveClusterNode(
            centroid=kmeans.centroids[0], data=left_data, node_id=last_id + 1
        )
        self.right = DivisiveClusterNode(
            centroid=kmeans.centroids[1], data=right_data, node_id=last_id + 2
        )
        return self.left, self.right


class BisectingKMeans(KMeans):
    """
    Bisecting K-Means clustering algorithm implementation using JAX.

    This class implements the divisive hierarchical clustering algorithm,
    which recursively bisects clusters using K-Means until the desired
    number of clusters is reached.

    Attributes
    ----------
    _n_clusters : int
        Number of clusters to form.
    """

    _n_clusters: int

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
        self._n_clusters = n_clusters
        last_id = 0
        self.root = DivisiveClusterNode(centroid=jnp.mean(x, axis=0), data=x, node_id=last_id)
        loss_history = [self.root.wcss]
        nodes = [HeapItem(self.root.wcss, self.root)]
        centroids = [self.root.centroid]
        while len(nodes) < self._n_clusters:
            node_to_divide: DivisiveClusterNode = heapq.heappop_max(nodes).item
            left, right = node_to_divide.divide(max_steps=max_steps, last_id=last_id)
            last_id = right.node_id
            heapq.heappush_max(nodes, HeapItem(left.wcss, left))
            heapq.heappush_max(nodes, HeapItem(right.wcss, right))
            centroids.append(left.centroid)
            centroids.append(right.centroid)
            loss_history.append(loss_history[-1] - node_to_divide.wcss + left.wcss + right.wcss)

        self.centroids = jnp.array(centroids)

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
        return jnp.array(
            [
                non_vec_search_leaf_node(self.root, instance, 2 * (self._n_clusters - 1))
                for instance in x
            ]
        )
