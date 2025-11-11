"""Agglomerative Clustering Module."""

from heapq import heapify, heappop, heappush

from jax import Array
from jax import numpy as jnp

from ml_algorithms.unsupervised_learning.hierarchical_clustering.base_hierarchical_algo import (
    BaseULAlgo,
    ClusterNode,
    HeapItem,
    non_vec_search_leaf_node,
)


def calculate_ward_merge_cost(c_a: Array, c_b: Array, n_a: Array, n_b: Array) -> Array:
    """
    Calculate the Ward's merge cost between two clusters.

    Parameters
    ----------
    c_a : Array
        Centroid of cluster A of shape (num_features,).
    c_b : Array
        Centroid of cluster B of shape (num_features,).
    n_a : Array
        Number of instances in cluster A.
    n_b : Array
        Number of instances in cluster B.

    Returns
    -------
    Array
        Ward's merge cost.
    """
    return ((n_a * n_b) / (n_a + n_b)) * jnp.sum((c_a - c_b) ** 2)


class AgglomerativeClusterNode(ClusterNode):
    """
    Represents a node in an agglomerative clustering tree.

    Attributes
    ----------
    sum_of_instances : Array
        Sum of all data points in this cluster.
    """

    sum_of_instances: Array

    def __init__(self, centroid: Array, data: Array, sum_of_instances: Array, node_id: int) -> None:
        self.sum_of_instances = sum_of_instances
        super().__init__(centroid=centroid, data=data, node_id=node_id)


class HierarchicalAgglomerativeClustering(BaseULAlgo):
    """
    Hierarchical Agglomerative Clustering (HAC) algorithm implementation using JAX.

    This class implements the HAC algorithm, which builds a hierarchy of clusters
    by iteratively merging the closest pairs of clusters until a stopping criterion is met.

    Attributes
    ----------
    _n_clusters : int
        Number of clusters to form.
    """

    _n_clusters: int

    def fit(self, x: Array, n_clusters: int) -> list[float]:
        """
        Fit the Hierarchical Agglomerative Clustering model to the data.

        Parameters
        ----------
        x : Array
            Array of shape (num_samples, num_features) with data points.
        n_clusters : int
            Desired number of clusters.
        """
        self._n_clusters = n_clusters

        cluster_list: list[AgglomerativeClusterNode] = []

        cluster_set = {
            AgglomerativeClusterNode(
                centroid=x[i],
                data=jnp.expand_dims(x[i], 0),
                sum_of_instances=x[i],
                node_id=-1,
            )
            for i in range(len(x))
        }

        cluster_distances = [
            HeapItem(float(jnp.sum((ci.sum_of_instances - cj.sum_of_instances) ** 2) / 2), (ci, cj))
            for ci in cluster_set
            for cj in cluster_set
            if ci != cj
        ]

        heapify(cluster_distances)

        loss_history = [0.0]

        while len(cluster_set) > 1:
            heap_item = heappop(cluster_distances)
            ward_cost = heap_item.priority
            ci: AgglomerativeClusterNode = heap_item.item[0]
            cj: AgglomerativeClusterNode = heap_item.item[1]
            if ci in cluster_set and cj in cluster_set:
                sum_of_instances = ci.sum_of_instances + cj.sum_of_instances
                num_instances = len(ci.data) + len(cj.data)
                new_cluster = AgglomerativeClusterNode(
                    centroid=sum_of_instances / num_instances,
                    data=jnp.concat((ci.data, cj.data)),
                    sum_of_instances=sum_of_instances,
                    node_id=-1,
                )

                new_cluster.left = ci
                new_cluster.right = cj

                cluster_set.remove(ci)
                cluster_set.remove(cj)

                for ck in cluster_set:
                    heappush(
                        cluster_distances,
                        HeapItem(
                            float(
                                calculate_ward_merge_cost(
                                    new_cluster.centroid,
                                    ck.centroid,
                                    jnp.array([len(new_cluster.data)]),
                                    jnp.array([len(ck.data)]),
                                )[0]
                            ),
                            (new_cluster, ck),
                        ),
                    )

                cluster_list.append(ci)
                cluster_list.append(cj)
                cluster_set.add(new_cluster)
                loss_history.append(loss_history[-1] + ward_cost)

        self.root = cluster_set.pop()
        self.root.node_id = 0
        centroids = [self.root.centroid]
        for idx, cluster in enumerate(reversed(cluster_list)):
            centroids.append(cluster.centroid)
            cluster.node_id = idx + 1

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
