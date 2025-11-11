"""Base algorithm class."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from jax import numpy as jnp

from ml_algorithms.base.base_algo import BaseAlgo

if TYPE_CHECKING:
    from jax import Array


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


class BaseULAlgo(BaseAlgo):
    """Abstract base class for unsupervised learning algorithms."""

    centroids: Array

    @abstractmethod
    def fit(self, x: Any) -> Any:
        """
        Fit the model to the data.

        Parameters
        ----------
        x : Any
            Input features.

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
