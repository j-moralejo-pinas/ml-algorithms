"""Base algorithm class."""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from ml_algorithms.base.base_algo import BaseAlgo

if TYPE_CHECKING:
    from jax import Array


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

    @abstractmethod
    def transform_list(self, data: list[Any]) -> Any:
        """
        Transform a list of data points to the corresponding data structure.

        Parameters
        ----------
        data : list[Any]
            List of data points to transform.

        Returns
        -------
        Any
            Transformed data in the desired data structure.
        """
