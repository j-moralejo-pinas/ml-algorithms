"""Linear Regression Algorithm Implementation."""

import jax
from jax import Array
from jax import numpy as jnp

from ml_algorithms.supervised_learning.base_sl_algo import BaseSLAlgo


def compute_output(coefficients: Array, intercept: Array, x: Array) -> Array:
    """
    Compute the linear output given coefficients, intercept, and input features.

    Parameters
    ----------
    coefficients : Array
        Coefficients of the linear model.
    intercept : Array
        Intercept of the linear model.
    x : Array
        Input features.

    Returns
    -------
    Array
        Computed linear output.
    """
    return x @ coefficients + intercept


def predict_positive(coefficients: Array, intercept: Array, x: Array) -> Array:
    """
    Predict probabilities using the logistic function.

    Parameters
    ----------
    coefficients : Array
        Coefficients of the logistic model.
    intercept : Array
        Intercept of the logistic model.
    x : Array
        Input features.

    Returns
    -------
    Array
        Predicted probabilities.
    """
    return 1 / (1 + jnp.exp(-compute_output(coefficients=coefficients, intercept=intercept, x=x)))


@jax.jit
def regression_gradient_descent_step(
    coefficients: Array,
    intercept: Array,
    x: Array,
    y: Array,
    lr: float,
) -> tuple[Array, Array, Array]:
    """
    Perform a single gradient descent step for linear regression.

    Parameters
    ----------
    coefficients : Array
        Current coefficients of the linear model.
    intercept : Array
        Current intercept of the linear model.
    x : Array
        Input features.
    y : Array
        Target values.
    lr : float
        Learning rate.

    Returns
    -------
    Array
        Updated coefficients.
    Array
        Updated intercept.
    Array
        Computed loss.
    """
    y_pred = compute_output(coefficients=coefficients, intercept=intercept, x=x)
    sample_dim = x.shape[0]
    neg_y_diff = y - y_pred
    neg_coeff_grad = (x.T @ neg_y_diff) / sample_dim
    neg_intercept_grad = jnp.sum(neg_y_diff) / sample_dim
    loss = jnp.pow(neg_y_diff, 2)
    return coefficients + lr * neg_coeff_grad, intercept + lr * neg_intercept_grad, jnp.mean(loss)


@jax.jit
def classification_gradient_descent_step(
    coefficients: Array,
    intercept: Array,
    x: Array,
    y: Array,
    lr: float,
) -> tuple[Array, Array, Array]:
    """
    Perform a single gradient descent step for logistic regression.

    Parameters
    ----------
    coefficients : Array
        Current coefficients of the logistic model.
    intercept : Array
        Current intercept of the logistic model.
    x : Array
        Input features.
    y : Array
        Target values.
    lr : float
        Learning rate.

    Returns
    -------
    Array
        Updated coefficients.
    Array
        Updated intercept.
    Array
        Computed loss.
    """
    y_pred = predict_positive(coefficients=coefficients, intercept=intercept, x=x)
    sample_dim = x.shape[0]
    neg_y_diff = y - y_pred
    neg_coeff_grad = (x.T @ neg_y_diff) / sample_dim
    neg_intercept_grad = jnp.sum(neg_y_diff) / sample_dim
    loss = -(y * jnp.log(y_pred + 1e-15) + (1 - y) * jnp.log(1 - y_pred + 1e-15))
    return coefficients + lr * neg_coeff_grad, intercept + lr * neg_intercept_grad, loss


def gradient_descent(
    coefficients: Array,
    intercept: Array,
    x: Array,
    y: Array,
    lr: float,
    epochs: int,
    *,
    logistic_regression: bool = False,
) -> tuple[Array, Array, list[float]]:
    """
    Perform gradient descent optimization.

    Parameters
    ----------
    coefficients : Array
        Initial coefficients of the model.
    intercept : Array
        Initial intercept of the model.
    x : Array
        Input features.
    y : Array
        Target values.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    logistic_regression : bool
        Whether to use logistic regression.

    Returns
    -------
    Array
        Optimized coefficients.
    Array
        Optimized intercept.
    list[float]
        History of loss values over epochs.
    """
    loss_history = []
    for _ in range(epochs):
        if logistic_regression:
            coefficients, intercept, loss = classification_gradient_descent_step(
                coefficients=coefficients,
                intercept=intercept,
                x=x,
                y=y,
                lr=lr,
            )
            loss_history.append(float(loss[0]))
        else:
            coefficients, intercept, loss = regression_gradient_descent_step(
                coefficients=coefficients,
                intercept=intercept,
                x=x,
                y=y,
                lr=lr,
            )
            loss_history.append(float(loss))
    return coefficients, intercept, loss_history


class LinearRegression(BaseSLAlgo):
    """
    Linear Regression and Logistic Regression model.

    Attributes
    ----------
    coefficients : Array
        Coefficients of the model.
    intercept : Array
        Intercept of the model.
    key : Array
        Random key for initialization.
    logistic_regression : bool
        Whether to use logistic regression.
    """

    coefficients: Array
    intercept: Array
    key: Array
    logistic_regression: bool

    def __init__(self, key: Array | None = None, *, logistic_regression: bool = False) -> None:
        self.key = key if key is not None else jax.random.key(42)
        self.logistic_regression = logistic_regression

    def _get_random_key(self) -> Array:
        """
        Get a random key for initialization.

        Returns
        -------
        Array
            A random key for initialization.
        """
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def fit(self, x: Array, y: Array, lr: float, epochs: int) -> list[float]:
        """
        Fit the Linear Regression or Logistic Regression model.

        Parameters
        ----------
        x : Array
            Input features.
        y : Array
            Target values.
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.

        Returns
        -------
        list[float]
            History of loss values over epochs.
        """
        self.coefficients = jax.random.normal(key=self._get_random_key(), shape=(x.shape[1],))
        self.intercept = jax.random.normal(key=self._get_random_key())
        self.coefficients, self.intercept, loss_history = gradient_descent(
            coefficients=self.coefficients,
            intercept=self.intercept,
            x=x,
            y=y,
            lr=lr,
            epochs=epochs,
            logistic_regression=self.logistic_regression,
        )
        return loss_history

    def predict(self, x: Array) -> Array:
        """
        Predict using the fitted Linear Regression or Logistic Regression model.

        Parameters
        ----------
        x : Array
            Input features.

        Returns
        -------
        Array
            Predicted values.
        """
        if self.logistic_regression:
            return predict_positive(coefficients=self.coefficients, intercept=self.intercept, x=x)
        return compute_output(coefficients=self.coefficients, intercept=self.intercept, x=x)

    def transform_list(self, data: list[Array]) -> Array:
        """
        Transform a list of arrays into a single JAX array.

        Parameters
        ----------
        data : list[Array]
            List of arrays to transform.

        Returns
        -------
        Array
            Transformed JAX array.
        """
        return jnp.array(data)
