"""Data visualization utilities for machine learning algorithms."""

from math import log
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ml_algorithms.supervised_learning.base_sl_algo import BaseSLAlgo
    from ml_algorithms.unsupervised_learning.base_ul_algo import BaseULAlgo


def _darken_color(color: tuple, factor: float = 0.7) -> tuple:
    """Darken a color by reducing its RGB values by a factor."""
    return tuple(c * factor for c in color[:3])


def plot_supervised_learning_data_and_model(
    x: Any,
    y: Any,
    model: BaseSLAlgo,
    title: str = "Data and Model Prediction",
) -> None:
    """
    Plot supervised learning data and model predictions.

    Parameters
    ----------
    x : Any
        Input features.
    y : Any
        Target values.
    model : BaseSLAlgo
        Fitted machine learning model.
    title : str, optional
        Title of the plot, by default "Data and Model Prediction".
    """
    plt.scatter(x, y, color="b", label="Data Points")

    steps = 100

    x = [float(xi[0]) for xi in x]
    y = [float(yi) for yi in y]

    min_x = min(x)
    x_diff = max(x) - min(x)
    steps_minus_1 = 1 / (steps - 1)
    x_range = [min_x + (x_diff * i * steps_minus_1) for i in range(steps)]

    y_pred = model.predict(model.transform_list([[jx] for jx in x_range]))

    plt.plot(x_range, y_pred, color="r", label="Model Prediction", linewidth=1)
    plt.ylabel("Target")
    plt.title(title)
    plt.show()


def plot_unsupervised_learning_data_and_model(
    x: Any,
    model: BaseULAlgo,
    title: str = "Data and Model Prediction",
) -> None:
    """
    Plot unsupervised learning data and model predictions.

    Parameters
    ----------
    x : Any
        Input features.
    model : BaseULAlgo
        Fitted machine learning model.
    title : str, optional
        Title of the plot, by default "Data and Model Prediction".
    """
    cluster_assignments = model.predict(x)

    centroids = model.centroids

    # Get unique cluster IDs
    unique_clusters = list({int(c) for c in cluster_assignments})

    # Create a color palette based on the number of centroids
    num_clusters = len(unique_clusters)
    if num_clusters <= 10:  # noqa: PLR2004
        colors = plt.cm.tab10([(i / num_clusters) for i in range(num_clusters)])  # pyright: ignore[reportAttributeAccessIssue]
    elif num_clusters <= 20:  # noqa: PLR2004
        colors = plt.cm.tab20([(i / num_clusters) for i in range(num_clusters)])  # pyright: ignore[reportAttributeAccessIssue]
    else:
        colors = plt.cm.hsv([(i / num_clusters) for i in range(num_clusters)])  # pyright: ignore[reportAttributeAccessIssue]

    # Create a mapping from cluster_id to color
    cluster_color_map = {cluster_id: colors[idx] for idx, cluster_id in enumerate(unique_clusters)}

    # Plot each cluster with a different color
    for cluster_id in unique_clusters:
        cluster_points = [x[i] for i in range(len(x)) if cluster_assignments[i] == cluster_id]
        if cluster_points:
            xs = [point[0] for point in cluster_points]
            ys = [point[1] for point in cluster_points]
            plt.scatter(xs, ys, color=cluster_color_map[cluster_id], label=f"Cluster {cluster_id}")

    # Plot centroids with darker shades of their cluster colors
    for cluster in unique_clusters:
        darker_color = _darken_color(cluster_color_map[cluster])
        plt.scatter(
            centroids[cluster][0], centroids[cluster][1], color=darker_color, marker="X", s=200
        )

    plt.title(title)
    plt.show()


def plot_loss_history(loss_history: list[float], *, log_scale: bool = False) -> None:
    """
    Plot the loss history over epochs.

    Parameters
    ----------
    loss_history : list[float]
        List of loss values over epochs.
    log_scale : bool, optional
        Whether to use logarithmic scale for the loss values, by default False.
    """
    if log_scale:
        loss_history = [log(float(lh)) for lh in loss_history]

    plt.plot(range(len(loss_history)), loss_history, color="g", label="Loss History", linewidth=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss History Over Epochs")
    plt.show()
