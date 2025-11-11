"""Test Models Module."""

from ml_algorithms.base.data_generation import generate_dataset
from ml_algorithms.base.data_visualization import (
    plot_loss_history,
    plot_supervised_learning_data_and_model,
    plot_unsupervised_learning_data_and_model,
)
from ml_algorithms.supervised_learning.linear_regression import LinearRegression
from ml_algorithms.unsupervised_learning.hierarchical_clustering.agglomerative_clustering import (
    HierarchicalAgglomerativeClustering,
)
from ml_algorithms.unsupervised_learning.hierarchical_clustering.divisive_clustering import (
    BisectingKMeans,
)
from ml_algorithms.unsupervised_learning.k_means import COPKMeans, KMeans

x, y = generate_dataset(
    dataset_style="linear_regression", data_structure_type="jax", n_features=1, n_samples=100
)
model = LinearRegression()
loss_history = model.fit(x, y, 0.01, 1000)

plot_supervised_learning_data_and_model(x, y, model)
plot_loss_history(loss_history, log_scale=True)

x, y = generate_dataset(
    dataset_style="binary_classification", data_structure_type="jax", n_features=1, n_samples=100
)

model = LinearRegression(logistic_regression=True)
loss_history = model.fit(x, y, 0.01, 10000)

plot_supervised_learning_data_and_model(x, y, model)
plot_loss_history(loss_history, log_scale=True)

x, y = generate_dataset(
    dataset_style="clustering", data_structure_type="jax", n_features=2, n_samples=100, n_clusters=3
)

model = KMeans()
loss_history = model.fit(x, 3, 100)

plot_unsupervised_learning_data_and_model(x, model)
plot_loss_history(loss_history, log_scale=False)

x, constrains = generate_dataset(
    dataset_style="constrained_clustering",
    data_structure_type="jax",
    n_features=2,
    n_samples=500,
    n_clusters=10,
)

model = COPKMeans()

loss_history = model.fit(x, 20, 100, constrains=constrains)

plot_unsupervised_learning_data_and_model(x, model)
plot_loss_history(loss_history, log_scale=False)

x, y = generate_dataset(
    dataset_style="clustering", data_structure_type="jax", n_features=2, n_samples=100, n_clusters=6
)

model = BisectingKMeans()

loss_history = model.fit(x, 6, 100)

plot_unsupervised_learning_data_and_model(x, model)
plot_loss_history(loss_history, log_scale=False)

x, y = generate_dataset(
    dataset_style="clustering", data_structure_type="jax", n_features=2, n_samples=100, n_clusters=6
)

model = HierarchicalAgglomerativeClustering()

loss_history = model.fit(x, 6)

plot_unsupervised_learning_data_and_model(x, model)
plot_loss_history(loss_history, log_scale=False)
