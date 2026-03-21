"""Visualization utilities for the SVM Wine Quality demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def plot_class_distribution(y, ax=None):
    """Bar chart showing the count of each quality score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    counts = y.value_counts().sort_index()
    colors = sns.color_palette("Set2", n_colors=len(counts))
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    ax.set_title("Wine Quality Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return ax


def plot_feature_distributions(X, y, cols_per_row=3):
    """Box-plots of each feature grouped by target class."""
    features = X.columns.tolist()
    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(5 * cols_per_row, 4 * n_rows))
    axes = axes.flatten()

    combined = X.copy()
    combined["quality"] = y.values

    for i, feat in enumerate(features):
        sns.boxplot(data=combined, x="quality", y=feat, ax=axes[i],
                    palette="Set2")
        axes[i].set_title(feat)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Quality", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(X, ax=None):
    """Correlation heatmap for the feature matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    corr = X.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return ax


def plot_confusion_matrix(cm, class_names=None, ax=None):
    """Heatmap of a confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return ax


def plot_decision_boundary(model, X_2d, y, title="Decision Boundary",
                           ax=None):
    """Plot the decision boundary of a classifier on 2-D data.

    Parameters
    ----------
    model : fitted classifier (must support predict)
    X_2d  : array-like of shape (n_samples, 2) — e.g. PCA-reduced features
    y     : target labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    h = 0.3
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="coolwarm",
                         edgecolors="k", s=20, alpha=0.7)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)
    plt.tight_layout()
    return ax


def plot_kernel_comparison(results_dict, ax=None):
    """Bar chart comparing accuracy across different SVM kernels.

    Parameters
    ----------
    results_dict : dict mapping kernel name -> accuracy (float)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    kernels = list(results_dict.keys())
    accuracies = list(results_dict.values())
    colors = sns.color_palette("Set2", n_colors=len(kernels))

    bars = ax.bar(kernels, accuracies, color=colors)
    ax.set_xlabel("Kernel")
    ax.set_ylabel("Accuracy")
    ax.set_title("SVM Kernel Comparison")
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    return ax
