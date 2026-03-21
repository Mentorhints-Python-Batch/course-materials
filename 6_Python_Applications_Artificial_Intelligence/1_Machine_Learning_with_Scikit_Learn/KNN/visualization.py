"""Visualization utilities for the KNN T-Shirt Size demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def plot_class_distribution(y, ax=None):
    """Bar chart showing the count of each T-shirt size class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    colors = sns.color_palette("Set2", n_colors=len(counts))
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("T-Shirt Size")
    ax.set_ylabel("Count")
    ax.set_title("T-Shirt Size Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return ax


def plot_scatter(X, y, feature_names=None, ax=None):
    """2-D scatter plot of the two features coloured by class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    if feature_names is None:
        feature_names = ["Feature 1", "Feature 2"]

    X_arr = np.asarray(X)
    for label, colour, marker in [("L", "tab:blue", "o"), ("M", "tab:orange", "s")]:
        mask = np.asarray(y) == label
        if not mask.any():
            continue
        ax.scatter(X_arr[mask, 0], X_arr[mask, 1],
                   label=label, c=colour, marker=marker, edgecolors="k", s=80)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Height vs Weight by T-Shirt Size")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_decision_boundary(model, X, y, title="Decision Boundary", ax=None,
                           h=0.2):
    """Plot the KNN decision boundary on a 2-D feature space.

    Parameters
    ----------
    model : fitted classifier
    X     : array-like of shape (n_samples, 2)
    y     : integer-encoded target labels
    h     : mesh step size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    X = np.asarray(X)
    y = np.asarray(y)

    cmap_light = ListedColormap(["#AADDFF", "#FFDDAA"])
    cmap_bold = ListedColormap(["#1f77b4", "#ff7f0e"])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
               edgecolors="k", s=60)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title(title)
    ax.set_xlabel("Feature 1 (scaled)")
    ax.set_ylabel("Feature 2 (scaled)")
    plt.tight_layout()
    return ax


def plot_k_vs_accuracy(k_scores, ax=None):
    """Line plot of K values vs cross-validation accuracy."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(k_scores.keys())
    accs = list(k_scores.values())
    ax.plot(ks, accs, "o-", color="tab:blue", linewidth=2, markersize=7)
    best_k = max(k_scores, key=k_scores.get)
    ax.axvline(best_k, color="tab:red", linestyle="--", alpha=0.7,
               label=f"Best K={best_k}")
    ax.set_xlabel("K (number of neighbours)")
    ax.set_ylabel("Mean CV Accuracy")
    ax.set_title("Effect of K on Cross-Validation Accuracy")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_confusion_matrix(cm, class_names=None, ax=None):
    """Heatmap of a confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return ax


def plot_metric_comparison(results_dict, ax=None):
    """Bar chart comparing accuracy across different distance metrics or settings.

    Parameters
    ----------
    results_dict : dict mapping label -> accuracy (float)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(results_dict.keys())
    accuracies = list(results_dict.values())
    colors = sns.color_palette("Set2", n_colors=len(labels))

    bars = ax.bar(labels, accuracies, color=colors)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparison")
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    return ax
