"""Visualization utilities for the Decision Trees demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


def plot_class_distribution(y, ax=None):
    """Bar chart showing the balance between target classes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts().sort_index()
    colors = sns.color_palette("Set2", n_colors=len(counts))
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Target Class Distribution")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
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
    combined["target"] = y.values

    for i, feat in enumerate(features):
        sns.boxplot(data=combined, x="target", y=feat, ax=axes[i],
                    palette="Set2")
        axes[i].set_title(feat)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Class", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(X, ax=None):
    """Correlation heatmap for the feature matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    corr = X.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return ax


def plot_confusion_matrix(cm, ax=None):
    """Heatmap of a confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return ax


def plot_feature_importance(model, feature_names, ax=None):
    """Horizontal bar chart of feature importances from a trained tree."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    colors = sns.color_palette("viridis", n_colors=len(indices))

    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    plt.tight_layout()
    return ax


def plot_decision_tree(model, feature_names, class_names=None,
                       max_depth=None, figsize=(20, 10)):
    """Render the decision tree using sklearn's plot_tree."""
    if class_names is None:
        class_names = ["Class 0", "Class 1"]
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, max_depth=max_depth,
              fontsize=9)
    ax.set_title("Decision Tree", fontsize=14)
    plt.tight_layout()
    return fig
