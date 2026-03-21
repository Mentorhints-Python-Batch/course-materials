"""Visualization utilities for the ANN Car Purchase demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(y, ax=None):
    """Histogram of car purchase amounts."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(y, bins=30, kde=True, ax=ax, color=sns.color_palette("Set2")[0])
    ax.set_xlabel("Car Purchase Amount ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Car Purchase Amounts", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


def plot_feature_distributions(df, features, cols_per_row=3):
    """Histograms / KDE for each input feature."""
    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(5 * cols_per_row, 4 * n_rows))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", n_features)

    for i, feat in enumerate(features):
        sns.histplot(df[feat], bins=30, kde=True, ax=axes[i], color=palette[i])
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, ax=None):
    """Correlation heatmap for all numeric columns."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, square=True)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


def plot_feature_vs_target(df, features, target, cols_per_row=3):
    """Scatter plots of each feature against the target variable."""
    n_features = len(features)
    n_rows = (n_features + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(5 * cols_per_row, 4 * n_rows))
    axes = axes.flatten()
    palette = sns.color_palette("Set2", n_features)

    for i, feat in enumerate(features):
        axes[i].scatter(df[feat], df[target], alpha=0.5, s=15, color=palette[i])
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel(target)
        axes[i].set_title(f"{feat} vs {target}", fontsize=11, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Features vs Target", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot train and validation loss curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, history["train_loss"], "o-", label="Train Loss",
            linewidth=2, markersize=3)
    ax.plot(epochs, history["val_loss"], "s-", label="Validation Loss",
            linewidth=2, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lr_schedule(history):
    """Plot the learning rate across epochs."""
    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, history["lr"], "D-", color="tab:green", linewidth=2,
            markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_predictions_vs_actual(y_true, y_pred, ax=None):
    """Scatter plot of predicted vs actual values with an ideal y=x line."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20, color=sns.color_palette("Set2")[1],
               edgecolors="k", linewidths=0.3)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1.5, label="Ideal (y = x)")
    ax.set_xlabel("Actual ($)")
    ax.set_ylabel("Predicted ($)")
    ax.set_title("Predicted vs Actual", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_residuals(y_true, y_pred):
    """Residual distribution and residuals vs predicted value."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(residuals, bins=30, kde=True, ax=ax1,
                 color=sns.color_palette("Set2")[2])
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Residual ($)")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual Distribution", fontsize=13, fontweight="bold")

    ax2.scatter(y_pred, residuals, alpha=0.6, s=20,
                color=sns.color_palette("Set2")[3], edgecolors="k", linewidths=0.3)
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax2.set_xlabel("Predicted ($)")
    ax2.set_ylabel("Residual ($)")
    ax2.set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_error_metrics(metrics, ax=None):
    """Bar chart of regression error metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    display = {
        "MSE": metrics["mse"],
        "RMSE": metrics["rmse"],
        "MAE": metrics["mae"],
        "R\u00b2": metrics["r2"],
    }
    names = list(display.keys())
    values = list(display.values())
    colors = sns.color_palette("Set2", n_colors=len(names))

    bars = ax.bar(names, values, color=colors)
    ax.set_title("Regression Metrics", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.2f}", ha="center", va="bottom", fontweight="bold",
                fontsize=10)

    plt.tight_layout()
    return ax
