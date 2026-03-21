"""Visualization utilities for CNN Intel Image Classification demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_loader import CLASS_NAMES, denormalize


def plot_class_distribution(dataset, title="Class Distribution", ax=None):
    """Bar chart of samples per class."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "dataset"):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = [label for _, label in dataset]

    counts = np.bincount(targets, minlength=len(CLASS_NAMES))
    bars = ax.bar(CLASS_NAMES, counts, color=sns.color_palette("Set2", len(CLASS_NAMES)))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return ax


def plot_sample_images(dataset, class_names=CLASS_NAMES, n_per_class=3):
    """Display a grid of sample images from each class."""
    fig, axes = plt.subplots(len(class_names), n_per_class,
                             figsize=(n_per_class * 3, len(class_names) * 2.5))
    class_counts = {i: 0 for i in range(len(class_names))}

    for img, label in dataset:
        if class_counts[label] < n_per_class:
            col = class_counts[label]
            ax = axes[label][col]
            img_display = denormalize(img).permute(1, 2, 0).numpy()
            ax.imshow(img_display)
            ax.axis("off")
            if col == 0:
                ax.set_title(class_names[label], fontsize=11, fontweight="bold")
            class_counts[label] += 1
        if all(c >= n_per_class for c in class_counts.values()):
            break

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_augmented_samples(dataset, idx=0, n_samples=6):
    """Show the same image with different random augmentations applied."""
    fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2.5, 2.5))
    for i, ax in enumerate(axes):
        img, label = dataset[idx]
        img_display = denormalize(img).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        ax.axis("off")
        ax.set_title(f"Aug {i+1}", fontsize=9)
    fig.suptitle(f"Augmented views — class: {CLASS_NAMES[label]}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """Plot loss and accuracy curves over training epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s-", label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", label="Train", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "s-", label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_lr_schedule(history):
    """Plot the learning rate across epochs."""
    epochs = range(1, len(history["lr"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, history["lr"], "D-", color="tab:green", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, ax=None):
    """Heatmap of the confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return ax


def print_classification_report(y_true, y_pred, class_names=CLASS_NAMES):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_predictions_grid(model, dataset, device, class_names=CLASS_NAMES,
                          n_samples=12, ncols=4):
    """Show a grid of predictions with correct/incorrect color-coding."""
    import torch

    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    model.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            prob = torch.softmax(output, dim=1)
            pred = prob.argmax(1).item()
            confidence = prob[0, pred].item()

        img_display = denormalize(img).permute(1, 2, 0).numpy()
        axes[i].imshow(img_display)
        axes[i].axis("off")

        color = "green" if pred == label else "red"
        axes[i].set_title(
            f"Pred: {class_names[pred]}\nTrue: {class_names[label]}\n"
            f"Conf: {confidence:.1%}",
            fontsize=8, color=color, fontweight="bold",
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Model Predictions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_top_losses(model, dataset, device, class_names=CLASS_NAMES, n=8):
    """Display the images where the model is most confidently wrong."""
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        with torch.no_grad():
            output = model(img.unsqueeze(0).to(device))
            loss = F.cross_entropy(output, torch.tensor([label]).to(device))
            pred = output.argmax(1).item()
        losses.append((loss.item(), idx, pred, label))

    losses.sort(reverse=True)
    top = losses[:n]

    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 3))
    if n == 1:
        axes = [axes]

    for i, (loss_val, idx, pred, label) in enumerate(top):
        img, _ = dataset[idx]
        img_display = denormalize(img).permute(1, 2, 0).numpy()
        axes[i].imshow(img_display)
        axes[i].axis("off")
        axes[i].set_title(
            f"Loss: {loss_val:.2f}\nPred: {class_names[pred]}\nTrue: {class_names[label]}",
            fontsize=8, color="red", fontweight="bold",
        )

    fig.suptitle("Highest-Loss Predictions", fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    return fig
