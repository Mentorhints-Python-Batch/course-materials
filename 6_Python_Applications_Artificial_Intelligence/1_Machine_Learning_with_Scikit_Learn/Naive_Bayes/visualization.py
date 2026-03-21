"""Visualization utilities for the Naive Bayes SMS Spam demo."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_class_distribution(y, ax=None):
    """Bar chart showing the count of ham vs spam messages."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    colors = ["#4CAF50", "#F44336"]
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Ham vs Spam Distribution")
    for i, v in enumerate(counts.values):
        pct = v / counts.sum() * 100
        ax.text(i, v + 30, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")
    plt.tight_layout()
    return ax


def plot_message_length_distribution(df, ax=None):
    """Overlaid histograms of message length for ham and spam."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [("ham", "#4CAF50"), ("spam", "#F44336")]:
        subset = df[df["label"] == label]
        ax.hist(subset["message_length"], bins=50, alpha=0.6,
                label=label, color=color, edgecolor="white")
    ax.set_xlabel("Message Length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Message Length Distribution by Class")
    ax.legend()
    plt.tight_layout()
    return ax


def plot_word_count_distribution(df, ax=None):
    """Box plot of word count for ham and spam."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    ham = df[df["label"] == "ham"]["word_count"]
    spam = df[df["label"] == "spam"]["word_count"]
    bp = ax.boxplot([ham, spam], labels=["Ham", "Spam"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax.set_ylabel("Word Count")
    ax.set_title("Word Count Distribution by Class")
    plt.tight_layout()
    return ax


def plot_confusion_matrix(cm, class_names=None, ax=None):
    """Heatmap of a confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    if class_names is None:
        class_names = ["Ham", "Spam"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return ax


def plot_roc_curve(model, X_test, y_test, ax=None):
    """ROC curve with AUC score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color="#1976D2", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax


def plot_top_features(vectorizer, model, n=15, ax=None):
    """Horizontal bar chart of the top N most informative words per class."""
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        axes = ax

    feature_names = np.array(vectorizer.get_feature_names_out())
    log_probs = model.feature_log_prob_

    for idx, (label, color) in enumerate([("Ham", "#4CAF50"), ("Spam", "#F44336")]):
        top_indices = np.argsort(log_probs[idx])[-n:]
        top_words = feature_names[top_indices]
        top_scores = log_probs[idx][top_indices]

        axes[idx].barh(range(n), top_scores, color=color, alpha=0.8)
        axes[idx].set_yticks(range(n))
        axes[idx].set_yticklabels(top_words)
        axes[idx].set_xlabel("Log Probability")
        axes[idx].set_title(f"Top {n} Words — {label}")
        axes[idx].invert_yaxis()

    plt.tight_layout()
    return axes


def plot_variant_comparison(results_dict, ax=None):
    """Grouped bar chart comparing NB variants across metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    metrics = ["accuracy", "precision", "recall", "f1"]
    variants = list(results_dict.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(variants)
    colors = sns.color_palette("Set2", n_colors=len(variants))

    for i, variant in enumerate(variants):
        values = [results_dict[variant][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=variant.title(),
                      color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Naive Bayes Variant Comparison")
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_alpha_tuning(alpha_scores, ax=None):
    """Line plot of alpha (smoothing) values vs accuracy."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    alphas = list(alpha_scores.keys())
    accs = list(alpha_scores.values())

    ax.plot(alphas, accs, "o-", color="#1976D2", linewidth=2, markersize=5)
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    ax.axvline(best_alpha, color="#F44336", linestyle="--", alpha=0.7,
               label=f"Best α={best_alpha:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (smoothing parameter)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Effect of Alpha on Model Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return ax
