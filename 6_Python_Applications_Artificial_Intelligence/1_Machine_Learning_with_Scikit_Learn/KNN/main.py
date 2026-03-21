#!/usr/bin/env python3
"""CLI entry-point that runs the full KNN T-Shirt Size pipeline.

Usage:
    python main.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import (
    load_tshirt_data,
    split_features_target,
    encode_target,
    get_feature_names,
)
from model import (
    scale_features,
    train_knn,
    evaluate_model,
    find_best_k,
    tune_hyperparameters,
)
from visualization import (
    plot_class_distribution,
    plot_scatter,
    plot_k_vs_accuracy,
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_metric_comparison,
)

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  KNN T-Shirt Size Classification Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/7] Loading data ...")
    df = load_tshirt_data()
    X, y = split_features_target(df)
    y_enc, encoder = encode_target(y)
    print(f"  Samples : {len(X)}")
    print(f"  Features: {get_feature_names()}")
    print(f"  Classes : {dict(zip(encoder.classes_, np.bincount(y_enc)))}")

    # --- Split & scale ---
    print("\n[2/7] Train/test split (80/20) + StandardScaler ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE,
    )
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    print(f"  Train: {X_train_sc.shape[0]}  |  Test: {X_test_sc.shape[0]}")

    # --- Find best K ---
    print("\n[3/7] Finding best K (LOOCV on training set) ...")
    k_scores = find_best_k(X_train_sc, y_train,
                           k_range=range(1, min(len(y_train), 14)),
                           cv=min(len(y_train), 5))
    best_k = max(k_scores, key=k_scores.get)
    print(f"  Best K={best_k}  (CV accuracy={k_scores[best_k]:.4f})")

    # --- Train with best K ---
    print("\n[4/7] Training KNN with K={} ...".format(best_k))
    model = train_knn(X_train_sc, y_train, n_neighbors=best_k)
    results = evaluate_model(model, X_test_sc, y_test)
    print(f"  Test accuracy: {results['accuracy']:.4f}")
    print(results["report"])

    # --- Metric comparison ---
    print("\n[5/7] Comparing distance metrics ...")
    metric_results = {}
    for metric in ("euclidean", "manhattan"):
        m = train_knn(X_train_sc, y_train, n_neighbors=best_k, metric=metric)
        r = evaluate_model(m, X_test_sc, y_test)
        metric_results[metric] = r["accuracy"]
        print(f"  {metric:>12s}: {r['accuracy']:.4f}")

    # --- GridSearchCV ---
    print("\n[6/7] Running GridSearchCV ...")
    search = tune_hyperparameters(X_train_sc, y_train,
                                  cv=min(len(y_train), 5))
    print(f"  Best params   : {search.best_params_}")
    print(f"  Best CV score : {search.best_score_:.4f}")

    # --- Save figures ---
    print("\n[7/7] Saving visualisations ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_class_distribution(y)
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close()

    plot_scatter(X, y, get_feature_names())
    plt.savefig(os.path.join(fig_dir, "scatter.png"), dpi=150)
    plt.close()

    plot_k_vs_accuracy(k_scores)
    plt.savefig(os.path.join(fig_dir, "k_vs_accuracy.png"), dpi=150)
    plt.close()

    plot_confusion_matrix(results["confusion_matrix"],
                          class_names=list(encoder.classes_))
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    X_all_sc = scaler.transform(X)
    plot_decision_boundary(
        train_knn(X_all_sc, y_enc, n_neighbors=best_k),
        X_all_sc, y_enc, title=f"Decision Boundary (K={best_k})")
    plt.savefig(os.path.join(fig_dir, "decision_boundary.png"), dpi=150)
    plt.close()

    print(f"  Figures saved to {fig_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
