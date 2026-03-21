#!/usr/bin/env python3
"""CLI entry-point that runs the full SVM Wine Quality pipeline.

Usage:
    python main.py
"""

import os

from sklearn.model_selection import train_test_split

from data_loader import (
    load_wine_data,
    split_features_target,
    binarize_target,
    get_feature_names,
)
from model import (
    scale_features,
    train_svm,
    evaluate_model,
    tune_hyperparameters,
)
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_kernel_comparison,
)

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  SVM Wine Quality Classification Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading data ...")
    df = load_wine_data()
    X, y = split_features_target(df)
    print(f"  Samples  : {len(X)}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Quality distribution:\n{y.value_counts().sort_index().to_string()}")

    # --- Binary target ---
    print("\n[2/6] Binarising target (good >= 7) ...")
    y_bin = binarize_target(y, threshold=7)
    print(f"  Class 0 (not good): {(y_bin == 0).sum()}")
    print(f"  Class 1 (good)    : {(y_bin == 1).sum()}")

    # --- Split & scale ---
    print("\n[3/6] Train/test split (80/20) + StandardScaler ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.2, stratify=y_bin, random_state=RANDOM_STATE,
    )
    X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
    print(f"  Train: {X_train_sc.shape[0]}  |  Test: {X_test_sc.shape[0]}")

    # --- Kernel comparison ---
    print("\n[4/6] Training SVMs with different kernels ...")
    kernel_results = {}
    for kernel in ("linear", "rbf", "poly"):
        model = train_svm(X_train_sc, y_train, kernel=kernel,
                          random_state=RANDOM_STATE)
        res = evaluate_model(model, X_test_sc, y_test)
        kernel_results[kernel] = res["accuracy"]
        print(f"  {kernel:>6s} kernel accuracy: {res['accuracy']:.4f}")

    # --- Hyperparameter tuning ---
    print("\n[5/6] Running GridSearchCV ...")
    search = tune_hyperparameters(X_train_sc, y_train)
    print(f"  Best params   : {search.best_params_}")
    print(f"  Best CV score : {search.best_score_:.4f}")

    best_model = search.best_estimator_
    best_res = evaluate_model(best_model, X_test_sc, y_test)
    print(f"  Test accuracy : {best_res['accuracy']:.4f}")
    print(best_res["report"])

    # --- Save figures ---
    print("\n[6/6] Saving visualisations ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_class_distribution(y)
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close()

    plot_confusion_matrix(best_res["confusion_matrix"],
                          class_names=["Not Good", "Good"])
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    plot_kernel_comparison(kernel_results)
    plt.savefig(os.path.join(fig_dir, "kernel_comparison.png"), dpi=150)
    plt.close()

    print(f"  Figures saved to {fig_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
