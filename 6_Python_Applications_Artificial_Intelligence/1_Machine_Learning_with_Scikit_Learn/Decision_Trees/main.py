#!/usr/bin/env python3
"""CLI entry-point that runs the full Decision Tree pipeline.

Usage:
    python main.py                           # uses default Dataset/ paths
    python main.py --output submission.csv   # custom output path
"""

import argparse
import os

from sklearn.model_selection import train_test_split

from data_loader import (
    load_training_data,
    load_testing_data,
    split_features_target,
    FEATURE_COLUMNS,
)
from model import (
    train_decision_tree,
    evaluate_model,
    predict,
    tune_hyperparameters,
    generate_submission,
)
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_decision_tree,
)

RANDOM_STATE = 42


def main(output_path="submission.csv"):
    print("=" * 60)
    print("  Decision Tree Classification Pipeline")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/6] Loading data ...")
    train_df = load_training_data()
    test_df = load_testing_data()
    X, y = split_features_target(train_df)
    print(f"  Training samples : {len(X)}")
    print(f"  Test samples     : {len(test_df)}")
    print(f"  Features         : {len(FEATURE_COLUMNS)}")
    print(f"  Class distribution:\n{y.value_counts().to_string()}")

    # --- Train / validation split ---
    print("\n[2/6] Splitting into train / validation (80/20) ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    print(f"  Train size : {len(X_train)}")
    print(f"  Val size   : {len(X_val)}")

    # --- Baseline model ---
    print("\n[3/6] Training baseline Decision Tree (Gini, no depth limit) ...")
    baseline = train_decision_tree(X_train, y_train, random_state=RANDOM_STATE)
    baseline_results = evaluate_model(baseline, X_val, y_val)
    print(f"  Validation accuracy: {baseline_results['accuracy']:.4f}")
    print(baseline_results["report"])

    # --- Hyperparameter tuning ---
    print("\n[4/6] Running GridSearchCV for hyperparameter tuning ...")
    search = tune_hyperparameters(X_train, y_train, random_state=RANDOM_STATE)
    print(f"  Best params   : {search.best_params_}")
    print(f"  Best CV score : {search.best_score_:.4f}")

    best_model = search.best_estimator_
    best_results = evaluate_model(best_model, X_val, y_val)
    print(f"  Validation accuracy (tuned): {best_results['accuracy']:.4f}")
    print(best_results["report"])

    # --- Generate submission ---
    print("\n[5/6] Generating predictions on test set ...")
    test_preds = predict(best_model, test_df)
    abs_output = os.path.join(os.path.dirname(__file__), output_path)
    submission = generate_submission(test_preds, abs_output)
    print(f"  Submission saved to {abs_output}  ({len(submission)} rows)")

    # --- Visualisations (saved to files) ---
    print("\n[6/6] Saving visualisations ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_class_distribution(y)
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close()

    plot_confusion_matrix(best_results["confusion_matrix"])
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    plot_feature_importance(best_model, FEATURE_COLUMNS)
    plt.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=150)
    plt.close()

    plot_decision_tree(best_model, FEATURE_COLUMNS, max_depth=3)
    plt.savefig(os.path.join(fig_dir, "decision_tree.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"  Figures saved to {fig_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree pipeline")
    parser.add_argument("--output", default="submission.csv",
                        help="Output CSV path for test predictions")
    args = parser.parse_args()
    main(output_path=args.output)
