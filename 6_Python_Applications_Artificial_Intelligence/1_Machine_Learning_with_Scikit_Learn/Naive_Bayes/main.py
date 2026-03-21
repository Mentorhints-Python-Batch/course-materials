#!/usr/bin/env python3
"""CLI entry-point that runs the full Naive Bayes SMS Spam Classification pipeline.

Usage:
    python main.py
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import (
    load_spam_data,
    split_features_target,
    encode_target,
    vectorize_text,
    add_text_stats,
    get_class_names,
)
from model import (
    train_naive_bayes,
    evaluate_model,
    compare_variants,
    alpha_sensitivity,
    tune_hyperparameters,
)
from visualization import (
    plot_class_distribution,
    plot_message_length_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_top_features,
    plot_variant_comparison,
    plot_alpha_tuning,
)

RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("  Naive Bayes — SMS Spam Classification Pipeline")
    print("=" * 60)

    # --- 1. Load data ---
    print("\n[1/7] Loading data ...")
    df = load_spam_data()
    X, y = split_features_target(df)
    y_enc, encoder = encode_target(y)
    print(f"  Samples : {len(X)}")
    print(f"  Classes : {dict(zip(encoder.classes_, np.bincount(y_enc)))}")

    df_stats = add_text_stats(df)
    print(f"  Avg message length (ham) : "
          f"{df_stats[df_stats['label'] == 'ham']['message_length'].mean():.0f} chars")
    print(f"  Avg message length (spam): "
          f"{df_stats[df_stats['label'] == 'spam']['message_length'].mean():.0f} chars")

    # --- 2. Train/test split ---
    print("\n[2/7] Train/test split (80/20) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    # --- 3. TF-IDF vectorization ---
    print("\n[3/7] TF-IDF vectorization ...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  TF-IDF matrix : {X_train_tfidf.shape}")

    # --- 4. Baseline MultinomialNB ---
    print("\n[4/7] Training MultinomialNB baseline (alpha=1.0) ...")
    baseline = train_naive_bayes(X_train_tfidf, y_train)
    baseline_results = evaluate_model(baseline, X_test_tfidf, y_test)
    print(f"  Accuracy : {baseline_results['accuracy']:.4f}")
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall   : {baseline_results['recall']:.4f}")
    print(f"  F1       : {baseline_results['f1']:.4f}")
    print(baseline_results["report"])

    # --- 5. Compare NB variants ---
    print("\n[5/7] Comparing NB variants ...")
    variant_results = compare_variants(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    for name, res in variant_results.items():
        print(f"  {name:>12s}: Acc={res['accuracy']:.4f}  "
              f"P={res['precision']:.4f}  R={res['recall']:.4f}  "
              f"F1={res['f1']:.4f}")

    # --- 6. Hyperparameter tuning ---
    print("\n[6/7] Running GridSearchCV (MultinomialNB, alpha sweep) ...")
    search = tune_hyperparameters(X_train_tfidf, y_train)
    print(f"  Best params   : {search.best_params_}")
    print(f"  Best CV score : {search.best_score_:.4f}")

    best_model = search.best_estimator_
    best_results = evaluate_model(best_model, X_test_tfidf, y_test)
    print(f"  Test accuracy : {best_results['accuracy']:.4f}")

    # --- 7. Save figures ---
    print("\n[7/7] Saving visualisations ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_class_distribution(y)
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=150)
    plt.close()

    plot_message_length_distribution(df_stats)
    plt.savefig(os.path.join(fig_dir, "message_length.png"), dpi=150)
    plt.close()

    plot_confusion_matrix(best_results["confusion_matrix"],
                          class_names=get_class_names())
    plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    plot_roc_curve(best_model, X_test_tfidf, y_test)
    plt.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_top_features(vectorizer, best_model, n=15, ax=axes)
    plt.savefig(os.path.join(fig_dir, "top_features.png"), dpi=150)
    plt.close()

    plot_variant_comparison(variant_results)
    plt.savefig(os.path.join(fig_dir, "variant_comparison.png"), dpi=150)
    plt.close()

    alpha_scores = alpha_sensitivity(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )
    plot_alpha_tuning(alpha_scores)
    plt.savefig(os.path.join(fig_dir, "alpha_tuning.png"), dpi=150)
    plt.close()

    print(f"  Figures saved to {fig_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
