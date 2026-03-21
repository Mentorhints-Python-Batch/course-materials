"""Data loading and preprocessing utilities for the Decision Trees demo."""

import os
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 10)]
TARGET_COLUMN = "target"

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")


def load_training_data(path=None):
    """Load training.csv (no header) and assign meaningful column names.

    Returns a DataFrame with columns feature_1..feature_9 and target.
    """
    if path is None:
        path = os.path.join(DEFAULT_DATASET_DIR, "training.csv")
    columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    return pd.read_csv(path, header=None, names=columns)


def load_testing_data(path=None):
    """Load testing.csv (no header, features only).

    Returns a DataFrame with columns feature_1..feature_9.
    """
    if path is None:
        path = os.path.join(DEFAULT_DATASET_DIR, "testing.csv")
    return pd.read_csv(path, header=None, names=FEATURE_COLUMNS)


def load_sample_submission(path=None):
    """Load sample.csv submission template (has header: id,class)."""
    if path is None:
        path = os.path.join(DEFAULT_DATASET_DIR, "sample.csv")
    return pd.read_csv(path)


def split_features_target(df):
    """Separate a training DataFrame into feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
