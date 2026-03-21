"""Data loading and preprocessing utilities for the SVM Wine Quality demo."""

import os
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET_COLUMN = "quality"

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "Dataset", "WineQuality.csv"
)


def load_wine_data(path=None):
    """Load WineQuality.csv and return a DataFrame."""
    if path is None:
        path = DEFAULT_DATASET_PATH
    return pd.read_csv(path)


def split_features_target(df):
    """Separate a wine DataFrame into feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def binarize_target(y, threshold=7):
    """Convert quality scores to binary labels (1 if >= threshold, else 0)."""
    return (y >= threshold).astype(int)


def get_feature_names():
    """Return the list of 11 feature column names."""
    return list(FEATURE_COLUMNS)
