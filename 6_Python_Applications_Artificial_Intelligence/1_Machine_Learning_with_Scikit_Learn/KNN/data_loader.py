"""Data loading and preprocessing utilities for the KNN T-Shirt Size demo."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = ["Height (in cms)", "Weight (in kgs)"]
TARGET_COLUMN = "T Shirt Size"

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "Dataset", "TShirt_size.csv"
)


def load_tshirt_data(path=None):
    """Load TShirt_size.csv and return a DataFrame."""
    if path is None:
        path = DEFAULT_DATASET_PATH
    return pd.read_csv(path)


def split_features_target(df):
    """Separate a T-shirt DataFrame into feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def encode_target(y):
    """Encode categorical target (M/L) to numeric (0/1).

    Returns
    -------
    y_encoded : ndarray of int
    encoder   : fitted LabelEncoder (use encoder.inverse_transform to decode)
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def get_feature_names():
    """Return the list of feature column names."""
    return list(FEATURE_COLUMNS)
