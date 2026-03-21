"""Data loading and preprocessing utilities for the ANN Car Purchase demo."""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = [
    "gender",
    "age",
    "annual Salary",
    "credit card debt",
    "net worth",
]
TARGET_COLUMN = "car purchase amount"
DROP_COLUMNS = ["customer name", "customer e-mail", "country"]

_BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATASET_PATH = os.path.join(_BASE_DIR, "Dataset", "car_purchasing.csv")


def load_car_data(path=None):
    """Load car_purchasing.csv, drop non-numeric columns, return a DataFrame."""
    if path is None:
        path = DEFAULT_DATASET_PATH
    df = pd.read_csv(path, encoding="latin-1")
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return df


def split_features_target(df):
    """Separate feature matrix *X* and target vector *y*.

    Returns
    -------
    X : pd.DataFrame  (n_samples, 5)
    y : pd.Series
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def get_scalers(X_train, y_train):
    """Fit MinMaxScaler on training features and target.

    Returns
    -------
    X_scaled : np.ndarray
    y_scaled : np.ndarray  (n, 1)
    feature_scaler : MinMaxScaler
    target_scaler  : MinMaxScaler
    """
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X_train)
    y_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    return X_scaled, y_scaled, feature_scaler, target_scaler


def transform_data(X, y, feature_scaler, target_scaler):
    """Apply already-fitted scalers to a feature/target pair.

    Returns
    -------
    X_scaled : np.ndarray
    y_scaled : np.ndarray  (n, 1)
    """
    X_scaled = feature_scaler.transform(X)
    y_scaled = target_scaler.transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled


def create_tensor_dataset(X_scaled, y_scaled):
    """Wrap NumPy arrays into a PyTorch TensorDataset."""
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def create_dataloaders(train_ds, val_ds, test_ds, batch_size=32):
    """Create DataLoader instances for train, val, and test splits.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_data(path=None, test_size=0.15, val_size=0.15, batch_size=32,
                 random_state=42):
    """End-to-end convenience: load, split, scale, and build DataLoaders.

    Returns
    -------
    loaders : tuple  (train_loader, val_loader, test_loader)
    scalers : tuple  (feature_scaler, target_scaler)
    splits  : dict   raw DataFrames for X/y per split
    """
    df = load_car_data(path)
    X, y = split_features_target(df)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state,
    )

    X_train_sc, y_train_sc, feat_scaler, tgt_scaler = get_scalers(X_train, y_train)
    X_val_sc, y_val_sc = transform_data(X_val, y_val, feat_scaler, tgt_scaler)
    X_test_sc, y_test_sc = transform_data(X_test, y_test, feat_scaler, tgt_scaler)

    train_ds = create_tensor_dataset(X_train_sc, y_train_sc)
    val_ds = create_tensor_dataset(X_val_sc, y_val_sc)
    test_ds = create_tensor_dataset(X_test_sc, y_test_sc)

    loaders = create_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    scalers = (feat_scaler, tgt_scaler)
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    return loaders, scalers, splits
