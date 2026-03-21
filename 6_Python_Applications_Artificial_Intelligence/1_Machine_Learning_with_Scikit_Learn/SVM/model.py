"""SVM model training, evaluation, and hyperparameter tuning utilities."""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


def scale_features(X_train, X_test=None):
    """Fit a StandardScaler on X_train and transform both sets.

    Returns
    -------
    X_train_scaled, X_test_scaled (or None), scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def train_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale", **kwargs):
    """Train an SVC with the given hyperparameters.

    Any additional keyword argument accepted by sklearn.svm.SVC can be passed.
    """
    clf = SVC(kernel=kernel, C=C, gamma=gamma, **kwargs)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return a results dictionary.

    Returns
    -------
    dict with keys: accuracy, report (str), confusion_matrix (ndarray)
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def predict(model, X):
    """Generate class predictions for feature matrix X."""
    return model.predict(X)


def tune_hyperparameters(X_train, y_train, param_grid=None, cv=5):
    """Run GridSearchCV over an SVM parameter grid.

    Parameters
    ----------
    param_grid : dict or None
        If None a sensible default grid is used.
    """
    if param_grid is None:
        param_grid = {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        }
    base = SVC()
    search = GridSearchCV(
        base, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search
