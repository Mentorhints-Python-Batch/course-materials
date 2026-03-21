"""KNN model training, evaluation, and hyperparameter tuning utilities."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV


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


def train_knn(X_train, y_train, n_neighbors=5, metric="euclidean",
              weights="uniform", **kwargs):
    """Train a KNeighborsClassifier with the given hyperparameters."""
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors, metric=metric, weights=weights, **kwargs
    )
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


def find_best_k(X, y, k_range=range(1, 16), cv=5):
    """Cross-validate each K value and return {k: mean_accuracy}.

    Uses Leave-One-Out when *cv* equals the number of samples.
    """
    scores = {}
    for k in k_range:
        clf = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        scores[k] = cv_scores.mean()
    return scores


def tune_hyperparameters(X_train, y_train, param_grid=None, cv=5):
    """Run GridSearchCV over a KNN parameter grid.

    Parameters
    ----------
    param_grid : dict or None
        If None a sensible default grid is used.
    """
    if param_grid is None:
        max_k = min(11, len(y_train))
        param_grid = {
            "n_neighbors": [k for k in range(1, max_k + 1, 2)],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "weights": ["uniform", "distance"],
        }
    base = KNeighborsClassifier()
    search = GridSearchCV(
        base, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search
