"""Naive Bayes model training, evaluation, and hyperparameter tuning utilities."""

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, GridSearchCV

VARIANT_MAP = {
    "multinomial": MultinomialNB,
    "bernoulli": BernoulliNB,
    "complement": ComplementNB,
}


def train_naive_bayes(X_train, y_train, variant="multinomial", alpha=1.0,
                      **kwargs):
    """Train a Naive Bayes classifier.

    Parameters
    ----------
    variant : str
        One of ``"multinomial"``, ``"bernoulli"``, ``"complement"``.
    alpha : float
        Laplace / additive smoothing parameter.
    """
    cls = VARIANT_MAP[variant]
    model = cls(alpha=alpha, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return a results dictionary.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, report (str),
    confusion_matrix (ndarray)
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def predict(model, X):
    """Generate class predictions for feature matrix X."""
    return model.predict(X)


def compare_variants(X_train, y_train, X_test, y_test, alpha=1.0):
    """Train all three NB variants and return a comparison dict.

    Returns
    -------
    dict mapping variant name -> evaluation results dict
    """
    results = {}
    for variant in VARIANT_MAP:
        model = train_naive_bayes(X_train, y_train, variant=variant,
                                  alpha=alpha)
        results[variant] = evaluate_model(model, X_test, y_test)
        results[variant]["model"] = model
    return results


def cross_validate_model(X, y, variant="multinomial", alpha=1.0, cv=5):
    """Run cross-validation and return per-fold scores.

    Returns
    -------
    dict with keys: scores (array), mean, std
    """
    cls = VARIANT_MAP[variant]
    model = cls(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}


def alpha_sensitivity(X_train, y_train, X_test, y_test,
                      variant="multinomial",
                      alphas=None):
    """Evaluate model accuracy across a range of alpha values.

    Returns
    -------
    dict mapping alpha -> accuracy
    """
    if alphas is None:
        alphas = np.logspace(-2, 1, 20)
    results = {}
    for a in alphas:
        model = train_naive_bayes(X_train, y_train, variant=variant, alpha=a)
        results[a] = evaluate_model(model, X_test, y_test)["accuracy"]
    return results


def tune_hyperparameters(X_train, y_train, variant="multinomial",
                         param_grid=None, cv=5):
    """Run GridSearchCV over a Naive Bayes parameter grid.

    Parameters
    ----------
    param_grid : dict or None
        If None a sensible default grid over alpha values is used.
    """
    if param_grid is None:
        param_grid = {
            "alpha": np.logspace(-2, 1, 20).tolist(),
        }
    cls = VARIANT_MAP[variant]
    base = cls()
    search = GridSearchCV(
        base, param_grid, cv=cv, scoring="accuracy",
        n_jobs=-1, return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search
