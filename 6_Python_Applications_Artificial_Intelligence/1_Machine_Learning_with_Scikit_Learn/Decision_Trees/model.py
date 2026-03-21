"""Decision Tree model training, evaluation, and prediction utilities."""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV


def train_decision_tree(X_train, y_train, **kwargs):
    """Train a DecisionTreeClassifier with the given hyper-parameters.

    Any keyword argument accepted by DecisionTreeClassifier can be passed
    (e.g. criterion, max_depth, min_samples_split, random_state).
    """
    clf = DecisionTreeClassifier(**kwargs)
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
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def predict(model, X):
    """Generate class predictions for feature matrix X."""
    return model.predict(X)


def tune_hyperparameters(X_train, y_train, param_grid=None, cv=5,
                         random_state=42):
    """Run GridSearchCV over a parameter grid and return the search object.

    Parameters
    ----------
    param_grid : dict or None
        If None a sensible default grid is used.
    """
    if param_grid is None:
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    base = DecisionTreeClassifier(random_state=random_state)
    search = GridSearchCV(base, param_grid, cv=cv, scoring="accuracy",
                          n_jobs=-1, return_train_score=True)
    search.fit(X_train, y_train)
    return search


def generate_submission(predictions, output_path):
    """Write predictions to a CSV file in the sample-submission format (id, class)."""
    submission = pd.DataFrame({
        "id": range(1, len(predictions) + 1),
        "class": predictions.astype(int),
    })
    submission.to_csv(output_path, index=False)
    return submission
