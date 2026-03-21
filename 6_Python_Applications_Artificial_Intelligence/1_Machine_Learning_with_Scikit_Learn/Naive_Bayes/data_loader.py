"""Data loading and preprocessing utilities for the Naive Bayes SMS Spam demo."""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

LABEL_COLUMN = "label"
MESSAGE_COLUMN = "message"

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "Dataset", "spam.csv"
)


def load_spam_data(path=None):
    """Load spam.csv, keep only the label and message columns.

    Returns a DataFrame with columns ``label`` (ham/spam) and ``message``.
    """
    if path is None:
        path = DEFAULT_DATASET_PATH
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = [LABEL_COLUMN, MESSAGE_COLUMN]
    return df


def split_features_target(df):
    """Separate a spam DataFrame into messages X and labels y."""
    X = df[MESSAGE_COLUMN].copy()
    y = df[LABEL_COLUMN].copy()
    return X, y


def encode_target(y):
    """Encode categorical target (ham/spam) to numeric (0/1).

    Returns
    -------
    y_encoded : ndarray of int
    encoder   : fitted LabelEncoder (ham=0, spam=1)
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def vectorize_text(X_train, X_test=None, max_features=5000,
                   stop_words="english", ngram_range=(1, 2)):
    """Fit a TfidfVectorizer on X_train and transform both sets.

    Returns
    -------
    X_train_tfidf : sparse matrix
    X_test_tfidf  : sparse matrix or None
    vectorizer    : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test) if X_test is not None else None
    return X_train_tfidf, X_test_tfidf, vectorizer


def add_text_stats(df):
    """Add message_length and word_count columns to a copy of df."""
    df = df.copy()
    df["message_length"] = df[MESSAGE_COLUMN].str.len()
    df["word_count"] = df[MESSAGE_COLUMN].str.split().str.len()
    return df


def get_class_names():
    """Return the ordered class labels."""
    return ["ham", "spam"]
