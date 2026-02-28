import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the Wine dataset and return features and target values.

    The Wine dataset contains 13 chemical analysis measurements from wines
    grown in the same region of Italy but derived from three different cultivars.

    Returns:
        X (numpy.ndarray): Feature matrix of shape (178, 13).
        y (numpy.ndarray): Target array with class labels (0, 1, 2).
        feature_names (list): Names of the 13 features.
        target_names (list): Names of the 3 wine cultivar classes.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    return X, y, list(wine.feature_names), list(wine.target_names)


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the data into training and testing sets.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target array.
        test_size (float): Fraction of data to reserve for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
