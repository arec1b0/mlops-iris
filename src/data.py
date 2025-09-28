"""Data loading and preprocessing functions for Iris classification."""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import numpy as np


def load_iris_data() -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load the Iris dataset.

    Returns:
        Tuple of (X, y, target_names) where:
        - X: feature matrix
        - y: target labels
        - target_names: list of class names
    """
    data = load_iris()
    X = data.data
    y = data.target
    target_names = data.target_names.tolist()

    return X, y, target_names


def split_data(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
