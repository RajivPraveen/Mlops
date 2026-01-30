"""
Data loading and preprocessing module for Breast Cancer Wisconsin dataset.

This module handles loading the dataset from scikit-learn and preparing
it for model training and inference.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict


def get_feature_names() -> List[str]:
    """
    Returns the list of feature names for the Breast Cancer dataset.
    
    The features are organized into three groups (mean, standard error, worst)
    for each of the 10 base measurements.
    
    Returns:
        List[str]: List of 30 feature names
    """
    base_features = [
        'radius', 'texture', 'perimeter', 'area', 'smoothness',
        'compactness', 'concavity', 'concave_points', 'symmetry', 
        'fractal_dimension'
    ]
    
    feature_names = []
    for prefix in ['mean', 'se', 'worst']:
        for feature in base_features:
            feature_names.append(f"{prefix}_{feature}")
    
    return feature_names


def get_target_names() -> Dict[int, str]:
    """
    Returns the mapping of target values to class names.
    
    Returns:
        Dict[int, str]: Mapping of {0: 'Malignant', 1: 'Benign'}
    """
    return {0: 'Malignant', 1: 'Benign'}


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the Breast Cancer Wisconsin dataset.
    
    Returns:
        Tuple containing:
            - X: Feature matrix (569 samples, 30 features)
            - y: Target array (569 samples)
            - feature_names: List of feature names
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = get_feature_names()
    
    return X, y, feature_names


def load_data_as_dataframe() -> pd.DataFrame:
    """
    Load the dataset as a pandas DataFrame with proper column names.
    
    Returns:
        pd.DataFrame: DataFrame with features and target column
    """
    X, y, feature_names = load_data()
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['diagnosis'] = df['target'].map(get_target_names())
    
    return df


def prepare_data(
    test_size: float = 0.2, 
    random_state: int = 42,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for training by splitting into train/test sets and optionally scaling.
    
    Args:
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        scale: Whether to apply standard scaling (default: True)
    
    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
            - scaler: Fitted StandardScaler (or None if scale=False)
    """
    X, y, _ = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    # Scale features if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def get_sample_data() -> Dict[str, float]:
    """
    Returns a sample data point for testing the API.
    This is the first sample from the dataset (a malignant case).
    
    Returns:
        Dict[str, float]: Dictionary with feature names as keys
    """
    X, _, feature_names = load_data()
    sample = X[0]  # First sample (malignant)
    
    return {name: float(value) for name, value in zip(feature_names, sample)}


def get_dataset_info() -> Dict:
    """
    Returns information about the dataset.
    
    Returns:
        Dict: Dataset metadata including shape, class distribution, etc.
    """
    X, y, feature_names = load_data()
    target_names = get_target_names()
    
    return {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "target_names": target_names,
        "class_distribution": {
            target_names[0]: int(np.sum(y == 0)),
            target_names[1]: int(np.sum(y == 1))
        },
        "description": "Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass."
    }


if __name__ == "__main__":
    # Test the data loading
    print("Dataset Information:")
    info = get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nSample Data Point:")
    sample = get_sample_data()
    for name, value in list(sample.items())[:5]:
        print(f"  {name}: {value:.4f}")
    print("  ...")
