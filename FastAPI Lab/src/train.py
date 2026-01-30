"""
Model training module for Breast Cancer classification.

This module handles training a Random Forest Classifier on the
Breast Cancer Wisconsin dataset and saving the model artifacts.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
from typing import Dict, Tuple, Any

from src.data import prepare_data, get_feature_names, get_target_names


def train_model(
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[RandomForestClassifier, Any, Dict]:
    """
    Train a Random Forest Classifier on the Breast Cancer dataset.
    
    Args:
        n_estimators: Number of trees in the forest (default: 100)
        max_depth: Maximum depth of the trees (default: 10)
        random_state: Random seed for reproducibility (default: 42)
        verbose: Whether to print training progress (default: True)
    
    Returns:
        Tuple containing:
            - model: Trained RandomForestClassifier
            - scaler: Fitted StandardScaler
            - metrics: Dictionary of evaluation metrics
    """
    if verbose:
        print("=" * 60)
        print("BREAST CANCER CLASSIFICATION - MODEL TRAINING")
        print("=" * 60)
        print("\n[1/4] Loading and preparing data...")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        test_size=0.2,
        random_state=random_state,
        scale=True
    )
    
    if verbose:
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
    
    # Initialize model
    if verbose:
        print("\n[2/4] Initializing Random Forest Classifier...")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Train model
    if verbose:
        print("\n[3/4] Training model...")
    
    model.fit(X_train, y_train)
    
    if verbose:
        print("  Training complete!")
    
    # Evaluate model
    if verbose:
        print("\n[4/4] Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average='weighted')),
        "recall": float(recall_score(y_test, y_pred, average='weighted')),
        "f1_score": float(f1_score(y_test, y_pred, average='weighted')),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "n_train_samples": X_train.shape[0],
        "n_test_samples": X_test.shape[0]
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT")
        print("="*60)
        target_names = get_target_names()
        print(classification_report(
            y_test, y_pred, 
            target_names=[target_names[0], target_names[1]]
        ))
        
        print("CONFUSION MATRIX")
        print("-"*30)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  True Negatives (Malignant correct):  {cm[0][0]}")
        print(f"  False Positives (Malignant as Benign): {cm[0][1]}")
        print(f"  False Negatives (Benign as Malignant): {cm[1][0]}")
        print(f"  True Positives (Benign correct):     {cm[1][1]}")
    
    # Get feature importances
    feature_names = get_feature_names()
    importances = model.feature_importances_
    importance_dict = {
        name: float(imp) 
        for name, imp in sorted(
            zip(feature_names, importances), 
            key=lambda x: x[1], 
            reverse=True
        )
    }
    metrics["feature_importances"] = importance_dict
    
    if verbose:
        print(f"\n{'='*60}")
        print("TOP 10 FEATURE IMPORTANCES")
        print("="*60)
        for i, (name, imp) in enumerate(list(importance_dict.items())[:10]):
            print(f"  {i+1:2d}. {name:<30} {imp:.4f}")
    
    return model, scaler, metrics


def save_model(
    model: RandomForestClassifier, 
    scaler: Any, 
    metrics: Dict,
    model_dir: str = "model"
) -> str:
    """
    Save the trained model, scaler, and metrics to disk.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        metrics: Model metrics dictionary
        model_dir: Directory to save model files (default: "model")
    
    Returns:
        str: Path to the saved model file
    """
    # Create model directory if it doesn't exist
    # Handle both running from src/ and from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_path = os.path.join(project_dir, model_dir)
    
    os.makedirs(model_path, exist_ok=True)
    
    # Save model artifacts
    model_file = os.path.join(model_path, "model.pkl")
    scaler_file = os.path.join(model_path, "scaler.pkl")
    metrics_file = os.path.join(model_path, "metrics.pkl")
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(metrics, metrics_file)
    
    print(f"\n{'='*60}")
    print("MODEL SAVED SUCCESSFULLY")
    print("="*60)
    print(f"  Model:   {model_file}")
    print(f"  Scaler:  {scaler_file}")
    print(f"  Metrics: {metrics_file}")
    
    return model_file


def main():
    """Main function to train and save the model."""
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Train model
    model, scaler, metrics = train_model(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        verbose=True
    )
    
    # Save model
    save_model(model, scaler, metrics)
    
    print(f"\n{'='*60}")
    print("TRAINING PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
