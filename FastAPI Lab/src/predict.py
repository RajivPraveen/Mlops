"""
Prediction module for Breast Cancer classification.

This module handles loading the trained model and making predictions
on new data.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from src.data import get_feature_names, get_target_names


class BreastCancerPredictor:
    """
    Predictor class for breast cancer classification.
    
    This class loads the trained model and scaler, and provides
    methods for making predictions on new data.
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize the predictor by loading model artifacts.
        
        Args:
            model_dir: Directory containing model files (default: "model")
        """
        self.model = None
        self.scaler = None
        self.metrics = None
        self.feature_names = get_feature_names()
        self.target_names = get_target_names()
        
        self._load_model(model_dir)
    
    def _load_model(self, model_dir: str) -> None:
        """
        Load model artifacts from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        # Handle different working directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        model_path = os.path.join(project_dir, model_dir)
        
        model_file = os.path.join(model_path, "model.pkl")
        scaler_file = os.path.join(model_path, "scaler.pkl")
        metrics_file = os.path.join(model_path, "metrics.pkl")
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file not found at {model_file}. "
                "Please run 'python src/train.py' first to train the model."
            )
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.metrics = joblib.load(metrics_file)
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Dictionary mapping feature names to values.
                     Must contain all 30 features.
        
        Returns:
            Dict containing:
                - prediction: Class name ('Malignant' or 'Benign')
                - prediction_code: Class code (0 or 1)
                - probability: Dict with probabilities for each class
        """
        # Validate input features
        self._validate_features(features)
        
        # Convert to array in correct order
        X = np.array([[features[name] for name in self.feature_names]])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction_code = int(self.model.predict(X_scaled)[0])
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        return {
            "prediction": self.target_names[prediction_code],
            "prediction_code": prediction_code,
            "probability": {
                "malignant": float(probabilities[0]),
                "benign": float(probabilities[1])
            }
        }
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of prediction results
        """
        return [self.predict(features) for features in features_list]
    
    def _validate_features(self, features: Dict[str, float]) -> None:
        """
        Validate that all required features are present.
        
        Args:
            features: Feature dictionary to validate
        
        Raises:
            ValueError: If features are missing or invalid
        """
        provided = set(features.keys())
        required = set(self.feature_names)
        
        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )
        
        extra = provided - required
        if extra:
            raise ValueError(
                f"Unknown features provided: {sorted(extra)}"
            )
        
        # Validate values are numeric
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Feature '{name}' must be numeric, got {type(value).__name__}"
                )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model metadata and performance metrics
        """
        # Get top 10 feature importances
        importances = self.metrics.get("feature_importances", {})
        top_features = dict(list(importances.items())[:10])
        
        return {
            "model_type": "RandomForestClassifier",
            "n_estimators": self.metrics.get("n_estimators"),
            "max_depth": self.metrics.get("max_depth"),
            "accuracy": self.metrics.get("accuracy"),
            "precision": self.metrics.get("precision"),
            "recall": self.metrics.get("recall"),
            "f1_score": self.metrics.get("f1_score"),
            "n_features": len(self.feature_names),
            "classes": self.target_names,
            "top_10_feature_importances": top_features
        }


# Global predictor instance (lazy loading)
_predictor: Optional[BreastCancerPredictor] = None


def get_predictor() -> BreastCancerPredictor:
    """
    Get or create the global predictor instance.
    
    Returns:
        BreastCancerPredictor: The predictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = BreastCancerPredictor()
    return _predictor


def predict(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Convenience function for making a single prediction.
    
    Args:
        features: Dictionary of feature values
    
    Returns:
        Prediction result dictionary
    """
    predictor = get_predictor()
    return predictor.predict(features)


if __name__ == "__main__":
    # Test prediction with sample data
    from src.data import get_sample_data
    
    print("Testing Breast Cancer Predictor")
    print("="*50)
    
    try:
        predictor = BreastCancerPredictor()
        
        # Get sample data (first sample - malignant)
        sample = get_sample_data()
        print("\nSample features (first 5):")
        for name, value in list(sample.items())[:5]:
            print(f"  {name}: {value:.4f}")
        print("  ...")
        
        # Make prediction
        result = predictor.predict(sample)
        
        print(f"\nPrediction Result:")
        print(f"  Diagnosis: {result['prediction']}")
        print(f"  Code: {result['prediction_code']}")
        print(f"  Probability (Malignant): {result['probability']['malignant']:.4f}")
        print(f"  Probability (Benign): {result['probability']['benign']:.4f}")
        
        # Get model info
        print(f"\nModel Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            if key != "top_10_feature_importances":
                print(f"  {key}: {value}")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run 'python train.py' first to train the model.")
