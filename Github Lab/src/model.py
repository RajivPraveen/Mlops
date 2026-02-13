"""
Model Module
=============
Implements a simple Linear Regression model from scratch
(no external ML libraries). Demonstrates core ML concepts:
training, prediction, evaluation metrics, and model persistence.
"""

import json
import math
import os


class LinearRegressionModel:
    """
    Simple Linear Regression using Ordinary Least Squares (OLS).

    Fits y = slope * x + intercept to minimize sum of squared residuals.

    Attributes:
        slope (float): Fitted slope coefficient.
        intercept (float): Fitted intercept.
        is_trained (bool): Whether the model has been trained.
    """

    def __init__(self):
        """Initialize an untrained model."""
        self.slope = None
        self.intercept = None
        self.is_trained = False

    def fit(self, x, y):
        """
        Train the model on the given data using OLS closed-form.

        Args:
            x (list): Feature values (independent variable).
            y (list): Target values (dependent variable).

        Returns:
            LinearRegressionModel: self, for method chaining.

        Raises:
            ValueError: If data is invalid.
        """
        if not x or not y:
            raise ValueError("Training data cannot be empty.")
        if len(x) != len(y):
            raise ValueError(
                "Feature and target lists must have the same length."
            )
        if len(x) < 2:
            raise ValueError("Need at least 2 data points to fit a line.")

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum(
            (xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)
        )
        denominator = sum((xi - mean_x) ** 2 for xi in x)

        if denominator == 0:
            raise ValueError(
                "Cannot fit model: all x-values are identical."
            )

        self.slope = numerator / denominator
        self.intercept = mean_y - self.slope * mean_x
        self.is_trained = True
        return self

    def predict(self, x):
        """
        Generate predictions for a list of feature values.

        Args:
            x (list): Feature values to predict on.

        Returns:
            list[float]: Predicted y values.

        Raises:
            RuntimeError: If the model is not trained.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before predictions. Call fit() first."
            )
        return [self.slope * xi + self.intercept for xi in x]

    def predict_single(self, x):
        """
        Predict for a single value.

        Args:
            x (float): A single feature value.

        Returns:
            float: Predicted y value.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before predictions. Call fit() first."
            )
        return self.slope * x + self.intercept

    def evaluate(self, x, y):
        """
        Evaluate model with MAE, MSE, RMSE, and R-squared.

        Args:
            x (list): Feature values.
            y (list): True target values.

        Returns:
            dict: Evaluation metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation.")
        if len(x) != len(y):
            raise ValueError(
                "Feature and target lists must have the same length."
            )
        if not x:
            raise ValueError("Evaluation data cannot be empty.")

        predictions = self.predict(x)
        n = len(y)

        mae = sum(abs(yi - pi) for yi, pi in zip(y, predictions)) / n
        mse = sum((yi - pi) ** 2 for yi, pi in zip(y, predictions)) / n
        rmse = math.sqrt(mse)

        mean_y = sum(y) / n
        ss_res = sum((yi - pi) ** 2 for yi, pi in zip(y, predictions))
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "r_squared": round(r_squared, 6),
        }

    def get_params(self):
        """Return model parameters as a dictionary."""
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "is_trained": self.is_trained,
        }

    def save_model(self, filepath):
        """
        Save trained model parameters to JSON.

        Args:
            filepath (str): Path to save the model.

        Raises:
            RuntimeError: If the model is not trained.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(
                {"slope": self.slope, "intercept": self.intercept},
                f, indent=2
            )

    def load_model(self, filepath):
        """
        Load model parameters from JSON.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            LinearRegressionModel: self, for method chaining.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r") as f:
            params = json.load(f)

        self.slope = params["slope"]
        self.intercept = params["intercept"]
        self.is_trained = True
        return self
