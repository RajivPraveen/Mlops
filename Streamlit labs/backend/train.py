import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data import load_data, split_data


def fit_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the wine dataset.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.

    Returns:
        RandomForestClassifier: The trained model.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def evaluate_model(model, X_test, y_test, target_names):
    """Print and save evaluation metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    metrics = {"accuracy": accuracy, "report": report}
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_model(X_train, y_train)
    joblib.dump(model, "model/wine_model.pkl")
    print("Model saved to model/wine_model.pkl")

    feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    with open("model/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    print("Feature importance saved to model/feature_importance.json")

    evaluate_model(model, X_test, y_test, target_names)
