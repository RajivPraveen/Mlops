from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

if __name__ == '__main__':
    wine = load_wine()
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        random_state=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'wine_model.pkl')
    print("\nModel saved as wine_model.pkl")
