"""Model training and management functions for Iris classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
from typing import Any, Tuple
import os
from pathlib import Path


class IrisModel:
    """Iris classification model wrapper with MLflow integration."""

    def __init__(self, model_path: str = "artifacts/model.pkl"):
        """
        Initialize the Iris model.

        Args:
            model_path: Path to save/load the model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.target_names = ["setosa", "versicolor", "virginica"]

    def train(self, X_train: Any, y_train: Any, max_iter: int = 200) -> "IrisModel":
        """
        Train the logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            max_iter: Maximum iterations for convergence

        Returns:
            Self for method chaining
        """
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test: Any, y_test: Any) -> float:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def predict(self, features: Any) -> Tuple[int, str]:
        """
        Make prediction on new data.

        Args:
            features: Input features for prediction

        Returns:
            Tuple of (prediction_class, class_name)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        prediction = self.model.predict([features])[0]
        class_name = self.target_names[prediction]
        return int(prediction), class_name

    def save(self, use_mlflow: bool = False) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Ensure artifacts directory exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if use_mlflow:
            # Log model to MLflow
            with mlflow.start_run():
                mlflow.sklearn.log_model(self.model, "model")
                print(f"Model logged to MLflow")
        else:
            # Save with joblib
            joblib.dump(self.model, self.model_path)
            print(f"Model saved to {self.model_path}")

    def load(self) -> "IrisModel":
        """Load a saved model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return self

    def log_to_mlflow(self, accuracy: float) -> str:
        """
        Log training metrics to MLflow.

        Args:
            accuracy: Model accuracy score

        Returns:
            MLflow run ID
        """
        with mlflow.start_run() as run:
            mlflow.log_metric("accuracy", accuracy)
            print(f"MLflow run ID: {run.info.run_id}")
            return run.info.run_id


def train_and_save_model(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    model_path: str = "artifacts/model.pkl",
    use_mlflow: bool = True,
) -> Tuple[float, str]:
    """
    Train and save the Iris model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_path: Path to save the model
        use_mlflow: Whether to use MLflow for logging

    Returns:
        Tuple of (accuracy, mlflow_run_id or empty string)
    """
    # Train model
    model = IrisModel(model_path)
    model.train(X_train, y_train)

    # Evaluate
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Log to MLflow if enabled
    run_id = ""
    if use_mlflow:
        run_id = model.log_to_mlflow(accuracy)

    # Save model
    model.save(use_mlflow=False)  # Always save locally for API

    return accuracy, run_id
