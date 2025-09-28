"""Model training and management functions for Iris classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from typing import Any, Tuple
from pathlib import Path
import onnxruntime as ort
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from .config import get_settings


class IrisModel:
    """Iris classification model for prediction only."""

    def __init__(self, model_path: str = None):
        """
        Initialize the Iris model.

        Args:
            model_path: Path to the ONNX model file (uses config default if None)
        """
        config = get_settings()
        self.model_path = Path(model_path or config.model_path)
        self.onnx_session = None
        self.target_names = ["setosa", "versicolor", "virginica"]

    def load(self) -> "IrisModel":
        """Load the ONNX model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load ONNX model
        self.onnx_session = ort.InferenceSession(str(self.model_path))
        print(f"ONNX model loaded from {self.model_path}")
        return self

    def predict(self, features: Any) -> Tuple[int, str]:
        """
        Make prediction on new data.

        Args:
            features: Input features for prediction

        Returns:
            Tuple of (prediction_class, class_name)
        """
        if self.onnx_session is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Convert features to numpy array and reshape for ONNX
        input_array = np.array([features], dtype=np.float32)

        # Run inference
        outputs = self.onnx_session.run(None, {"input": input_array})
        prediction = int(outputs[0][0])

        class_name = self.target_names[prediction]
        return prediction, class_name


class IrisTrainer:
    """Handles model training and evaluation."""

    def __init__(self):
        """Initialize the trainer."""
        self.model = None

    def train(
        self, X_train: Any, y_train: Any, max_iter: int = None
    ) -> LogisticRegression:
        """
        Train the logistic regression model.

        Args:
            X_train: Training features
            y_train: Training labels
            max_iter: Maximum iterations for convergence (uses config default if None)

        Returns:
            Trained model
        """
        config = get_settings()
        max_iterations = max_iter or config.training_max_iter
        random_state = config.training_random_state

        self.model = LogisticRegression(
            max_iter=max_iterations, random_state=random_state
        )
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate(self, model: LogisticRegression, X_test: Any, y_test: Any) -> float:
        """
        Evaluate model on test data.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score
        """
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy


class ModelPersistence:
    """Handles model saving and loading."""

    def save_onnx_model(self, model: LogisticRegression, model_path: str) -> None:
        """
        Save model in ONNX format.

        Args:
            model: Trained model
            model_path: Path to save the model
        """
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Define the input type for ONNX conversion
        initial_type = [("input", FloatTensorType([None, 4]))]

        # Convert the model to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)

        # Save the ONNX model
        with open(path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved to {path}")

    def log_to_mlflow(self, model: LogisticRegression, accuracy: float) -> str:
        """
        Log model and metrics to MLflow.

        Args:
            model: Trained model
            accuracy: Model accuracy score

        Returns:
            MLflow run ID
        """
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy)
            print(f"Model logged to MLflow with run ID: {run.info.run_id}")
            return run.info.run_id


def train_and_save_model(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    model_path: str = None,
    use_mlflow: bool = None,
) -> Tuple[float, str]:
    """
    Train and save the Iris model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_path: Path to save the model (uses config default if None)
        use_mlflow: Whether to use MLflow for logging (uses config default if None)

    Returns:
        Tuple of (accuracy, mlflow_run_id or empty string)
    """
    config = get_settings()

    # Use configuration defaults if not specified
    save_path = model_path or config.model_path
    enable_mlflow = use_mlflow if use_mlflow is not None else config.mlflow_enabled

    # Initialize components
    trainer = IrisTrainer()
    persistence = ModelPersistence()

    # Train model
    trained_model = trainer.train(X_train, y_train)

    # Evaluate
    accuracy = trainer.evaluate(trained_model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    # Log to MLflow if enabled
    run_id = ""
    if enable_mlflow:
        run_id = persistence.log_to_mlflow(trained_model, accuracy)

    # Save model as ONNX
    persistence.save_onnx_model(trained_model, save_path)

    return accuracy, run_id
