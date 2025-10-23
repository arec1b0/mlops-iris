"""Model training and management functions for Iris classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import mlflow.onnx
from mlflow.tracking import MlflowClient
from typing import Any, Tuple, Optional
from pathlib import Path
import onnxruntime as ort
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tempfile
import os

from .config import get_settings


class IrisModel:
    """Iris classification model for prediction only."""

    def __init__(self, model_path: str = None, use_registry: bool = None, 
                 registry_model_name: str = None, registry_stage: str = None):
        """
        Initialize the Iris model.

        Args:
            model_path: Path to the ONNX model file (uses config default if None)
            use_registry: Whether to load from MLflow Registry (uses config default if None)
            registry_model_name: Model name in MLflow Registry (uses config default if None)
            registry_stage: Registry stage to load from (uses config default if None)
        """
        config = get_settings()
        self.model_path = Path(model_path or config.model_path)
        self.onnx_session = None
        self.target_names = ["setosa", "versicolor", "virginica"]
        self.use_registry = use_registry if use_registry is not None else config.mlflow_use_registry
        self.registry_model_name = registry_model_name or config.mlflow_registry_model_name
        self.registry_stage = registry_stage or config.mlflow_registry_stage
        self.model_version = None  # Will be set when loading from registry
        self.model_source = None  # Track where model was loaded from

    def load(self) -> "IrisModel":
        """Load the ONNX model from file or MLflow Registry."""
        if self.use_registry:
            return self._load_from_registry()
        else:
            return self._load_from_file()

    def _load_from_file(self) -> "IrisModel":
        """Load the ONNX model from local file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load ONNX model
        self.onnx_session = ort.InferenceSession(str(self.model_path))
        self.model_source = f"file:{self.model_path}"
        print(f"ONNX model loaded from {self.model_path}")
        return self

    def _load_from_registry(self) -> "IrisModel":
        """Load the ONNX model from MLflow Model Registry."""
        config = get_settings()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        try:
            # Construct model URI
            model_uri = f"models:/{self.registry_model_name}/{self.registry_stage}"
            
            # Load model metadata to get version
            client = MlflowClient()
            model_version_details = client.get_latest_versions(
                self.registry_model_name, 
                stages=[self.registry_stage] if self.registry_stage != "None" else None
            )
            
            if not model_version_details:
                raise ValueError(
                    f"No model found in registry with name '{self.registry_model_name}' "
                    f"and stage '{self.registry_stage}'"
                )
            
            latest_version = model_version_details[0]
            self.model_version = latest_version.version
            
            # Download model to temp directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{latest_version.source}/model.onnx",
                    dst_path=tmp_dir
                )
                
                # Load ONNX model
                self.onnx_session = ort.InferenceSession(model_path)
                self.model_source = f"registry:{self.registry_model_name}/{self.registry_stage}/v{self.model_version}"
                print(f"ONNX model loaded from MLflow Registry: {self.model_source}")
                
            return self
            
        except Exception as e:
            print(f"Failed to load from registry: {e}")
            print(f"Falling back to local file: {self.model_path}")
            return self._load_from_file()

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
    
    def get_model_info(self) -> dict:
        """
        Get model metadata information.
        
        Returns:
            Dictionary with model version and source information
        """
        return {
            "model_source": self.model_source or "unknown",
            "model_version": self.model_version or "unknown",
            "registry_name": self.registry_model_name if self.use_registry else None,
            "registry_stage": self.registry_stage if self.use_registry else None,
        }


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

    def log_to_mlflow(self, model: LogisticRegression, accuracy: float, 
                      register_model: bool = False, model_name: str = None) -> Tuple[str, Optional[str]]:
        """
        Log model and metrics to MLflow and optionally register it.

        Args:
            model: Trained model
            accuracy: Model accuracy score
            register_model: Whether to register the model in MLflow Registry
            model_name: Name for model registration (uses config default if None)

        Returns:
            Tuple of (MLflow run ID, model version if registered else None)
        """
        config = get_settings()
        registry_name = model_name or config.mlflow_registry_model_name
        
        with mlflow.start_run() as run:
            # Log sklearn model
            mlflow.sklearn.log_model(model, "sklearn_model")
            
            # Convert and log ONNX model
            initial_type = [("input", FloatTensorType([None, 4]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                onnx_path = os.path.join(tmp_dir, "model.onnx")
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                # Log ONNX model as artifact
                mlflow.log_artifact(onnx_path, "onnx_model")
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model parameters
            mlflow.log_params({
                "model_type": "LogisticRegression",
                "max_iter": model.max_iter,
                "random_state": getattr(model, 'random_state', None)
            })
            
            run_id = run.info.run_id
            print(f"Model logged to MLflow with run ID: {run_id}")
            
            # Register model if requested
            model_version = None
            if register_model:
                try:
                    # Register the sklearn model
                    model_uri = f"runs:/{run_id}/sklearn_model"
                    mv = mlflow.register_model(model_uri, registry_name)
                    model_version = mv.version
                    print(f"Model registered as '{registry_name}' version {model_version}")
                    
                    # Add description
                    client = MlflowClient()
                    client.update_model_version(
                        name=registry_name,
                        version=model_version,
                        description=f"Iris classifier with accuracy: {accuracy:.4f}"
                    )
                except Exception as e:
                    print(f"Warning: Failed to register model: {e}")
            
            return run_id, model_version


def train_and_save_model(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    model_path: str = None,
    use_mlflow: bool = None,
    register_model: bool = False,
) -> Tuple[float, str, Optional[str]]:
    """
    Train and save the Iris model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        model_path: Path to save the model (uses config default if None)
        use_mlflow: Whether to use MLflow for logging (uses config default if None)
        register_model: Whether to register model in MLflow Registry

    Returns:
        Tuple of (accuracy, mlflow_run_id or empty string, model_version or None)
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
    model_version = None
    if enable_mlflow:
        run_id, model_version = persistence.log_to_mlflow(
            trained_model, accuracy, register_model=register_model
        )

    # Save model as ONNX
    persistence.save_onnx_model(trained_model, save_path)

    return accuracy, run_id, model_version
