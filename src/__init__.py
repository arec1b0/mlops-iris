"""MLOps Iris Classification API.

A production-ready machine learning API for classifying iris flowers using
scikit-learn, FastAPI, and modern MLOps practices.

This package provides:
- Secure ONNX-based model serialization
- FastAPI REST API with automatic documentation
- MLflow experiment tracking integration
- Comprehensive testing and CI/CD pipeline
- Docker containerization support

Example:
    >>> from src.api.main import app
    >>> from src.model import IrisModel
    >>>
    >>> # Load and use a trained model
    >>> model = IrisModel("artifacts/model.onnx")
    >>> model.load()
    >>> prediction, class_name = model.predict([5.1, 3.5, 1.4, 0.2])
    >>> print(f"Predicted: {class_name}")
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"
__email__ = "mlops@example.com"

from .api.main import app
from .model import IrisModel, IrisTrainer, ModelPersistence
from .config import get_settings, Settings

__all__ = [
    "app",
    "IrisModel",
    "IrisTrainer",
    "ModelPersistence",
    "get_settings",
    "Settings",
    "__version__",
]
