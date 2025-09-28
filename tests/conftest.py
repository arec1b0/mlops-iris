"""Shared test fixtures and utilities for MLOps Iris tests."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.data import load_iris_data, split_data
from src.model import IrisModel


@pytest.fixture(scope="session")
def iris_data():
    """Load Iris dataset once for all tests."""
    X, y, target_names = load_iris_data()
    return X, y, target_names


@pytest.fixture(scope="session")
def iris_data_split(iris_data):
    """Split Iris data into train/test sets."""
    X, y, _ = iris_data
    return split_data(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def trained_model(tmp_path, iris_data_split):
    """Create and return a trained Iris model."""
    X_train, X_test, y_train, y_test = iris_data_split
    model = IrisModel(tmp_path / "test_model.pkl")
    model.train(X_train, y_train)
    return model, X_test, y_test


@pytest.fixture
def model_artifacts_dir(tmp_path):
    """Create a temporary artifacts directory."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    return artifacts_dir


@pytest.fixture
def sample_iris_features():
    """Return sample Iris features for different classes."""
    return {
        "setosa": [5.1, 3.5, 1.4, 0.2],
        "versicolor": [6.0, 2.9, 4.5, 1.5],
        "virginica": [7.3, 2.9, 6.3, 1.8],
        "edge_cases": [
            [0.0, 0.0, 0.0, 0.0],  # All zeros
            [10.0, 10.0, 10.0, 10.0],  # Large values
            [0.1, 0.1, 0.1, 0.1],  # Very small values
        ],
    }


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with (
        pytest.mock.patch("mlflow.start_run") as mock_run,
        pytest.mock.patch("mlflow.log_metric") as mock_log,
        pytest.mock.patch("mlflow.sklearn.log_model") as mock_log_model,
    ):
        mock_run.return_value.__enter__.return_value = MagicMock()
        mock_run.return_value.__enter__.return_value.info.run_id = "test_run_123"

        yield {
            "start_run": mock_run,
            "log_metric": mock_log,
            "log_model": mock_log_model,
        }


@pytest.fixture
def performance_timer():
    """Fixture to measure test execution time."""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.duration = self.end - self.start

    return Timer()


# Custom markers for test categorization
def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "performance: Performance and scalability tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")


# Test data validation utilities
def validate_iris_prediction(prediction_data):
    """Validate the structure of Iris prediction response."""
    required_fields = ["prediction", "class_name", "confidence"]

    for field in required_fields:
        assert field in prediction_data, f"Missing required field: {field}"

    assert isinstance(prediction_data["prediction"], int)
    assert 0 <= prediction_data["prediction"] <= 2
    assert isinstance(prediction_data["class_name"], str)
    assert prediction_data["class_name"] in ["setosa", "versicolor", "virginica"]
    assert isinstance(prediction_data["confidence"], (int, float))


def create_test_model_file(tmp_path, model_name="test_model.pkl"):
    """Create a test model file for API testing."""
    from src.data import load_iris_data, split_data
    from src.model import IrisModel

    X, y, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    model_path = tmp_path / model_name
    model = IrisModel(str(model_path))
    model.train(X_train, y_train)
    model.save()

    return str(model_path)


# Test data constants
IRIS_FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
IRIS_CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Expected performance thresholds
MIN_MODEL_ACCURACY = 0.8
MAX_PREDICTION_TIME = 0.001  # 1ms
MAX_API_RESPONSE_TIME = 1.0  # 1 second


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    from src.api.main import app

    return TestClient(app)
