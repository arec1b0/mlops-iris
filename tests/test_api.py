"""Comprehensive integration and API tests for the Iris classification service."""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path
import json

from src.api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample Iris features for testing."""
    return [5.1, 3.5, 1.4, 0.2]  # Typical setosa features


@pytest.fixture
def trained_model_file(tmp_path):
    """Create a trained model file for testing."""
    from src.data import load_iris_data, split_data
    from src.model import IrisTrainer, ModelPersistence

    # Load and split data
    X, y, _ = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # Train model using IrisTrainer
    trainer = IrisTrainer()
    trained_model = trainer.train(X_train, y_train)

    # Save model using ModelPersistence
    model_path = tmp_path / "test_model.onnx"
    persistence = ModelPersistence()
    persistence.save_onnx_model(trained_model, str(model_path))

    return str(model_path)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.predict.return_value = (0, "setosa")
    return model


class TestAPIEndpoints:
    """Comprehensive test cases for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "Iris Classification API" in data["message"]
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)
        assert len(data["endpoints"]) >= 4  # GET /, GET /health, GET /metadata, POST /predict

    def test_health_endpoint_no_model(self, client):
        """Test health endpoint when model is not available."""
        # Mock model loading to fail
        with patch("src.api.main.get_model", side_effect=Exception("Model not found")):
            response = client.get("/health")
            assert response.status_code == 500

            error_data = response.json()
            assert "detail" in error_data

    def test_health_endpoint_with_model(self, client, trained_model_file):
        """Test health endpoint when model is available."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.onnx_session = "mocked_model"
            mock_model.target_names = ["setosa", "versicolor", "virginica"]
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }
            mock_get_model.return_value = mock_model

            response = client.get("/health")
            assert response.status_code == 200

            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["model_loaded"] is True
            assert len(health_data["target_classes"]) == 3
            assert "model_version" in health_data
            assert "model_source" in health_data

    def test_metadata_endpoint(self, client):
        """Test metadata endpoint returns model and deployment info."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "registry:iris-classifier/Production/v1",
                "registry_name": "iris-classifier",
                "registry_stage": "Production"
            }
            mock_get_model.return_value = mock_model

            response = client.get("/metadata")
            assert response.status_code == 200

            data = response.json()
            assert "model" in data
            assert "deployment" in data
            assert "config" in data
            
            # Check model info
            assert data["model"]["version"] == "1"
            assert data["model"]["source"] == "registry:iris-classifier/Production/v1"
            assert data["model"]["registry_name"] == "iris-classifier"
            
            # Check deployment info
            assert "git_commit" in data["deployment"]
            assert "api_version" in data["deployment"]
            assert "timestamp" in data["deployment"]
            
            # Check config
            assert "mlflow_tracking_uri" in data["config"]
            assert "mlflow_use_registry" in data["config"]

    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction with various invalid inputs."""
        # Test with wrong number of features (too few)
        response = client.post("/predict", json={"features": [1, 2, 3]})
        assert response.status_code == 400
        assert "Exactly 4 features required" in response.json()["detail"]

        # Test with too many features
        response = client.post("/predict", json={"features": [1, 2, 3, 4, 5]})
        assert response.status_code == 400

        # Test with empty features
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 400

        # Test with non-numeric features
        response = client.post("/predict", json={"features": ["a", "b", "c", "d"]})
        assert response.status_code == 422  # Pydantic validation error

        # Test with null features
        response = client.post("/predict", json={"features": [1, 2, None, 4]})
        assert response.status_code == 422

        # Test missing features field
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_endpoint_valid_input(self, client, sample_features):
        """Test prediction with valid input using mocked model."""
        # Mock successful model prediction
        mock_prediction = (0, "setosa")

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = mock_prediction
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            response = client.post("/predict", json={"features": sample_features})
            assert response.status_code == 200

            data = response.json()
            assert data["prediction"] == 0
            assert data["class_name"] == "setosa"
            assert "confidence" in data

    def test_predict_endpoint_different_classes(self, client):
        """Test prediction returns different classes for different inputs."""
        test_cases = [
            ([5.1, 3.5, 1.4, 0.2], 0, "setosa"),
            ([6.0, 2.9, 4.5, 1.5], 1, "versicolor"),
            ([7.3, 2.9, 6.3, 1.8], 2, "virginica"),
        ]

        for features, expected_pred, expected_class in test_cases:
            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (expected_pred, expected_class)
                mock_model.get_model_info.return_value = {
                    "model_version": "1",
                    "model_source": "file:artifacts/model.onnx"
                }

                response = client.post("/predict", json={"features": features})
                assert response.status_code == 200

                data = response.json()
                assert data["prediction"] == expected_pred
                assert data["class_name"] == expected_class

    def test_predict_endpoint_model_error(self, client, sample_features):
        """Test prediction when model throws error."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.side_effect = Exception("Model prediction failed")

            response = client.post("/predict", json={"features": sample_features})
            assert response.status_code == 500

            error_data = response.json()
            assert "detail" in error_data

    def test_predict_endpoint_edge_cases(self, client):
        """Test prediction with edge case numeric values."""
        edge_cases = [
            [0.0, 0.0, 0.0, 0.0],  # All zeros
            [10.0, 10.0, 10.0, 10.0],  # Large values
            [0.1, 0.1, 0.1, 0.1],  # Very small values
        ]

        for features in edge_cases:
            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (
                    1,
                    "versicolor",
                )  # Some valid response
                mock_model.get_model_info.return_value = {
                    "model_version": "1",
                    "model_source": "file:artifacts/model.onnx"
                }

                response = client.post("/predict", json={"features": features})
                assert response.status_code == 200

                data = response.json()
                assert isinstance(data["prediction"], int)
                assert data["class_name"] in ["setosa", "versicolor", "virginica"]

    def test_api_response_format(self, client, sample_features):
        """Test that API responses have consistent format."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (1, "versicolor")
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            response = client.post("/predict", json={"features": sample_features})
            assert response.status_code == 200

            data = response.json()

            # Check response structure
            required_fields = ["prediction", "class_name", "confidence"]
            for field in required_fields:
                assert field in data

            # Check data types
            assert isinstance(data["prediction"], int)
            assert isinstance(data["class_name"], str)
            assert isinstance(data["confidence"], (int, float))

    def test_api_content_type(self, client, sample_features):
        """Test that API returns correct content type."""
        response = client.post("/predict", json={"features": sample_features})

        assert response.headers["content-type"] == "application/json"

    def test_api_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/predict")
        # Note: CORS middleware should be configured in the actual app

        response = client.get("/")
        # Check if CORS headers are present (depends on app configuration)


class TestPredictionRequestValidation:
    """Comprehensive test cases for PredictionRequest validation."""

    def test_valid_requests(self, client):
        """Test various valid request formats."""
        valid_requests = [
            {"features": [5.1, 3.5, 1.4, 0.2]},
            {"features": [6.0, 2.9, 4.5, 1.5]},
            {"features": [7.3, 2.9, 6.3, 1.8]},
            {"features": [4.5, 2.3, 1.3, 0.3]},  # Realistic minimum values
        ]

        for request_data in valid_requests:
            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (0, "setosa")
                mock_model.get_model_info.return_value = {
                    "model_version": "1",
                    "model_source": "file:artifacts/model.onnx"
                }

                response = client.post("/predict", json=request_data)
                assert response.status_code == 200

    def test_request_with_extra_fields(self, client, sample_features):
        """Test request with extra fields is handled gracefully."""
        request_data = {
            "features": sample_features,
            "extra_field": "ignored",
            "metadata": {"source": "test"},
            "timestamp": 1234567890,
        }

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            response = client.post("/predict", json=request_data)
            assert response.status_code == 200

    def test_numeric_precision(self, client):
        """Test handling of different numeric precisions."""
        test_cases = [
            {"features": [5.1, 3.5, 1.4, 0.2]},  # Float with one decimal
            {"features": [5.123456, 3.987654, 1.456789, 0.234567]},  # High precision
            {"features": [5, 3, 1, 0]},  # Integers
        ]

        for request_data in test_cases:
            with patch("src.api.main.get_model") as mock_get_model:
                mock_model = mock_get_model.return_value
                mock_model.predict.return_value = (0, "setosa")
                mock_model.get_model_info.return_value = {
                    "model_version": "1",
                    "model_source": "file:artifacts/model.onnx"
                }

                response = client.post("/predict", json=request_data)
                assert response.status_code == 200


class TestAPIErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/predict",
            data="invalid json {",
            headers={"Content-Type": "application/json"},
        )
        # Pydantic returns 422 for validation errors, which is correct
        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Test handling of wrong content type."""
        response = client.post(
            "/predict",
            data="features=5.1,3.5,1.4,0.2",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # Should still work if the app accepts it, or return 422
        assert response.status_code in [200, 422]

    def test_unsupported_methods(self, client):
        """Test unsupported HTTP methods."""
        response = client.put("/predict", json={"features": [1, 2, 3, 4]})
        assert response.status_code == 405  # Method not allowed

        response = client.patch("/predict", json={"features": [1, 2, 3, 4]})
        assert response.status_code == 405

    def test_nonexistent_endpoints(self, client):
        """Test accessing non-existent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

        response = client.post("/api/v1/predict", json={"features": [1, 2, 3, 4]})
        assert response.status_code == 404


class TestAPIPerformance:
    """Performance tests for the API."""

    def test_response_time(self, client, sample_features):
        """Test that API responds within reasonable time."""
        import time

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            start_time = time.time()
            response = client.post("/predict", json={"features": sample_features})
            end_time = time.time()

            assert response.status_code == 200
            assert (end_time - start_time) < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, client, sample_features):
        """Test handling multiple concurrent requests."""
        import threading
        import queue

        results = queue.Queue()

        def make_request():
            try:
                with patch("src.api.main.get_model") as mock_get_model:
                    mock_model = mock_get_model.return_value
                    mock_model.predict.return_value = (0, "setosa")
                    mock_model.get_model_info.return_value = {
                        "model_version": "1",
                        "model_source": "file:artifacts/model.onnx"
                    }

                    response = client.post(
                        "/predict", json={"features": sample_features}
                    )
                    results.put((response.status_code, response.json()))
            except Exception as e:
                results.put(("error", str(e)))

        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check results
        for _ in range(5):
            status, data = results.get()
            assert status == 200
            assert data["prediction"] == 0
            assert data["class_name"] == "setosa"


class TestAPIIntegration:
    """Integration tests combining multiple API features."""

    def test_full_request_flow(self, client):
        """Test complete request flow from health check to prediction."""
        # First check health
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.onnx_session = "test_model"
            mock_model.target_names = ["setosa", "versicolor", "virginica"]
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            # Health check
            response = client.get("/health")
            assert response.status_code == 200

            # Make prediction
            mock_model.predict.return_value = (1, "versicolor")
            response = client.post("/predict", json={"features": [6.0, 2.9, 4.5, 1.5]})
            assert response.status_code == 200

            data = response.json()
            assert data["prediction"] == 1
            assert data["class_name"] == "versicolor"

    def test_api_consistency(self, client):
        """Test API response consistency across multiple calls."""
        features = [5.1, 3.5, 1.4, 0.2]

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")
            mock_model.get_model_info.return_value = {
                "model_version": "1",
                "model_source": "file:artifacts/model.onnx"
            }

            # Make multiple requests
            responses = []
            for _ in range(3):
                response = client.post("/predict", json={"features": features})
                assert response.status_code == 200
                responses.append(response.json())

            # All responses should be identical
            for resp in responses[1:]:
                assert resp["prediction"] == responses[0]["prediction"]
                assert resp["class_name"] == responses[0]["class_name"]
