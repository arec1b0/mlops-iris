"""Integration tests for the Iris API."""

import pytest
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch

from src.api.main import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_features():
    """Sample Iris features for testing."""
    return [5.1, 3.5, 1.4, 0.2]  # Typical setosa features


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "Iris Classification API" in data["message"]
        assert "endpoints" in data

    def test_health_endpoint_no_model(self, client):
        """Test health endpoint when model is not available."""
        # Mock model loading to fail
        with patch("src.api.main.get_model", side_effect=Exception("Model not found")):
            response = client.get("/health")
            assert response.status_code == 500

    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction with invalid input."""
        # Test with wrong number of features
        response = client.post(
            "/predict", json={"features": [1, 2, 3]}
        )  # Only 3 features
        assert response.status_code == 400

        # Test with empty features
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 400

    def test_predict_endpoint_valid_input(self, client, sample_features):
        """Test prediction with valid input."""
        # Mock successful model prediction
        mock_prediction = (0, "setosa")

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = mock_prediction

            response = client.post("/predict", json={"features": sample_features})
            assert response.status_code == 200

            data = response.json()
            assert data["prediction"] == 0
            assert data["class_name"] == "setosa"
            assert "confidence" in data

    def test_predict_endpoint_model_error(self, client, sample_features):
        """Test prediction when model throws error."""
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.side_effect = Exception("Model prediction failed")

            response = client.post("/predict", json={"features": sample_features})
            assert response.status_code == 500


class TestPredictionRequest:
    """Test cases for PredictionRequest validation."""

    def test_valid_request(self, client, sample_features):
        """Test valid prediction request structure."""
        request_data = {"features": sample_features}

        # Mock successful prediction
        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")

            response = client.post("/predict", json=request_data)
            assert response.status_code == 200

    def test_request_with_extra_fields(self, client, sample_features):
        """Test request with extra fields is handled gracefully."""
        request_data = {"features": sample_features, "extra_field": "ignored"}

        with patch("src.api.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.predict.return_value = (0, "setosa")

            response = client.post("/predict", json=request_data)
            assert response.status_code == 200
