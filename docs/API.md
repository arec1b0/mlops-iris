# API Documentation

## Overview

The MLOps Iris Classification API provides a RESTful interface for making predictions on iris flower features. The API is built with FastAPI and provides automatic OpenAPI/Swagger documentation.

## OpenAPI Specification

When the API is running, you can access the interactive documentation at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Endpoints

### Health Check

**GET** `/health`

Returns the current health status of the API service.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "target_classes": ["setosa", "versicolor", "virginica"],
  "model_version": "1",
  "model_source": "registry:iris-classifier/Production/v1"
}
```

**Status Codes:**

- `200`: Service is healthy
- `500`: Service is unhealthy or model failed to load

### Model Metadata

**GET** `/metadata`

Returns detailed metadata about the currently deployed model, including version, source, and deployment information. This endpoint is essential for monitoring and debugging in production.

**Response:**

```json
{
  "model": {
    "version": "1",
    "source": "registry:iris-classifier/Production/v1",
    "registry_name": "iris-classifier",
    "registry_stage": "Production"
  },
  "deployment": {
    "git_commit": "abc123def456...",
    "api_version": "1.0.0",
    "timestamp": "2025-10-23T12:34:56.789012"
  },
  "config": {
    "mlflow_tracking_uri": "sqlite:///mlruns.db",
    "mlflow_use_registry": true
  }
}
```

**Use Cases:**

- Verify which model version is currently serving
- Track deployments and correlate with git commits
- Monitor model source (file vs registry)
- Debug production issues

**Status Codes:**

- `200`: Metadata retrieved successfully
- `500`: Model not loaded or metadata unavailable

### Root Information

**GET** `/`

Returns basic information about the API.

**Response:**

```json
{
  "message": "Iris Classification API",
  "version": "1.0.0",
  "endpoints": {
    "GET /": "This information",
    "GET /health": "Health check",
    "GET /metadata": "Model metadata",
    "POST /predict": "Make prediction"
  }
}
```

### Prediction

**POST** `/predict`

Makes a prediction on iris flower features.

**Request Body:**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Feature Requirements:**

- Exactly 4 features required
- Features in order: `[sepal_length, sepal_width, petal_length, petal_width]`
- All values must be numeric (float)
- No null or missing values allowed

**Response:**

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "confidence": 0.0
}
```

**Class Mapping:**

- `0` → `"setosa"`
- `1` → `"versicolor"`
- `2` → `"virginica"`

**Status Codes:**

- `200`: Successful prediction
- `400`: Invalid request format or missing features
- `422`: Validation error (non-numeric values, etc.)
- `500`: Internal server error
- `503`: Service temporarily unavailable

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get model metadata
curl http://localhost:8000/metadata

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
health = response.json()
print(f"Status: {health['status']}")

# Make prediction
data = {"features": [5.1, 3.5, 1.4, 0.2]}
response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
print(f"Predicted: {prediction['class_name']}")
```

### Using the API Client

```python
from examples.api_client import IrisAPIClient

client = IrisAPIClient("http://localhost:8000")

# Check health
health = client.check_health()
print(f"Model loaded: {health['model_loaded']}")

# Make prediction
result = client.predict_single([5.1, 3.5, 1.4, 0.2])
print(f"Prediction: {result['class_name']}")
```

## Error Handling

The API provides detailed error messages for different failure scenarios:

### Invalid Features (400)

```json
{
  "detail": "Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]"
}
```

### Non-numeric Values (422)

```json
{
  "detail": "All features must be numeric values"
}
```

### Service Unavailable (503)

```json
{
  "detail": "Model service temporarily unavailable. Please try again later."
}
```

## Rate Limiting

Currently, there are no explicit rate limits implemented. However, the API is designed to handle reasonable request volumes. For production deployments, consider implementing rate limiting based on your specific requirements.

## Security Considerations

- All model files are loaded securely using ONNX runtime
- Input validation prevents malicious payloads
- No sensitive data is logged in production
- The API runs in a containerized environment with security best practices

## Monitoring and Observability

The API includes comprehensive monitoring capabilities:

### Endpoints for Monitoring

- **Health endpoint**: `/health` - Service health and model status
- **Metadata endpoint**: `/metadata` - Model version and deployment info

### Structured Logging

All predictions are logged in JSON format with:
- Input features
- Predictions
- Model version
- Timestamps

Example log entry:
```json
{
  "timestamp": "2025-10-23T12:34:56.789Z",
  "level": "INFO",
  "logger": "iris_api",
  "message": "Prediction made",
  "event": "prediction",
  "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
  "prediction": {"class_id": 0, "class_name": "setosa"},
  "model": {"version": "1", "source": "registry:iris-classifier/Production/v1"}
}
```

### Integration with Monitoring Systems

- **Loki + Grafana**: Parse JSON logs for dashboards and alerts
- **Elasticsearch + Kibana**: Index logs for search and visualization
- **MLflow**: Track model performance and experiments
- **Prometheus**: Expose metrics via custom exporters

For detailed monitoring setup, see [docs/MLOPS.md](MLOPS.md)

## Development

To contribute to the API:

1. **Add new endpoints** in `src/api/main.py`
2. **Update models** in `src/api/main.py` or create new response models
3. **Add tests** in `tests/test_api.py`
4. **Update documentation** in this file and the README

The OpenAPI specification will be automatically updated when you restart the server.
