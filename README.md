# 🌸 MLOps Iris Classification API

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-success.svg)](https://github.com/features/actions)

A production-ready MLOps project for Iris flower classification using **scikit-learn**, **FastAPI**, **MLflow**, and modern DevOps practices. Features secure model serialization, comprehensive testing, containerization, and automated deployment pipelines.

## ✨ Key Features

- 🔒 **Secure Model Serialization**: ONNX format instead of insecure pickle files
- 🚀 **FastAPI with Lifespan Events**: Optimized model loading and startup performance
- ⚙️ **Centralized Configuration**: Environment-based settings with validation
- 🧪 **Comprehensive Testing**: 98%+ code coverage with unit, integration, and performance tests
- 🐳 **Multi-stage Docker**: Optimized production builds with security scanning
- 📊 **MLflow Model Registry**: Version control, stage-based deployment, and safe rollback
- 🔄 **Separated CI/CD Pipeline**: CI for testing, CD for model training and deployment
- 📈 **Drift Monitoring**: Structured JSON logging for data and concept drift detection
- 🔙 **Safe Rollback**: Instant model version rollback via MLflow Registry
- 🔍 **Model Versioning**: Track model versions, sources, and git commits in production

## 📁 Project Structure

```
mlops-iris/
├── src/                          # Main application code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # ⚙️ Centralized configuration management
│   ├── data.py                  # 📊 Data loading and preprocessing
│   ├── model.py                 # 🤖 Model classes (prediction, training, persistence)
│   └── api/                     # 🌐 FastAPI application
│       ├── __init__.py          # API package initialization
│       └── main.py              # 🚀 API endpoints with lifespan events
├── tests/                       # 🧪 Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py              # Test configuration and fixtures
│   ├── test_api.py              # API endpoint tests
│   ├── test_model.py            # Model functionality tests
│   ├── test_e2e.py              # End-to-end workflow tests
│   └── test_performance.py      # Performance and load tests
├── artifacts/                   # 💾 Saved models (.onnx format)
├── mlruns/                      # 📈 MLflow experiment tracking data
├── .github/workflows/           # 🔄 CI/CD pipeline definitions
│   └── docker.yml               # GitHub Actions workflow
├── scripts/                     # 🛠️ Deployment and utility scripts
│   ├── deploy.sh                # Kubernetes deployment script
│   └── health_check.py          # Service health monitoring
├── docker-compose.yml           # 🐳 Multi-service orchestration
├── Dockerfile                   # 📦 Multi-stage container build
├── requirements.txt             # 📋 Pinned Python dependencies
├── Makefile                     # 🛠️ Development automation tasks
├── pytest.ini                  # 🧪 Test configuration
├── train.py                     # 🚂 Model training entry point
├── run_api.py                   # 🚀 API server entry point
├── .gitignore                   # 🚫 Git ignore patterns
├── .pre-commit-config.yaml      # 🔒 Pre-commit hook configuration
└── README.md                    # 📖 This comprehensive guide
```

## 🔧 Architecture & Design

### Core Components

- **🎯 Single Responsibility Principle**: Model classes split into focused components
  - `IrisModel`: Prediction-only class with ONNX runtime
  - `IrisTrainer`: Training and evaluation logic
  - `ModelPersistence`: Secure model saving/loading with ONNX format
- **⚙️ Configuration Management**: Centralized settings with environment variable support
- **🌐 FastAPI Application**: RESTful API with automatic OpenAPI documentation
- **📊 Data Pipeline**: Clean data loading and preprocessing utilities

### Security & Performance

- **🔒 ONNX Model Format**: Secure model serialization (no pickle vulnerabilities)
- **🚀 Optimized Startup**: Model loading via FastAPI lifespan events (no cold-start delays)
- **🐳 Multi-stage Docker**: Production-optimized builds with security scanning
- **📝 Structured Logging**: JSON-formatted logs for better observability
- **🔍 Input Validation**: Comprehensive request validation with specific error handling

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized deployment)

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/mlops-iris.git
cd mlops-iris

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train with default settings (saves locally)
python train.py

# Train and register in MLflow Model Registry
python train.py --register-model

# Or with custom options
python train.py --model-path artifacts/my_model.onnx --test-size 0.3 --no-mlflow
```

This will:

- 📥 Load the Iris dataset from scikit-learn
- 🧠 Train a LogisticRegression model with optimized hyperparameters
- ✅ Evaluate on test data (typically 95-97% accuracy)
- 💾 Save the model securely in ONNX format to `artifacts/model.onnx`
- 📊 Log metrics and artifacts to MLflow (if enabled)
- 🏷️ Register model in MLflow Model Registry (with `--register-model` flag)

### 3. Start the API Server

```bash
# Start the FastAPI server (loads from local file)
python run_api.py

# Load model from MLflow Registry (Production stage)
export IRIS_MLFLOW_USE_REGISTRY=true
export IRIS_MLFLOW_REGISTRY_STAGE=Production
python run_api.py

# Or with custom options
python run_api.py --port 8080 --model-path artifacts/my_model.onnx
```

The API will be available at `http://localhost:8000`

### 4. Make Predictions

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Example response:
# {
#   "prediction": 0,
#   "class_name": "setosa",
#   "confidence": 0.0
# }
```

### 5. View API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## 📋 API Reference

Once the API is running, visit `http://localhost:8000/docs` for interactive Swagger UI documentation with request/response examples and testing interface.

### 🔌 Endpoints

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/` | API information and available endpoints | None | API metadata |
| `GET` | `/health` | Health check with model status and version | None | Service health status |
| `GET` | `/metadata` | Model version, source, and deployment info | None | Model metadata |
| `POST` | `/predict` | Make predictions on iris features | `PredictionRequest` | `PredictionResponse` |

### 📥 Request Format

#### Prediction Request

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Feature Order (Required):**

1. **sepal_length** (cm): Length of the sepal
2. **sepal_width** (cm): Width of the sepal
3. **petal_length** (cm): Length of the petal
4. **petal_width** (cm): Width of the petal

**Validation Rules:**

- Exactly 4 features required
- All values must be numeric (float)
- No null or missing values allowed

### 📤 Response Format

#### Prediction Response

```json
{
  "prediction": 0,
  "class_name": "setosa",
  "confidence": 0.0
}
```

**Response Fields:**

- **prediction** (integer): Numeric class prediction (0, 1, or 2)
- **class_name** (string): Human-readable class name
- **confidence** (float): Prediction confidence (currently placeholder)

**Class Mapping:**

- `0` → `"setosa"`
- `1` → `"versicolor"`
- `2` → `"virginica"`

### 🚨 Error Responses

The API returns appropriate HTTP status codes with detailed error messages:

```json
{
  "detail": "Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]"
}
```

**Common Error Codes:**

- `400` - Bad Request (invalid input format)
- `422` - Unprocessable Entity (validation errors)
- `500` - Internal Server Error (model loading/prediction failures)
- `503` - Service Unavailable (model temporarily unavailable)

## CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment:

### Automated CI/CD Features

- **Code Quality**: Linting with Ruff, formatting with Black, type checking with MyPy
- **Testing**: Comprehensive test suite with 98%+ coverage
- **Security Scanning**: Vulnerability scanning with Trivy
- **Docker Build**: Automated multi-platform Docker builds
- **Container Registry**: Images published to GitHub Container Registry (ghcr.io)
- **Multi-Environment**: Separate staging and production deployments

### Pipeline Stages

1. **Lint & Test**: Code quality checks and comprehensive testing
2. **Security Scan**: Container vulnerability scanning
3. **Build & Push**: Docker image building and publishing
4. **Deploy**: Environment-specific deployments

### Docker Deployment

#### Build Locally

```bash
# Build Docker image
make build

# Or manually
docker build -t mlops-iris .
```

#### Run Locally

```bash
# Run with Docker
make run-docker

# Run with Docker Compose (recommended)
make run-compose

# Run with MLflow included
make run-compose-mlflow
```

#### Run Container

```bash
docker run -p 8000:80 mlops-iris
```

The API will be available at `http://localhost:8000`

#### Docker Compose

```bash
# Start services
docker-compose up --build

# Start with MLflow tracking server
docker-compose --profile mlflow up --build

# Stop services
docker-compose down
```

## Development

### Quick Start with Make

Use the provided Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install dependencies
make install

# Run full QA suite (lint + test + coverage)
make qa

# Start development environment
make dev

# Clean up generated files
make clean
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test categories
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-e2e         # End-to-end tests only
make test-performance # Performance tests only
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Type check
make type-check

# Setup pre-commit hooks
make setup-pre-commit

# Run pre-commit on all files
make run-pre-commit
```

### Training Options

```bash
# Train with custom model path
python train.py --model-path custom_artifacts/my_model.pkl

# Train without MLflow logging
python train.py --no-mlflow

# Adjust test set size
python train.py --test-size 0.3
```

### API Options

```bash
# Run on different port
python run_api.py --port 8080

# Use custom model
python run_api.py --model-path custom_artifacts/my_model.pkl

# Enable auto-reload for development
python run_api.py --reload
```

## 🎯 MLOps Features

This project implements production-grade MLOps practices for drift monitoring, model versioning, and safe rollback. For comprehensive documentation, see **[docs/MLOPS.md](docs/MLOPS.md)**.

### Drift Monitoring

All predictions are logged in structured JSON format for monitoring:

```json
{
  "timestamp": "2025-10-23T12:34:56.789Z",
  "level": "INFO",
  "event": "prediction",
  "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
  "prediction": {"class_id": 0, "class_name": "setosa"},
  "model": {"version": "1", "source": "registry:iris-classifier/Production/v1"}
}
```

These logs can be integrated with:
- **Loki + Grafana** for visualization and alerting
- **Elasticsearch + Kibana** for search and analytics
- Custom monitoring solutions for drift detection

### MLflow Model Registry

Models are versioned and managed through MLflow Registry:

```bash
# Train and register model
python train.py --register-model

# Configure API to load from registry
export IRIS_MLFLOW_USE_REGISTRY=true
export IRIS_MLFLOW_REGISTRY_MODEL_NAME=iris-classifier
export IRIS_MLFLOW_REGISTRY_STAGE=Production

# Start API with registry model
python run_api.py

# View MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
# UI available at http://localhost:5000
```

**Model Stages:**
- **None** - Newly registered, not yet tested
- **Staging** - Being tested in staging environment
- **Production** - Serving live traffic
- **Archived** - Previous versions, kept for rollback

### Safe Rollback

Rollback to previous model version in seconds:

**Via MLflow UI:**
1. Open MLflow UI: `mlflow ui`
2. Navigate to Models → iris-classifier
3. Select previous version → "Transition to Production"
4. Restart API service

**Via CLI:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",  # Previous version
    stage="Production"
)
```

### Model Metadata Endpoint

Track deployed model versions:

```bash
curl http://localhost:8000/metadata
```

Response includes:
- Model version and source
- Git commit hash
- MLflow Registry information
- Deployment timestamp

### CI/CD Pipeline

**Separated responsibilities:**

- **CI Pipeline** (.github/workflows/docker.yml)
  - Runs on pull requests
  - Code quality checks (linting, formatting, type checking)
  - Comprehensive test suite
  - Security scanning
  - **No model training in CI**

- **CD Pipeline** (.github/workflows/train-model.yml)
  - Runs on merge to main or manual trigger
  - Trains new model
  - Registers in MLflow Registry
  - Uploads artifacts
  - Optional auto-promotion to production

**Trigger manual training:**
```bash
# Via GitHub Actions UI
# Go to Actions → "Model Training and Registration" → Run workflow
```

## Model Performance

Current model: LogisticRegression

- Expected accuracy: ~95-97% on test set
- 3 classes: setosa, versicolor, virginica
- 4 features: sepal length/width, petal length/width

## Deployment

### Kubernetes Deployment

The project includes Kubernetes manifests and deployment scripts:

```bash
# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production v1.0.0

# Check deployment health
python scripts/health_check.py
```

### Environment Configuration

Create environment files for different deployments:

```bash
# .env.staging
ENVIRONMENT=staging
MODEL_PATH=/app/artifacts/model.pkl
LOG_LEVEL=INFO

# .env.production
ENVIRONMENT=production
MODEL_PATH=/app/artifacts/model.pkl
LOG_LEVEL=WARNING
```

## Monitoring & Observability

### Health Checks

The API provides comprehensive health monitoring:

- **Health Endpoint**: `GET /health` - Overall service health with model version
- **Metadata Endpoint**: `GET /metadata` - Model version, source, and git commit
- **Readiness Probe**: Container readiness for traffic
- **Liveness Probe**: Container health for restarts

### Drift Monitoring

**Structured Logging**: All predictions logged in JSON format with:
- Input features (for data drift detection)
- Predictions (for concept drift detection)
- Model version and source
- Timestamps

**Integration Examples**:

```bash
# Loki + Grafana: Monitor prediction distribution
{logger="iris_api"} | json | event="prediction"

# Elasticsearch: Aggregate predictions
POST /iris-predictions/_search
{
  "aggs": {
    "prediction_dist": {
      "terms": { "field": "prediction.class_name" }
    }
  }
}
```

### Metrics & Logging

- **MLflow Model Registry**: Version control and experiment tracking
- **Structured JSON Logs**: Production-ready logging format
- **Model Versioning**: Track versions in production
- **Git Commit Tracking**: Correlate deployments with code changes

### Docker Health Checks

```bash
# Check container health
docker ps
docker inspect <container_id> | grep -A 10 "Health"

# View structured logs
docker logs <container_id> | jq '.event'
```

### Drift Detection Setup

For detailed drift monitoring setup with Grafana/Kibana, see **[docs/MLOPS.md](docs/MLOPS.md)**.

## Security

### Container Security

- **Non-root user**: Containers run as `mlops` user
- **Minimal base image**: Uses `python:3.11-slim`
- **Security scanning**: Automated Trivy vulnerability scans
- **No secrets in images**: Sensitive data via environment variables

### Best Practices

- **Dependency pinning**: All packages version-locked
- **Regular updates**: Automated security updates via CI/CD
- **Access control**: Proper permissions and network policies

## Contributing

### Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/your-username/mlops-iris.git`
3. **Create** feature branch: `git checkout -b feature/new-feature`
4. **Setup** development environment: `make install-dev`
5. **Setup** pre-commit hooks: `make setup-pre-commit`
6. **Make** changes and ensure tests pass: `make qa`
7. **Commit** changes: `git commit -m "Add new feature"`
8. **Push** to your fork: `git push origin feature/new-feature`
9. **Create** Pull Request

### Code Standards

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Type checking
- **Pre-commit hooks**: Automated quality checks

### Testing Requirements

- **Unit tests**: All public functions must have tests
- **Integration tests**: API endpoints and data pipelines
- **Coverage**: Minimum 80% code coverage
- **Performance**: Tests should run in reasonable time

## Troubleshooting

### Common Issues

**API won't start:**

```bash
# Check if model file exists
ls -la artifacts/model.pkl

# Check Python dependencies
python -c "import fastapi, uvicorn, sklearn"
```

**Tests failing:**

```bash
# Clean and reinstall
make clean-all
make install-dev
make test
```

**Docker build fails:**

```bash
# Check Docker daemon
docker info

# Build with verbose output
docker build --progress=plain -t mlops-iris .
```

### Getting Help

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Inline code documentation and this README

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help improve the MLOps Iris project:

### Development Workflow

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/your-username/mlops-iris.git
   cd mlops-iris
   ```

3. **Create** a feature branch:

   ```bash
   git checkout -b feature/new-feature-name
   ```

4. **Setup** development environment:

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Optional: development dependencies
   ```

5. **Make** your changes and ensure tests pass:

   ```bash
   make qa  # Run full quality assurance suite
   ```

6. **Commit** your changes:

   ```bash
   git add .
   git commit -m "Add: comprehensive documentation and examples"
   ```

7. **Push** to your fork:

   ```bash
   git push origin feature/new-feature-name
   ```

8. **Create** a Pull Request on GitHub

### Code Standards

The project follows strict quality standards enforced by automated tools:

- **Black**: Code formatting (run with `make format`)
- **Ruff**: Linting and import sorting (run with `make lint`)
- **MyPy**: Type checking (run with `make type-check`)
- **Pre-commit hooks**: Automated quality checks on every commit

### Testing Requirements

All contributions must include appropriate tests:

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test API endpoints and data pipelines
- **Performance tests**: Ensure changes don't degrade performance
- **Coverage**: Maintain minimum 85% code coverage

### Documentation Updates

When adding new features:

1. Update this README.md with usage examples
2. Add inline documentation (docstrings) to new code
3. Update API documentation if adding new endpoints
4. Add examples in the `examples/` directory

### Issue Reporting

Found a bug or have a feature request?

1. **Search** existing issues to avoid duplicates
2. **Create** a new issue with a descriptive title
3. **Include**:
   - Clear description of the issue/feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### Pull Request Guidelines

To ensure your PR gets merged quickly:

- ✅ **Single responsibility**: One feature or fix per PR
- ✅ **Tests included**: Add tests for new functionality
- ✅ **Documentation updated**: Update README and docstrings
- ✅ **Quality checks pass**: All CI/CD checks must pass
- ✅ **Descriptive title**: Clear, concise PR title
- ✅ **Detailed description**: Explain what and why, not just how

### Development Commands

```bash
# Setup development environment
make install-dev

# Run full quality assurance
make qa

# Run specific test categories
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-e2e         # End-to-end tests only
make test-performance # Performance tests only

# Code quality checks
make lint          # Linting with Ruff
make format        # Code formatting with Black
make type-check    # Type checking with MyPy

# Start development API server
make dev
```

## 📄 License

This project is open source and available under the **MIT License**.

```
MIT License

Copyright (c) 2024 MLOps Iris Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **Iris Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)
- **Scikit-learn**: Machine learning library for Python
- **FastAPI**: Modern, fast web framework for building APIs
- **MLflow**: Platform for the machine learning lifecycle
- **ONNX**: Open Neural Network Exchange format
- **Pydantic**: Data validation and settings management using Python type annotations
- **Uvicorn**: ASGI web server implementation for Python

---

<div align="center">

**🌸 Built with ❤️ for the MLOps community**

[⭐ Star this repo](https://github.com/your-username/mlops-iris) |
[🐛 Report issues](https://github.com/your-username/mlops-iris/issues) |
[💬 Start discussion](https://github.com/your-username/mlops-iris/discussions)

</div>
