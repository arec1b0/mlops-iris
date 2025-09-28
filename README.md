# MLOps Iris Classification Project

A modular MLOps project for Iris flower classification using scikit-learn, FastAPI, and MLflow.

## Project Structure

```
mlops-iris/
├── src/                          # Main application code
│   ├── __init__.py              # Package initialization
│   ├── data.py                  # Data loading and preprocessing
│   ├── model.py                 # Model training and management
│   └── api/                     # FastAPI application
│       ├── __init__.py
│       └── main.py              # API endpoints
├── tests/                       # Unit and integration tests
│   ├── test_model.py            # Model tests
│   └── test_api.py              # API tests
├── artifacts/                   # Saved models and artifacts
├── mlflow/                      # MLflow tracking data
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── train.py                     # Training script entry point
├── run_api.py                   # API server entry point
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Features

- **Modular Architecture**: Clean separation of concerns with data, model, and API modules
- **MLflow Integration**: Experiment tracking and model versioning
- **FastAPI**: Modern, high-performance API with automatic documentation
- **Docker Support**: Containerized deployment
- **Comprehensive Testing**: Unit and integration tests
- **Type Hints**: Full type annotation for better code quality

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:

- Load the Iris dataset
- Train a LogisticRegression model
- Evaluate on test data
- Save the model to `artifacts/model.pkl`
- Log metrics to MLflow

### 3. Run the API

```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### 4. Make Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## API Documentation

Once the API is running, visit `http://localhost:8000/docs` for interactive Swagger documentation.

### Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make predictions

### Prediction Request Format

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Features should be in order: `[sepal_length, sepal_width, petal_length, petal_width]`

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

## MLflow Tracking

The project uses MLflow for experiment tracking:

```bash
# View MLflow UI
mlflow ui

# UI will be available at http://localhost:5000
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

- **Health Endpoint**: `GET /health` - Overall service health
- **Readiness Probe**: Container readiness for traffic
- **Liveness Probe**: Container health for restarts

### Metrics & Logging

- **MLflow Integration**: Experiment tracking and metrics
- **Structured Logging**: JSON format logs for better parsing
- **Performance Metrics**: Response times and throughput tracking

### Docker Health Checks

```bash
# Check container health
docker ps
docker inspect <container_id> | grep -A 10 "Health"

# View container logs
docker logs <container_id>
```

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

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **Iris Dataset**: UCI Machine Learning Repository
- **Scikit-learn**: Machine learning library
- **FastAPI**: Modern Python web framework
- **MLflow**: MLOps lifecycle management
