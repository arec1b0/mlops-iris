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

## Docker Deployment

### Build Image

```bash
docker build -t mlops-iris .
```

### Run Container

```bash
docker run -p 8000:80 mlops-iris
```

The API will be available at `http://localhost:8000`

## Development

### Running Tests

```bash
# Install test dependencies (if needed)
pip install pytest

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is open source and available under the MIT License.
