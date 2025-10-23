# MLOps Enhancements Guide

This document describes the MLOps enhancements implemented in the Iris Classification system, covering drift monitoring, model versioning with MLflow Registry, and safe rollback procedures.

## Table of Contents

1. [Drift Monitoring and Detection](#drift-monitoring-and-detection)
2. [MLflow Model Registry](#mlflow-model-registry)
3. [Safe Rollback Procedures](#safe-rollback-procedures)
4. [CI/CD Pipeline](#cicd-pipeline)
5. [Configuration](#configuration)

---

## Drift Monitoring and Detection

### Overview

The system now implements comprehensive monitoring capabilities to detect data drift and concept drift in production.

### Structured Logging

All predictions are logged in structured JSON format with the following information:

```json
{
  "timestamp": "2025-10-23T12:34:56.789Z",
  "level": "INFO",
  "logger": "iris_api",
  "message": "Prediction made",
  "event": "prediction",
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "prediction": {
    "class_id": 0,
    "class_name": "setosa"
  },
  "model": {
    "version": "1",
    "source": "registry:iris-classifier/Production/v1"
  }
}
```

### Integration with Monitoring Systems

These structured logs can be integrated with:

**Loki + Grafana:**
```bash
# Example LogQL query to monitor feature distribution
{logger="iris_api"} | json | event="prediction" | line_format "{{.features.sepal_length}}"
```

**Elasticsearch + Kibana:**
```json
POST /iris-predictions/_search
{
  "aggs": {
    "prediction_distribution": {
      "terms": { "field": "prediction.class_name" }
    }
  }
}
```

### Metadata Endpoint

The `/metadata` endpoint provides detailed information about the deployed model:

```bash
curl http://localhost:80/metadata
```

Response:
```json
{
  "model": {
    "version": "1",
    "source": "registry:iris-classifier/Production/v1",
    "registry_name": "iris-classifier",
    "registry_stage": "Production"
  },
  "deployment": {
    "git_commit": "abc123...",
    "api_version": "1.0.0",
    "timestamp": "2025-10-23T12:34:56.789Z"
  },
  "config": {
    "mlflow_tracking_uri": "sqlite:///mlruns.db",
    "mlflow_use_registry": true
  }
}
```

### Enhanced Health Check

The `/health` endpoint now includes model version information:

```bash
curl http://localhost:80/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "target_classes": ["setosa", "versicolor", "virginica"],
  "model_version": "1",
  "model_source": "registry:iris-classifier/Production/v1"
}
```

---

## MLflow Model Registry

### Overview

The system now uses MLflow Model Registry for model versioning, enabling:
- Centralized model storage
- Version control
- Stage-based deployment (None → Staging → Production)
- Safe rollback capabilities

### Training with Registry

Train and register a model:

```bash
# Train and register model
python train.py --register-model

# Train without registration (local file only)
python train.py
```

### Model Stages

Models progress through these stages:

1. **None** - Newly registered, not yet tested
2. **Staging** - Being tested in staging environment
3. **Production** - Serving live traffic
4. **Archived** - Previous versions, kept for rollback

### Loading Models from Registry

#### Via Configuration

Set environment variables:

```bash
export IRIS_MLFLOW_USE_REGISTRY=true
export IRIS_MLFLOW_REGISTRY_MODEL_NAME=iris-classifier
export IRIS_MLFLOW_REGISTRY_STAGE=Production
export IRIS_MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

#### Via Code

```python
from src.model import IrisModel

# Load from registry
model = IrisModel(
    use_registry=True,
    registry_model_name="iris-classifier",
    registry_stage="Production"
)
model.load()

# Load from local file
model = IrisModel(model_path="artifacts/model.onnx")
model.load()
```

### Programmatic Model Management

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all versions
versions = client.search_model_versions("name='iris-classifier'")

# Get latest production version
prod_versions = client.get_latest_versions(
    "iris-classifier", 
    stages=["Production"]
)

# Promote model to production
client.transition_model_version_stage(
    name="iris-classifier",
    version="2",
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",
    stage="Archived"
)
```

---

## Safe Rollback Procedures

### Quick Rollback via MLflow UI

1. Open MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlruns.db
   ```

2. Navigate to Models → iris-classifier
3. Find the previous production version
4. Click "Stage" → "Transition to Production"
5. Restart the API service (it will load the new production model)

### Rollback via CLI

```bash
# Install MLflow client
pip install mlflow

# Run rollback script
python << 'EOF'
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Rollback to version 1
client.transition_model_version_stage(
    name="iris-classifier",
    version="1",
    stage="Production"
)

print("✅ Rolled back to version 1")
EOF

# Restart API to load the rolled-back model
docker-compose restart api
```

### Rollback via GitHub Actions

The model training workflow includes rollback instructions in the promotion comment. You can also create a manual rollback workflow.

### Zero-Downtime Rollback

For production systems:

1. Update the model stage in MLflow Registry
2. The API can be configured to periodically check for model updates
3. Implement a blue-green deployment strategy
4. Use Kubernetes readiness probes with `/health` endpoint

---

## CI/CD Pipeline

### Overview

The CI/CD pipeline is now separated into distinct responsibilities:

- **CI (Pull Requests)**: Code quality, tests, security scans
- **CD (Merge to main)**: Model training, registration, deployment

### CI Pipeline (.github/workflows/docker.yml)

Runs on every pull request:

```yaml
- Lint and format checks (Ruff, Black, MyPy)
- Unit and integration tests
- Security scanning (Trivy)
- Docker build and test
```

**Key Change**: Model training removed from CI. Tests should use a pre-trained golden model.

### CD Pipeline (.github/workflows/train-model.yml)

Runs on:
- Push to main (automatic)
- Manual trigger via workflow_dispatch

**Workflow:**
1. Train new model
2. Register in MLflow Registry (stage: None/Staging)
3. Upload artifacts
4. Comment on commit with results
5. (Optional) Auto-promote to Production

### Manual Model Training

Trigger via GitHub Actions UI:

1. Go to Actions → "Model Training and Registration"
2. Click "Run workflow"
3. Select options:
   - Register model: Yes/No
   - Target stage: Staging/Production
4. Review results and promote if needed

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train model locally
python train.py --register-model

# Run tests
pytest tests/ -v

# Start API
uvicorn src.api.main:app --reload

# Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

---

## Configuration

### Environment Variables

All settings can be configured via environment variables with the `IRIS_` prefix:

#### MLflow Registry Settings

```bash
# Enable MLflow Registry
export IRIS_MLFLOW_USE_REGISTRY=true

# Model name in registry
export IRIS_MLFLOW_REGISTRY_MODEL_NAME=iris-classifier

# Stage to load (None, Staging, Production, Archived)
export IRIS_MLFLOW_REGISTRY_STAGE=Production

# MLflow tracking server
export IRIS_MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Enable MLflow tracking during training
export IRIS_MLFLOW_ENABLED=true
```

#### API Settings

```bash
export IRIS_API_HOST=0.0.0.0
export IRIS_API_PORT=80
export IRIS_API_DEBUG=false
```

#### Model Path (for local file loading)

```bash
export IRIS_MODEL_PATH=artifacts/model.onnx
```

### Docker Compose Configuration

Example `docker-compose.yml` for production:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "80:80"
    environment:
      - IRIS_MLFLOW_USE_REGISTRY=true
      - IRIS_MLFLOW_REGISTRY_MODEL_NAME=iris-classifier
      - IRIS_MLFLOW_REGISTRY_STAGE=Production
      - IRIS_MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
    restart: always

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0 --port 5000
    volumes:
      - mlflow-data:/mlflow
    restart: always

volumes:
  mlflow-data:
```

### Kubernetes Configuration

Example for Kubernetes deployment with MLflow Registry:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris-api
  template:
    metadata:
      labels:
        app: iris-api
    spec:
      containers:
      - name: api
        image: your-registry/iris-api:latest
        env:
        - name: IRIS_MLFLOW_USE_REGISTRY
          value: "true"
        - name: IRIS_MLFLOW_REGISTRY_MODEL_NAME
          value: "iris-classifier"
        - name: IRIS_MLFLOW_REGISTRY_STAGE
          value: "Production"
        - name: IRIS_MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
        ports:
        - containerPort: 80
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
```

---

## Best Practices

### 1. Drift Monitoring

- Set up alerts on prediction distribution changes
- Monitor feature distributions for data drift
- Track model performance metrics over time
- Implement automatic retraining triggers

### 2. Model Versioning

- Always register models with meaningful descriptions
- Use semantic versioning for model names if needed
- Test in Staging before promoting to Production
- Keep at least 2 previous versions in Archived state

### 3. Rollback Strategy

- Test rollback procedures regularly
- Document rollback steps in incident playbooks
- Implement automated rollback on critical metric degradation
- Keep production-ready models in Archived stage

### 4. CI/CD

- Run comprehensive tests before model registration
- Require manual approval for production promotion
- Implement canary deployments for gradual rollout
- Monitor post-deployment metrics closely

### 5. Monitoring and Observability

- Integrate logs with centralized logging system
- Set up dashboards for key metrics
- Configure alerts for anomalies
- Track API latency and throughput

---

## Troubleshooting

### Model Not Loading from Registry

Check:
1. MLflow tracking URI is accessible
2. Model name and stage are correct
3. MLflow server is running
4. Network connectivity between API and MLflow

### Logs Not Appearing

Check:
1. Log level configuration
2. JSON formatter is properly configured
3. Logger name matches in queries
4. Log aggregation system is receiving logs

### Rollback Not Working

Check:
1. Previous model version exists in registry
2. Model stage transition succeeded
3. API service restarted after stage change
4. `/health` endpoint shows correct model version

---

## Additional Resources

- [MLflow Model Registry Documentation](https://mlflow.org/docs/latest/model-registry.html)
- [Grafana Loki LogQL](https://grafana.com/docs/loki/latest/logql/)
- [FastAPI Logging](https://fastapi.tiangolo.com/tutorial/logging/)
- [Model Drift Detection](https://www.evidentlyai.com/blog/ml-monitoring-drift-detection)
