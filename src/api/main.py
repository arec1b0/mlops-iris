"""FastAPI application for Iris classification model serving."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

from ..model import IrisModel
from ..config import get_settings

# Configure structured JSON logging
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
            
        return json.dumps(log_obj)

# Set up logger
logger = logging.getLogger("iris_api")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Global model instance (will be set during startup)
model_instance = None


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for model loading."""
    global model_instance

    # Startup: Load the model
    config = get_settings()
    model_path = os.getenv("MODEL_PATH", config.model_path)
    
    # Create model instance with registry support
    model_instance = IrisModel(
        model_path=model_path,
        use_registry=config.mlflow_use_registry,
        registry_model_name=config.mlflow_registry_model_name,
        registry_stage=config.mlflow_registry_stage
    )
    
    try:
        model_instance.load()
        model_info = model_instance.get_model_info()
        logger.info(
            "Model loaded successfully",
            extra={
                "extra": {
                    "event": "model_loaded",
                    "model_source": model_info.get("model_source"),
                    "model_version": model_info.get("model_version"),
                }
            }
        )
    except FileNotFoundError:
        # Log the error but don't crash the app - endpoints will handle missing model
        logger.warning(
            f"Model file not found at {model_path}. Please train the model first.",
            extra={"extra": {"event": "model_load_failed", "model_path": str(model_path)}}
        )

    yield

    # Shutdown: Clean up resources (if needed)
    logger.info("Shutting down API...", extra={"extra": {"event": "shutdown"}})


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Iris Classification API",
    description="API for Iris flower classification using ML model",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model() -> IrisModel:
    """Dependency to get the loaded model instance."""
    if model_instance is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure model file exists and restart the service.",
        )
    return model_instance


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    features: List[
        float
    ]  # 4 features: sepal_length, sepal_width, petal_length, petal_width

    model_config = ConfigDict(
        json_schema_extra={"example": {"features": [5.1, 3.5, 1.4, 0.2]}},
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: int
    class_name: str
    confidence: float = 0.0  # Placeholder for future probability scores


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This information",
            "GET /health": "Health check",
            "GET /metadata": "Model metadata and version info",
            "POST /predict": "Make prediction",
        },
    }


@app.get("/health")
async def health_check(model: IrisModel = Depends(get_model)):
    """Health check endpoint."""
    model_info = model.get_model_info()
    return {
        "status": "healthy",
        "model_loaded": model.onnx_session is not None,
        "target_classes": model.target_names,
        "model_version": model_info.get("model_version"),
        "model_source": model_info.get("model_source"),
    }


@app.get("/metadata")
async def metadata(model: IrisModel = Depends(get_model)):
    """
    Get model metadata including version and deployment information.
    
    This endpoint provides detailed information about the currently deployed model,
    including version, source, git commit, and configuration for monitoring purposes.
    """
    model_info = model.get_model_info()
    git_commit = get_git_commit()
    config = get_settings()
    
    return {
        "model": {
            "version": model_info.get("model_version"),
            "source": model_info.get("model_source"),
            "registry_name": model_info.get("registry_name"),
            "registry_stage": model_info.get("registry_stage"),
        },
        "deployment": {
            "git_commit": git_commit,
            "api_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
        },
        "config": {
            "mlflow_tracking_uri": config.mlflow_tracking_uri,
            "mlflow_use_registry": config.mlflow_use_registry,
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, model: IrisModel = Depends(get_model)):
    """
    Make prediction on iris flower features.

    Expects 4 features: [sepal_length, sepal_width, petal_length, petal_width]
    """
    # Validate input - check length
    if len(request.features) != 4:
        raise HTTPException(
            status_code=400,
            detail="Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]",
        )

    # Validate input - check for non-numeric values
    try:
        features_array = [float(f) for f in request.features]
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="All features must be numeric values",
        )

    # Make prediction
    try:
        prediction, class_name = model.predict(features_array)
        
        # Log prediction with structured JSON logging
        model_info = model.get_model_info()
        logger.info(
            "Prediction made",
            extra={
                "extra": {
                    "event": "prediction",
                    "features": {
                        "sepal_length": features_array[0],
                        "sepal_width": features_array[1],
                        "petal_length": features_array[2],
                        "petal_width": features_array[3],
                    },
                    "prediction": {
                        "class_id": prediction,
                        "class_name": class_name,
                    },
                    "model": {
                        "version": model_info.get("model_version"),
                        "source": model_info.get("model_source"),
                    },
                }
            }
        )
        
    except ValueError as e:
        # Handle model-specific errors
        logger.error(
            f"Prediction error: {str(e)}",
            extra={"extra": {"event": "prediction_error", "error": str(e)}}
        )
        if "not loaded" in str(e).lower():
            raise HTTPException(
                status_code=503,
                detail="Model service temporarily unavailable. Please try again later.",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Prediction processing failed",
            )
    except Exception as e:
        # Handle unexpected errors during prediction
        logger.error(
            f"Unexpected prediction error: {str(e)}",
            extra={"extra": {"event": "prediction_error", "error": str(e)}}
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction",
        )

    return PredictionResponse(
        prediction=prediction,
        class_name=class_name,
        confidence=0.0,  # TODO: Add probability scores if needed
    )


# For local development: uvicorn src.api.main:app --reload
