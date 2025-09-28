"""FastAPI application for Iris classification model serving."""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ConfigDict
from typing import List
from contextlib import asynccontextmanager
import os
from pathlib import Path

from ..model import IrisModel
from ..config import get_settings


# Global model instance (will be set during startup)
model_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for model loading."""
    global model_instance

    # Startup: Load the model
    config = get_settings()
    model_path = os.getenv("MODEL_PATH", config.model_path)
    model_instance = IrisModel(model_path)
    try:
        model_instance.load()
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        # Log the error but don't crash the app - endpoints will handle missing model
        print(
            f"Warning: Model file not found at {model_path}. Please train the model first."
        )

    yield

    # Shutdown: Clean up resources (if needed)
    print("Shutting down API...")


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
            "POST /predict": "Make prediction",
        },
    }


@app.get("/health")
async def health_check(model: IrisModel = Depends(get_model)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model.model is not None,
        "target_classes": model.target_names,
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
    except ValueError as e:
        # Handle model-specific errors
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
