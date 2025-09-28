"""FastAPI application for Iris classification model serving."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List
import os
from pathlib import Path

from ..model import IrisModel


# Initialize FastAPI app
app = FastAPI(
    title="Iris Classification API",
    description="API for Iris flower classification using ML model",
    version="1.0.0",
)

# Global model instance
model = None


def get_model() -> IrisModel:
    """Get or load the model instance."""
    global model
    if model is None:
        model_path = os.getenv("MODEL_PATH", "artifacts/model.pkl")
        model = IrisModel(model_path)
        try:
            model.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Model file not found: {model_path}. Please train the model first.",
            )
    return model


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
async def health_check():
    """Health check endpoint."""
    try:
        model_instance = get_model()
        return {
            "status": "healthy",
            "model_loaded": model_instance.model is not None,
            "target_classes": model_instance.target_names,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction on iris flower features.

    Expects 4 features: [sepal_length, sepal_width, petal_length, petal_width]
    """
    try:
        # Validate input
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400,
                detail="Exactly 4 features required: [sepal_length, sepal_width, petal_length, petal_width]",
            )

        # Get model and make prediction
        model_instance = get_model()
        prediction, class_name = model_instance.predict(request.features)

        return PredictionResponse(
            prediction=prediction,
            class_name=class_name,
            confidence=0.0,  # TODO: Add probability scores if needed
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# For local development: uvicorn src.api.main:app --reload
