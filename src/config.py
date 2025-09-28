"""Configuration management for the MLOps Iris application.

This module provides centralized configuration management using Pydantic settings
with environment variable support and validation.

Environment Variables:
    All settings can be overridden using IRIS_ prefixed environment variables:
    - IRIS_API_HOST, IRIS_API_PORT, IRIS_API_DEBUG
    - IRIS_MODEL_PATH, IRIS_MODEL_DEFAULT_PATH
    - IRIS_TRAINING_TEST_SIZE, IRIS_TRAINING_MAX_ITER, IRIS_TRAINING_RANDOM_STATE
    - IRIS_MLFLOW_TRACKING_URI, IRIS_MLFLOW_EXPERIMENT_NAME, IRIS_MLFLOW_ENABLED
    - IRIS_LOG_LEVEL, IRIS_LOG_FORMAT
    - IRIS_DATA_RANDOM_STATE

Example:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> print(settings.api_host)
    '0.0.0.0'
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support and validation.

    This class defines all configuration parameters for the MLOps Iris application,
    with support for environment variable overrides and automatic validation.

    Attributes:
        api_host (str): Host address for the API server. Defaults to "0.0.0.0".
        api_port (int): Port number for the API server. Must be between 1-65535.
        api_debug (bool): Enable debug mode for development. Defaults to False.
        model_path (str): Path to the trained model file. Must end with .onnx or .pkl.
        model_default_path (str): Default model path for fallbacks.
        training_test_size (float): Proportion of data for testing. Must be 0.1-0.5.
        training_max_iter (int): Maximum training iterations. Must be 1-10000.
        training_random_state (int): Random seed for reproducible training.
        mlflow_tracking_uri (str): MLflow tracking server URI. Must be valid SQLite or HTTP URL.
        mlflow_experiment_name (str): Name of the MLflow experiment.
        mlflow_enabled (bool): Enable MLflow experiment tracking. Defaults to True.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format (str): Format string for log messages.
        data_random_state (int): Random seed for data splitting operations.
    """

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="Host address for the API server",
        examples=["0.0.0.0", "localhost", "127.0.0.1"],
    )
    api_port: int = Field(
        default=80,
        ge=1,
        le=65535,
        description="Port number for the API server",
        examples=[80, 8080, 9000],
    )
    api_debug: bool = Field(
        default=False,
        description="Enable debug mode for development",
        examples=[False, True],
    )

    # Model Configuration
    model_path: str = Field(
        default="artifacts/model.onnx",
        description="Path to the trained model file",
        examples=["artifacts/model.onnx", "models/iris_v1.onnx"],
    )
    model_default_path: str = Field(
        default="artifacts/model.onnx",
        description="Default model path for fallback scenarios",
    )

    # Training Configuration
    training_test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Proportion of data reserved for testing",
        examples=[0.2, 0.25, 0.3],
    )
    training_max_iter: int = Field(
        default=200,
        ge=1,
        le=10000,
        description="Maximum number of training iterations",
        examples=[100, 200, 500],
    )
    training_random_state: int = Field(
        default=42,
        description="Random seed for reproducible training results",
        examples=[42, 123, 2024],
    )

    # MLflow Configuration
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlruns.db",
        description="MLflow tracking server URI",
        examples=[
            "sqlite:///mlruns.db",
            "http://localhost:5000",
            "https://mlflow.example.com",
        ],
    )
    mlflow_experiment_name: str = Field(
        default="iris_classification",
        description="Name of the MLflow experiment",
        examples=["iris_classification", "iris_prod_v1", "iris_experiment"],
    )
    mlflow_enabled: bool = Field(
        default=True,
        description="Enable MLflow experiment tracking and logging",
        examples=[True, False],
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level for application logs",
        examples=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages",
        examples=["%(asctime)s - %(name)s - %(levelname)s - %(message)s"],
    )

    # Data Configuration
    data_random_state: int = Field(
        default=42,
        description="Random seed for data preprocessing and splitting",
        examples=[42, 123, 2024],
    )

    model_config = {
        "env_prefix": "IRIS_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

    @field_validator("mlflow_tracking_uri")
    @classmethod
    def validate_mlflow_uri(cls, v):
        """Validate MLflow tracking URI format."""
        if v.startswith("sqlite:///") or v.startswith("http"):
            return v
        raise ValueError("MLflow tracking URI must be a valid SQLite path or HTTP URL")

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v):
        """Ensure model path has proper extension."""
        if not v.endswith((".onnx", ".pkl")):
            raise ValueError("Model path must end with .onnx or .pkl")
        return v


# Global settings instance - loaded once at module import time
settings = Settings()


def get_settings() -> Settings:
    """Get the global application settings instance.

    This function returns the singleton Settings instance that has been
    configured with environment variables and validated.

    Returns:
        Settings: The configured application settings instance.

    Example:
        >>> from src.config import get_settings
        >>> config = get_settings()
        >>> print(config.api_port)
        80
    """
    return settings


def create_config_dict() -> dict[str, dict[str, any]]:
    """Create a nested dictionary representation of all configuration settings.

    This function converts the Settings object into a nested dictionary
    structure suitable for serialization or passing to other components.

    Returns:
        dict: Nested dictionary with configuration organized by category:
            - api: API server configuration
            - model: Model file paths and settings
            - training: Training hyperparameters
            - mlflow: MLflow tracking configuration
            - logging: Logging configuration
            - data: Data processing settings

    Example:
        >>> from src.config import create_config_dict
        >>> config = create_config_dict()
        >>> print(config['api']['host'])
        '0.0.0.0'
    """
    return {
        "api": {
            "host": settings.api_host,
            "port": settings.api_port,
            "debug": settings.api_debug,
        },
        "model": {
            "path": settings.model_path,
            "default_path": settings.model_default_path,
        },
        "training": {
            "test_size": settings.training_test_size,
            "max_iter": settings.training_max_iter,
            "random_state": settings.training_random_state,
        },
        "mlflow": {
            "tracking_uri": settings.mlflow_tracking_uri,
            "experiment_name": settings.mlflow_experiment_name,
            "enabled": settings.mlflow_enabled,
        },
        "logging": {
            "level": settings.log_level,
            "format": settings.log_format,
        },
        "data": {
            "random_state": settings.data_random_state,
        },
    }
