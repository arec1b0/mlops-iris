#!/usr/bin/env python3
"""Entry point for running the Iris classification API."""

import uvicorn
import argparse
import os


def main():
    """Run the FastAPI application."""
    parser = argparse.ArgumentParser(description="Run Iris Classification API")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.onnx",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Set environment variable for model path
    os.environ["MODEL_PATH"] = args.model_path

    print(f"Starting Iris Classification API on {args.host}:{args.port}")
    print(f"Model path: {args.model_path}")
    print("API documentation available at: http://localhost:8000/docs")

    # Run the FastAPI application
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
