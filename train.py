#!/usr/bin/env python3
"""Entry point for training the Iris classification model."""

import argparse
from pathlib import Path

from src.data import load_iris_data, split_data
from src.model import train_and_save_model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Iris classification model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/model.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow logging"
    )

    args = parser.parse_args()

    # Ensure artifacts directory exists
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    # Load and split data
    print("Loading Iris dataset...")
    X, y, target_names = load_iris_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {target_names}")

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Train and save model
    print("Training model...")
    accuracy, run_id = train_and_save_model(
        X_train,
        X_test,
        y_train,
        y_test,
        model_path=args.model_path,
        use_mlflow=not args.no_mlflow,
    )

    print("\nTraining completed!")
    print(".2f")
    if run_id:
        print(f"MLflow run ID: {run_id}")
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    main()
