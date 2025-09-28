#!/usr/bin/env python3
"""Example script showing how to train and use the Iris model programmatically.

This script demonstrates the complete workflow from data loading to model training,
evaluation, and making predictions.
"""

from pathlib import Path
from typing import Tuple
import numpy as np

# Import our custom modules
from src.data import load_iris_data, split_data
from src.model import IrisModel, IrisTrainer, ModelPersistence
from src.config import get_settings


def main():
    """Demonstrate the complete model training and prediction workflow."""
    print("🌸 MLOps Iris Model Training Example")
    print("=" * 50)

    # 1. Load configuration
    config = get_settings()
    print(f"Using configuration: test_size={config.training_test_size}, max_iter={config.training_max_iter}")

    # 2. Load and prepare data
    print("\n📥 Loading Iris dataset...")
    X, y, target_names = load_iris_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(X[0])}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")

    # 3. Split the data
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 4. Train the model
    print("\n🧠 Training model...")
    trainer = IrisTrainer()
    trained_model = trainer.train(X_train, y_train)

    print("Model trained successfully!")

    # 5. Evaluate the model
    print("\n✅ Evaluating model...")
    accuracy = trainer.evaluate(trained_model, X_test, y_test)

    print(".2f")

    # 6. Save the model securely
    print("\n💾 Saving model in ONNX format...")
    persistence = ModelPersistence()
    model_path = "examples/trained_model.onnx"
    persistence.save_onnx_model(trained_model, model_path)

    print(f"Model saved to: {model_path}")

    # 7. Load and test predictions
    print("\n🔮 Testing predictions...")
    model = IrisModel(model_path)
    model.load()

    # Test with some sample data
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Should be setosa
        [6.3, 3.3, 6.0, 2.5],  # Should be virginica
        [5.7, 2.8, 4.5, 1.3],  # Should be versicolor
    ]

    print("Predictions:")
    for i, sample in enumerate(test_samples, 1):
        prediction, class_name = model.predict(sample)
        print(f"Sample {i}: {sample} → {class_name} (class {prediction})")

    # 8. Log to MLflow (optional)
    if config.mlflow_enabled:
        print("\n📊 Logging to MLflow...")
        run_id = persistence.log_to_mlflow(trained_model, accuracy)
        print(f"MLflow run ID: {run_id}")

    print("\n🎉 Training and evaluation complete!")
    print(f"Model accuracy: {accuracy".2%"}")
    print(f"Model saved as: {model_path}")


def demonstrate_configuration():
    """Show how configuration affects the training process."""
    print("\n⚙️  Configuration Demonstration")
    print("-" * 40)

    config = get_settings()

    print("Current Configuration:")
    print(f"  Test size: {config.training_test_size}")
    print(f"  Max iterations: {config.training_max_iter}")
    print(f"  Random state: {config.training_random_state}")
    print(f"  Model path: {config.model_path}")
    print(f"  MLflow enabled: {config.mlflow_enabled}")

    # Show how to override configuration
    print("\nConfiguration can be overridden with environment variables:")
    print("  export IRIS_TRAINING_TEST_SIZE=0.3")
    print("  export IRIS_TRAINING_MAX_ITER=500")
    print("  export IRIS_MLFLOW_ENABLED=false")


if __name__ == "__main__":
    main()
    demonstrate_configuration()
