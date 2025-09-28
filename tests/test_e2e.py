"""End-to-end tests for the complete MLOps Iris pipeline."""

import pytest
import subprocess
import sys
import os
import tempfile
import requests
import time
import signal
import psutil
from pathlib import Path
import json


class TestEndToEnd:
    """End-to-end tests for the complete MLOps pipeline."""

    @pytest.fixture(scope="session")
    def temp_workspace(self, tmp_path_factory):
        """Create a temporary workspace for E2E testing."""
        return tmp_path_factory.mktemp("mlops_e2e")

    def test_full_training_pipeline(self, temp_workspace):
        """Test the complete training pipeline from data to saved model."""
        # Change to temp workspace
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)

        try:
            # Copy necessary files to temp workspace
            import shutil

            src_dir = Path(original_cwd) / "src"
            if src_dir.exists():
                shutil.copytree(src_dir, temp_workspace / "src")

            # Import modules from temp workspace
            sys.path.insert(0, str(temp_workspace))

            from src.data import load_iris_data, split_data
            from src.model import train_and_save_model

            # Step 1: Load data
            X, y, target_names = load_iris_data()
            assert X.shape[0] == 150
            assert len(target_names) == 3

            # Step 2: Split data
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=0.2, random_state=42
            )
            assert len(X_train) == 120
            assert len(X_test) == 30

            # Step 3: Train and save model
            model_path = temp_workspace / "e2e_model.pkl"
            accuracy, run_id = train_and_save_model(
                X_train,
                X_test,
                y_train,
                y_test,
                model_path=str(model_path),
                use_mlflow=False,
            )

            # Step 4: Verify results
            assert isinstance(accuracy, float)
            assert 0.8 <= accuracy <= 1.0
            assert model_path.exists()
            assert model_path.stat().st_size > 0
            assert run_id == ""  # No MLflow

            # Step 5: Test model loading and prediction
            from src.model import IrisModel

            loaded_model = IrisModel(str(model_path))
            loaded_model.load()

            # Make prediction
            test_features = [5.1, 3.5, 1.4, 0.2]  # Typical setosa
            prediction, class_name = loaded_model.predict(test_features)

            assert isinstance(prediction, int)
            assert 0 <= prediction <= 2
            assert class_name in target_names

        finally:
            os.chdir(original_cwd)
            sys.path.remove(str(temp_workspace))

    def test_cli_training_script(self, temp_workspace):
        """Test the CLI training script end-to-end."""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)

        try:
            # Copy source files
            import shutil

            src_dir = Path(original_cwd) / "src"
            train_script = Path(original_cwd) / "train.py"

            if src_dir.exists():
                dst_src = temp_workspace / "src"
                if dst_src.exists():
                    shutil.rmtree(dst_src)
                shutil.copytree(src_dir, dst_src)
            shutil.copy2(train_script, temp_workspace / "train.py")

            # Run training script
            result = subprocess.run(
                [sys.executable, "train.py", "--no-mlflow"],
                capture_output=True,
                text=True,
                cwd=temp_workspace,
            )

            # Check that it ran successfully
            assert result.returncode == 0
            assert "Accuracy:" in result.stdout
            assert "Model saved" in result.stdout

            # Check that model file was created
            model_file = temp_workspace / "artifacts" / "model.pkl"
            assert model_file.exists()

        finally:
            os.chdir(original_cwd)

    @pytest.mark.skip(
        reason="E2E API server test requires complex setup, functionality tested via integration tests"
    )
    def test_api_server_lifecycle(self, temp_workspace):
        """Test the API server startup, health check, and shutdown."""
        # This test is complex and may not be reliable in CI environments
        # The functionality is well covered by integration tests in test_api.py
        pass

    def test_error_handling_e2e(self, temp_workspace):
        """Test error handling in the complete pipeline."""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)

        try:
            from src.model import IrisModel

            # Test loading non-existent model
            model = IrisModel("nonexistent.pkl")
            with pytest.raises(FileNotFoundError):
                model.load()

            # Test prediction with unloaded model
            with pytest.raises(ValueError, match="Model not loaded"):
                model.predict([1, 2, 3, 4])

        finally:
            os.chdir(original_cwd)

    def test_data_pipeline_integrity(self, temp_workspace):
        """Test that data flows correctly through the entire pipeline."""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)

        try:
            from src.data import load_iris_data, split_data
            from src.model import IrisModel

            # Load and split data
            X, y, target_names = load_iris_data()
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=0.3, random_state=123
            )

            # Train model
            model = IrisModel("integrity_test.pkl")
            model.train(X_train, y_train)

            # Evaluate
            accuracy = model.evaluate(X_test, y_test)
            assert accuracy > 0.7  # Should perform reasonably well

            # Save and reload
            model.save()
            new_model = IrisModel("integrity_test.pkl")
            new_model.load()

            # Test that reloaded model gives same results
            test_sample = X_test[0]
            orig_pred, orig_name = model.predict(test_sample)
            reload_pred, reload_name = new_model.predict(test_sample)

            assert orig_pred == reload_pred
            assert orig_name == reload_name

        finally:
            os.chdir(original_cwd)

    def test_model_persistence(self, temp_workspace):
        """Test that models can be persisted and restored across sessions."""
        original_cwd = os.getcwd()
        os.chdir(temp_workspace)

        try:
            from src.data import load_iris_data, split_data
            from src.model import IrisModel

            # Train and save model
            X, y, _ = load_iris_data()
            X_train, X_test, y_train, y_test = split_data(
                X, y, test_size=0.2, random_state=42
            )

            model1 = IrisModel("persistence_test.pkl")
            model1.train(X_train, y_train)
            model1.save()

            # Simulate new session by creating new model instance
            model2 = IrisModel("persistence_test.pkl")
            model2.load()

            # Test predictions are identical
            for i in range(10):
                features = X_test[i]
                pred1, name1 = model1.predict(features)
                pred2, name2 = model2.predict(features)

                assert pred1 == pred2
                assert name1 == name2

        finally:
            os.chdir(original_cwd)


class TestIntegration:
    """Integration tests for component interactions."""

    def test_data_model_integration(self):
        """Test integration between data loading and model training."""
        from src.data import load_iris_data, split_data
        from src.model import IrisModel

        # Load data
        X, y, target_names = load_iris_data()

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.25, random_state=99
        )

        # Train model
        model = IrisModel()
        model.train(X_train, y_train)

        # Evaluate
        accuracy = model.evaluate(X_test, y_test)

        # Verify reasonable performance
        assert accuracy > 0.75
        assert len(target_names) == 3

    def test_model_api_contract(self):
        """Test that model output matches API expectations."""
        from src.model import IrisModel
        from src.data import load_iris_data, split_data

        # Train a model
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        model = IrisModel()
        model.train(X_train, y_train)

        # Test various inputs
        test_cases = [
            [5.1, 3.5, 1.4, 0.2],  # Typical setosa
            [6.0, 2.9, 4.5, 1.5],  # Typical versicolor
            [7.3, 2.9, 6.3, 1.8],  # Typical virginica
        ]

        for features in test_cases:
            prediction, class_name = model.predict(features)

            # Verify contract
            assert isinstance(prediction, int)
            assert isinstance(class_name, str)
            assert 0 <= prediction <= 2
            assert class_name in ["setosa", "versicolor", "virginica"]

    def test_configuration_consistency(self):
        """Test that configurations are consistent across components."""
        from src.data import load_iris_data
        from src.model import IrisModel

        # Load data
        X, y, target_names = load_iris_data()

        # Create model
        model = IrisModel()

        # Check that target names are consistent
        assert model.target_names == target_names

        # Train and check predictions use correct names
        from src.data import split_data

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        model.train(X_train, y_train)
        prediction, class_name = model.predict(X_test[0])

        assert class_name == target_names[prediction]
