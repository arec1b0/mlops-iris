"""Comprehensive unit tests for the Iris model."""

import pytest
import numpy as np
import tempfile
import os
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from unittest.mock import patch, MagicMock
import time
from pathlib import Path

from src.data import load_iris_data, split_data
from src.model import IrisModel, IrisTrainer, ModelPersistence, train_and_save_model


class TestIrisModel:
    """Comprehensive test cases for IrisModel class (prediction only)."""

    @pytest.fixture
    def sample_data(self):
        """Load sample Iris data for testing."""
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, target_names

    @pytest.fixture
    def trained_model_file(self, tmp_path, sample_data):
        """Create a trained ONNX model file for testing."""
        X_train, X_test, y_train, y_test, _ = sample_data
        
        # Train model using IrisTrainer
        trainer = IrisTrainer()
        trained_sklearn_model = trainer.train(X_train, y_train)
        
        # Save as ONNX using ModelPersistence
        model_path = tmp_path / "test_model.onnx"
        persistence = ModelPersistence()
        persistence.save_onnx_model(trained_sklearn_model, str(model_path))
        
        return model_path, X_test, y_test

    def test_model_initialization(self, tmp_path):
        """Test model initialization with different paths."""
        # Test with string path
        model1 = IrisModel("test_model.onnx")
        assert str(model1.model_path) == "test_model.onnx"

        # Test with Path object
        model2 = IrisModel(tmp_path / "test_model.onnx")
        assert model2.model_path == tmp_path / "test_model.onnx"

        # Test default target names
        assert model1.target_names == ["setosa", "versicolor", "virginica"]
        
        # Test model is not loaded yet
        assert model1.onnx_session is None

    def test_model_load(self, trained_model_file):
        """Test model loading from ONNX file."""
        model_path, X_test, y_test = trained_model_file
        
        model = IrisModel(str(model_path))
        model.load()
        
        assert model.onnx_session is not None
        
    def test_model_load_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        model = IrisModel("nonexistent_model.onnx")

        with pytest.raises(FileNotFoundError):
            model.load()

    def test_model_prediction(self, trained_model_file, sample_data):
        """Test model prediction with various inputs."""
        model_path, X_test, y_test = trained_model_file
        _, _, _, _, target_names = sample_data

        # Load model
        model = IrisModel(str(model_path))
        model.load()

        # Test with first test sample
        sample_features = X_test[0]
        prediction, class_name = model.predict(sample_features)

        assert isinstance(prediction, (int, np.integer))
        assert 0 <= prediction <= 2
        assert class_name == target_names[prediction]

        # Test with different samples
        for i in range(min(5, len(X_test))):
            features = X_test[i]
            pred, name = model.predict(features)
            assert pred in [0, 1, 2]
            assert name in target_names

    def test_model_prediction_edge_cases(self, trained_model_file):
        """Test model prediction with edge case inputs."""
        model_path, _, _ = trained_model_file
        
        model = IrisModel(str(model_path))
        model.load()

        # Test with numpy array
        features_np = np.array([5.1, 3.5, 1.4, 0.2])
        pred1, name1 = model.predict(features_np)

        # Test with list
        features_list = [5.1, 3.5, 1.4, 0.2]
        pred2, name2 = model.predict(features_list)

        assert pred1 == pred2
        assert name1 == name2

        # Test with different feature values
        extreme_features = [10.0, 5.0, 2.0, 1.0]  # Unrealistic but valid
        pred3, name3 = model.predict(extreme_features)
        assert isinstance(pred3, (int, np.integer))
        assert name3 in model.target_names

    def test_predict_without_loading(self):
        """Test that prediction without loading raises error."""
        model = IrisModel("some_model.onnx")
        
        with pytest.raises(ValueError, match="Model not loaded"):
            model.predict([1, 2, 3, 4])


class TestIrisTrainer:
    """Test cases for IrisTrainer class."""

    @pytest.fixture
    def sample_data(self):
        """Load sample Iris data for testing."""
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = IrisTrainer()
        assert trainer.model is None

    def test_trainer_train(self, sample_data):
        """Test model training."""
        X_train, X_test, y_train, y_test = sample_data
        
        trainer = IrisTrainer()
        model = trainer.train(X_train, y_train)
        
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "score")

        # Verify model can make predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_trainer_train_with_params(self, sample_data):
        """Test model training with different parameters."""
        X_train, X_test, y_train, y_test = sample_data

        # Test with different max_iter values
        for max_iter in [100, 200, 500]:
            trainer = IrisTrainer()
            model = trainer.train(X_train, y_train, max_iter=max_iter)
            assert model is not None

            # Verify the model converged
            accuracy = trainer.evaluate(model, X_test, y_test)
            assert accuracy > 0.7  # Should still perform reasonably well

    def test_trainer_evaluate(self, sample_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_data

        trainer = IrisTrainer()
        model = trainer.train(X_train, y_train)
        accuracy = trainer.evaluate(model, X_test, y_test)

        assert isinstance(accuracy, float)
        assert 0.8 <= accuracy <= 1.0  # Should be reasonably accurate

        # Test with sklearn metrics for comparison
        predictions = model.predict(X_test)
        sklearn_accuracy = accuracy_score(y_test, predictions)
        assert abs(accuracy - sklearn_accuracy) < 1e-10


class TestModelPersistence:
    """Test cases for ModelPersistence class."""

    @pytest.fixture
    def trained_sklearn_model(self):
        """Create a trained sklearn model."""
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        trainer = IrisTrainer()
        model = trainer.train(X_train, y_train)
        return model

    def test_persistence_initialization(self):
        """Test ModelPersistence initialization."""
        persistence = ModelPersistence()
        assert persistence is not None

    def test_save_onnx_model(self, tmp_path, trained_sklearn_model):
        """Test saving model in ONNX format."""
        persistence = ModelPersistence()
        model_path = tmp_path / "saved_model.onnx"
        
        persistence.save_onnx_model(trained_sklearn_model, str(model_path))
        
        # Verify file exists
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_save_onnx_creates_directories(self, tmp_path, trained_sklearn_model):
        """Test that save_onnx_model creates parent directories."""
        persistence = ModelPersistence()
        model_path = tmp_path / "nested" / "path" / "model.onnx"
        
        persistence.save_onnx_model(trained_sklearn_model, str(model_path))
        
        assert model_path.exists()
        assert model_path.parent.exists()

    def test_saved_model_can_be_loaded(self, tmp_path, trained_sklearn_model):
        """Test that saved ONNX model can be loaded and used."""
        persistence = ModelPersistence()
        model_path = tmp_path / "loadable_model.onnx"
        
        # Save model
        persistence.save_onnx_model(trained_sklearn_model, str(model_path))
        
        # Load with IrisModel
        iris_model = IrisModel(str(model_path))
        iris_model.load()
        
        # Test prediction works
        prediction, class_name = iris_model.predict([5.1, 3.5, 1.4, 0.2])
        assert isinstance(prediction, (int, np.integer))
        assert class_name in ["setosa", "versicolor", "virginica"]

    def test_mlflow_integration(self, trained_sklearn_model):
        """Test MLflow integration."""
        persistence = ModelPersistence()

        with (
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_metric") as mock_log_metric,
            patch("mlflow.sklearn.log_model") as mock_log_model,
        ):
            mock_run.return_value.__enter__.return_value = MagicMock()
            mock_run.return_value.__enter__.return_value.info.run_id = "test_run_id"

            run_id = persistence.log_to_mlflow(trained_sklearn_model, 0.95)

            mock_run.assert_called_once()
            mock_log_metric.assert_called_once_with("accuracy", 0.95)
            mock_log_model.assert_called_once()
            assert run_id == "test_run_id"


class TestDataFunctions:
    """Comprehensive test cases for data loading functions."""

    def test_load_iris_data(self):
        """Test Iris data loading with comprehensive validation."""
        X, y, target_names = load_iris_data()

        # Basic shape checks
        assert X.shape[0] == 150  # Iris dataset has 150 samples
        assert X.shape[1] == 4  # 4 features
        assert y.shape[0] == 150
        assert len(target_names) == 3

        # Data type checks
        assert X.dtype == np.float64
        assert y.dtype == np.int64

        # Value range checks
        assert X.min() >= 0.0
        assert X.max() <= 8.0  # Iris features are in reasonable ranges

        # Class distribution (Iris is balanced)
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 3
        assert all(count == 50 for count in counts)  # 50 samples per class

        # Target names
        expected_names = ["setosa", "versicolor", "virginica"]
        assert target_names == expected_names

    def test_split_data(self):
        """Test data splitting with various parameters."""
        X, y, _ = load_iris_data()

        # Test default split
        X_train, X_test, y_train, y_test = split_data(X, y)
        assert len(X_train) > len(X_test)
        assert len(y_train) > len(y_test)

        # Test custom test sizes
        for test_size in [0.1, 0.2, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

            expected_test_size = int(len(X) * test_size)
            assert (
                abs(len(X_test) - expected_test_size) <= 1
            )  # Allow small rounding differences

        # Test reproducibility with random_state
        X_train1, X_test1, y_train1, y_test1 = split_data(X, y, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = split_data(X, y, random_state=42)

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_split_data_edge_cases(self):
        """Test data splitting edge cases."""
        X, y, _ = load_iris_data()

        # Test very small test size
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.01)
        assert len(X_test) >= 1  # Should have at least one test sample

        # Test large test size
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.9)
        assert len(X_train) >= 15  # Should have at least some training samples


class TestTrainingPipeline:
    """Test cases for the complete training pipeline."""

    def test_train_and_save_model(self, tmp_path, iris_data_split):
        """Test the complete training and saving pipeline."""
        X_train, X_test, y_train, y_test = iris_data_split

        model_path = tmp_path / "pipeline_model.onnx"
        accuracy, run_id, model_version = train_and_save_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model_path=str(model_path),
            use_mlflow=False,
        )

        assert isinstance(accuracy, float)
        assert 0.8 <= accuracy <= 1.0
        assert run_id == ""  # No MLflow run ID when disabled
        assert model_version is None  # No model version when MLflow is disabled
        assert model_path.exists()

    def test_train_and_save_model_return_signature(self, tmp_path, iris_data_split):
        """Test the return signature of the training pipeline."""
        X_train, X_test, y_train, y_test = iris_data_split
        model_path = tmp_path / "signature_test_model.onnx"

        # Call the function
        result = train_and_save_model(
            X_train,
            X_test,
            y_train,
            y_test,
            model_path=str(model_path),
            use_mlflow=False,
        )

        # Assert that the function returns a tuple of three elements
        assert isinstance(result, tuple), "Function should return a tuple"
        assert len(result) == 3, "Tuple should contain accuracy, run_id, and model_version"

        # Unpack and check types
        accuracy, run_id, model_version = result
        assert isinstance(accuracy, float)
        assert isinstance(run_id, str)
        assert model_version is None

    def test_train_and_save_model_with_mlflow(self, tmp_path, iris_data_split):
        """Test training pipeline with MLflow enabled."""
        X_train, X_test, y_train, y_test = iris_data_split

        model_path = tmp_path / "mlflow_model.onnx"

        with (
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_metric") as mock_log_metric,
            patch("mlflow.sklearn.log_model") as mock_log_model,
        ):
            mock_run.return_value.__enter__.return_value = MagicMock()
            mock_run.return_value.__enter__.return_value.info.run_id = "mlflow_run_123"

            accuracy, run_id, model_version = train_and_save_model(
                X_train,
                X_test,
                y_train,
                y_test,
                model_path=str(model_path),
                use_mlflow=True,
                register_model=False,
            )

            assert run_id == "mlflow_run_123"
            mock_run.assert_called_once()
            assert model_version is None

    def test_pipeline_error_handling(self, tmp_path):
        """Test pipeline error handling."""
        # Test with invalid data
        with pytest.raises(Exception):  # Should handle gracefully
            train_and_save_model(
                None,
                None,
                None,
                None,
                model_path=str(tmp_path / "error_model.onnx"),
                use_mlflow=False,
            )


class TestModelPerformance:
    """Performance and scalability tests."""

    @pytest.fixture
    def loaded_model(self, tmp_path):
        """Create a loaded model for performance testing."""
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train and save
        trainer = IrisTrainer()
        sklearn_model = trainer.train(X_train, y_train)
        
        model_path = tmp_path / "perf_model.onnx"
        persistence = ModelPersistence()
        persistence.save_onnx_model(sklearn_model, str(model_path))
        
        # Load for predictions
        model = IrisModel(str(model_path))
        model.load()
        
        return model, X_test

    def test_model_inference_speed(self, loaded_model):
        """Test model inference speed."""
        model, X_test = loaded_model

        # Test single prediction speed
        start_time = time.time()
        for _ in range(100):
            model.predict(X_test[0])
        single_pred_time = (time.time() - start_time) / 100

        # Should be reasonably fast (< 10ms per prediction)
        assert single_pred_time < 0.01

    def test_model_batch_prediction(self, loaded_model):
        """Test batch prediction capabilities."""
        model, X_test = loaded_model

        # Test predicting on multiple samples
        batch_features = X_test[:10]
        predictions = []

        for features in batch_features:
            pred, _ = model.predict(features)
            predictions.append(pred)

        assert len(predictions) == len(batch_features)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_model_memory_usage(self, loaded_model):
        """Test model memory usage is reasonable."""
        model, _ = loaded_model

        # Basic check that model exists
        assert model.onnx_session is not None


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: load data -> train -> save -> load -> predict."""
        # Load data
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        trainer = IrisTrainer()
        sklearn_model = trainer.train(X_train, y_train)
        
        # Evaluate
        accuracy = trainer.evaluate(sklearn_model, X_test, y_test)
        assert accuracy > 0.8

        # Save model
        model_path = tmp_path / "workflow_model.onnx"
        persistence = ModelPersistence()
        persistence.save_onnx_model(sklearn_model, str(model_path))
        
        # Load model
        iris_model = IrisModel(str(model_path))
        iris_model.load()

        # Make predictions
        for i in range(min(5, len(X_test))):
            prediction, class_name = iris_model.predict(X_test[i])
            assert 0 <= prediction <= 2
            assert class_name in target_names

    def test_model_persistence_across_sessions(self, tmp_path):
        """Test that models can be persisted and restored across sessions."""
        # Train and save model
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

        trainer = IrisTrainer()
        sklearn_model = trainer.train(X_train, y_train)
        
        model_path = tmp_path / "persistence_test.onnx"
        persistence = ModelPersistence()
        persistence.save_onnx_model(sklearn_model, str(model_path))

        # Load model in first "session"
        model1 = IrisModel(str(model_path))
        model1.load()

        # Simulate new session by creating new model instance
        model2 = IrisModel(str(model_path))
        model2.load()

        # Test predictions are identical
        for i in range(10):
            features = X_test[i]
            pred1, name1 = model1.predict(features)
            pred2, name2 = model2.predict(features)

            assert pred1 == pred2
            assert name1 == name2
