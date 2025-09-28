"""Comprehensive unit tests for the Iris model."""

import pytest
import numpy as np
import tempfile
import os
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from unittest.mock import patch, MagicMock
import time

from src.data import load_iris_data, split_data
from src.model import IrisModel, train_and_save_model


class TestIrisModel:
    """Comprehensive test cases for IrisModel class."""

    @pytest.fixture
    def sample_data(self):
        """Load sample Iris data for testing."""
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test, target_names

    @pytest.fixture
    def trained_model(self, tmp_path, sample_data):
        """Create and train a model for testing."""
        X_train, X_test, y_train, y_test, _ = sample_data
        model = IrisModel(tmp_path / "test_model.pkl")
        model.train(X_train, y_train)
        return model, X_test, y_test

    def test_model_initialization(self, tmp_path):
        """Test model initialization with different paths."""
        # Test with string path
        model1 = IrisModel("test_model.pkl")
        assert str(model1.model_path) == "test_model.pkl"

        # Test with Path object
        model2 = IrisModel(tmp_path / "test_model.pkl")
        assert model2.model_path == tmp_path / "test_model.pkl"

        # Test default target names
        assert model1.target_names == ["setosa", "versicolor", "virginica"]

    def test_model_training(self, trained_model):
        """Test that model can be trained successfully."""
        model, X_test, y_test = trained_model

        assert model.model is not None
        assert hasattr(model.model, "predict")
        assert hasattr(model.model, "score")

        # Verify model can make predictions
        predictions = model.model.predict(X_test)
        assert len(predictions) == len(y_test)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_model_evaluation(self, trained_model):
        """Test model evaluation with comprehensive metrics."""
        model, X_test, y_test = trained_model

        accuracy = model.evaluate(X_test, y_test)

        assert isinstance(accuracy, float)
        assert 0.8 <= accuracy <= 1.0  # Should be reasonably accurate

        # Test with sklearn metrics for comparison
        predictions = model.model.predict(X_test)
        sklearn_accuracy = accuracy_score(y_test, predictions)
        assert abs(accuracy - sklearn_accuracy) < 1e-10

    def test_model_prediction(self, trained_model, sample_data):
        """Test model prediction with various inputs."""
        model, X_test, y_test = trained_model
        _, _, _, _, target_names = sample_data

        # Test with first test sample
        sample_features = X_test[0]
        prediction, class_name = model.predict(sample_features)

        assert isinstance(prediction, int)
        assert 0 <= prediction <= 2
        assert class_name == target_names[prediction]

        # Test with different samples
        for i in range(min(5, len(X_test))):
            features = X_test[i]
            pred, name = model.predict(features)
            assert pred in [0, 1, 2]
            assert name in target_names

    def test_model_prediction_edge_cases(self, trained_model):
        """Test model prediction with edge case inputs."""
        model, _, _ = trained_model

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
        assert isinstance(pred3, int)
        assert name3 in model.target_names

    def test_model_save_load(self, tmp_path, trained_model):
        """Test comprehensive model save and load functionality."""
        model, X_test, y_test = trained_model

        # Save model
        save_path = tmp_path / "saved_model.pkl"
        model.model_path = save_path
        model.save()

        # Verify file exists
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Load model in new instance
        new_model = IrisModel(save_path)
        new_model.load()

        assert new_model.model is not None

        # Test that loaded model produces same predictions
        test_features = X_test[0]
        orig_pred, orig_name = model.predict(test_features)
        loaded_pred, loaded_name = new_model.predict(test_features)

        assert orig_pred == loaded_pred
        assert orig_name == loaded_name

    def test_model_mlflow_integration(self, trained_model):
        """Test MLflow integration."""
        model, X_test, y_test = trained_model

        with (
            patch("mlflow.start_run") as mock_run,
            patch("mlflow.log_metric") as mock_log,
        ):
            mock_run.return_value.__enter__.return_value = MagicMock()
            mock_run.return_value.__enter__.return_value.info.run_id = "test_run_id"

            run_id = model.log_to_mlflow(0.95)

            mock_run.assert_called_once()
            mock_log.assert_called_once_with("accuracy", 0.95)
            assert run_id == "test_run_id"

    def test_model_save_with_mlflow(self, tmp_path, trained_model):
        """Test model saving with MLflow option."""
        model, _, _ = trained_model

        with (
            patch("mlflow.start_run"),
            patch("mlflow.sklearn.log_model") as mock_log_model,
        ):
            # Test save without MLflow (default)
            model.save(use_mlflow=False)
            mock_log_model.assert_not_called()

            # Test save with MLflow
            model.save(use_mlflow=True)
            mock_log_model.assert_called_once()

    def test_untrained_model_errors(self):
        """Test that untrained model raises appropriate errors."""
        model = IrisModel()

        # Test evaluate without training
        with pytest.raises(ValueError, match="Model not trained"):
            model.evaluate([[1, 2, 3, 4]], [0])

        # Test save without training
        with pytest.raises(ValueError, match="Model not trained"):
            model.save()

        # Test predict without loading
        with pytest.raises(ValueError, match="Model not loaded"):
            model.predict([1, 2, 3, 4])

    def test_model_load_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        model = IrisModel("nonexistent_model.pkl")

        with pytest.raises(FileNotFoundError):
            model.load()

    def test_model_training_parameters(self, sample_data):
        """Test model training with different parameters."""
        X_train, X_test, y_train, y_test = sample_data[:4]

        # Test with different max_iter values
        for max_iter in [100, 200, 500]:
            model = IrisModel()
            model.train(X_train, y_train, max_iter=max_iter)
            assert model.model is not None

            # Verify the model converged
            accuracy = model.evaluate(X_test, y_test)
            assert accuracy > 0.7  # Should still perform reasonably well

    def test_model_performance_metrics(self, trained_model):
        """Test detailed performance metrics."""
        model, X_test, y_test = trained_model

        predictions = model.model.predict(X_test)

        # Test accuracy
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy > 0.8  # Should be reasonably accurate

        # Test classification report
        report = classification_report(y_test, predictions, output_dict=True)
        assert "weighted avg" in report
        assert "precision" in report["weighted avg"]
        assert "recall" in report["weighted avg"]
        assert "f1-score" in report["weighted avg"]


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

        model_path = tmp_path / "pipeline_model.pkl"
        accuracy, run_id = train_and_save_model(
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
        assert model_path.exists()

    def test_train_and_save_model_with_mlflow(self, tmp_path, iris_data_split):
        """Test training pipeline with MLflow enabled."""
        X_train, X_test, y_train, y_test = iris_data_split

        model_path = tmp_path / "mlflow_model.pkl"

        with patch("mlflow.start_run") as mock_run:
            mock_run.return_value.__enter__.return_value = MagicMock()
            mock_run.return_value.__enter__.return_value.info.run_id = "mlflow_run_123"

            accuracy, run_id = train_and_save_model(
                X_train,
                X_test,
                y_train,
                y_test,
                model_path=str(model_path),
                use_mlflow=True,
            )

            assert run_id == "mlflow_run_123"
            mock_run.assert_called_once()

    def test_pipeline_error_handling(self, tmp_path):
        """Test pipeline error handling."""
        # Test with invalid data
        with pytest.raises(Exception):  # Should handle gracefully
            train_and_save_model(
                None,
                None,
                None,
                None,
                model_path=str(tmp_path / "error_model.pkl"),
                use_mlflow=False,
            )


class TestModelPerformance:
    """Performance and scalability tests."""

    def test_model_inference_speed(self, trained_model):
        """Test model inference speed."""
        model, X_test, _ = trained_model

        # Test single prediction speed
        start_time = time.time()
        for _ in range(100):
            model.predict(X_test[0])
        single_pred_time = (time.time() - start_time) / 100

        # Should be reasonably fast (< 1ms per prediction)
        assert single_pred_time < 0.001

    def test_model_batch_prediction(self, trained_model):
        """Test batch prediction capabilities."""
        model, X_test, _ = trained_model

        # Test predicting on multiple samples
        batch_features = X_test[:10]
        predictions = []

        for features in batch_features:
            pred, _ = model.predict(features)
            predictions.append(pred)

        assert len(predictions) == len(batch_features)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_model_memory_usage(self, trained_model):
        """Test model memory usage is reasonable."""
        model, _, _ = trained_model

        # Basic check that model exists and has reasonable size
        import pickle
        import io

        buffer = io.BytesIO()
        pickle.dump(model.model, buffer)
        model_size = len(buffer.getvalue())

        # Model should be reasonably small (< 10KB)
        assert model_size < 10000
