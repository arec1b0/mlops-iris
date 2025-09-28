"""Unit tests for the Iris model."""

import pytest
import numpy as np
from sklearn.datasets import load_iris

from src.data import load_iris_data, split_data
from src.model import IrisModel


class TestIrisModel:
    """Test cases for IrisModel class."""

    @pytest.fixture
    def sample_data(self):
        """Load sample Iris data for testing."""
        X, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def trained_model(self, tmp_path, sample_data):
        """Create and train a model for testing."""
        X_train, X_test, y_train, y_test = sample_data
        model = IrisModel(tmp_path / "test_model.pkl")
        model.train(X_train, y_train)
        return model, X_test, y_test

    def test_model_training(self, trained_model):
        """Test that model can be trained."""
        model, X_test, y_test = trained_model
        assert model.model is not None

    def test_model_evaluation(self, trained_model):
        """Test model evaluation returns valid accuracy."""
        model, X_test, y_test = trained_model
        accuracy = model.evaluate(X_test, y_test)
        assert 0.0 <= accuracy <= 1.0

    def test_model_prediction(self, trained_model):
        """Test model prediction."""
        model, X_test, y_test = trained_model

        # Test with first test sample
        sample_features = X_test[0]
        prediction, class_name = model.predict(sample_features)

        assert isinstance(prediction, int)
        assert 0 <= prediction <= 2  # Iris has 3 classes
        assert class_name in ["setosa", "versicolor", "virginica"]

    def test_model_save_load(self, tmp_path, trained_model):
        """Test model save and load functionality."""
        model, _, _ = trained_model

        # Save model
        save_path = tmp_path / "saved_model.pkl"
        model.model_path = save_path
        model.save()

        # Load model in new instance
        new_model = IrisModel(save_path)
        new_model.load()

        assert new_model.model is not None

    def test_untrained_model_error(self):
        """Test that untrained model raises appropriate errors."""
        model = IrisModel()

        with pytest.raises(ValueError, match="Model not trained"):
            model.evaluate([[1, 2, 3, 4]], [0])

        with pytest.raises(ValueError, match="Model not loaded"):
            model.predict([1, 2, 3, 4])


class TestDataFunctions:
    """Test cases for data loading functions."""

    def test_load_iris_data(self):
        """Test Iris data loading."""
        X, y, target_names = load_iris_data()

        assert X.shape[0] > 0
        assert y.shape[0] == X.shape[0]
        assert len(target_names) == 3
        assert set(target_names) == {"setosa", "versicolor", "virginica"}

    def test_split_data(self):
        """Test data splitting."""
        X, y, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        assert X_train.shape[0] > X_test.shape[0]
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]
