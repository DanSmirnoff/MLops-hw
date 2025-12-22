import pytest
import numpy as np
import os
import json
import tempfile
from app.ml.model_manager import ModelManager


class TestModelManagerBasicLogic:
    """Tests for ModelManager"""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for model storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_model_manager_initialization(self, temp_models_dir):
        """Test ModelManager initializes correctly"""
        manager = ModelManager(models_dir=temp_models_dir)
        assert manager.models_dir == temp_models_dir
        assert isinstance(manager.models, dict)

    def test_get_available_model_classes(self, temp_models_dir):
        """Test getting available model classes"""
        manager = ModelManager(models_dir=temp_models_dir)
        available_models = manager.get_available_model_classes()

        assert "logistic_regression" in available_models
        assert "random_forest_classifier" in available_models
        assert len(available_models) >= 2

    def test_create_model(self, temp_models_dir):
        """Test model creation"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model(
            "logistic_regression", regularisation="l2", C=1.0
        )

        assert model_id is not None
        assert model_id in manager.models

    def test_create_model_invalid_class(self, temp_models_dir):
        """Test creating model with invalid class name"""
        manager = ModelManager(models_dir=temp_models_dir)

        with pytest.raises(ValueError, match="Unknown model class"):
            manager.create_model("invalid_model_class")

    def test_train_model(self, temp_models_dir):
        """Test model training"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("logistic_regression")

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        result = manager.train_model(model_id, X, y)

        assert result["model_id"] == model_id
        assert result["model_type"] == "LogisticRegression"

    def test_train_nonexistent_model(self, temp_models_dir):
        """Test training non-existent model raises error"""
        manager = ModelManager(models_dir=temp_models_dir)
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)

        with pytest.raises(ValueError, match="Model .* not found"):
            manager.train_model("nonexistent_model_id", X, y)

    def test_predict(self, temp_models_dir):
        """Test prediction with trained model"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("logistic_regression")

        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)
        manager.train_model(model_id, X_train, y_train)

        X_test = np.random.rand(10, 5)
        predictions = manager.predict(model_id, X_test)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_nonexistent_model(self, temp_models_dir):
        """Test prediction with non-existent model"""
        manager = ModelManager(models_dir=temp_models_dir)
        X = np.random.rand(10, 5)

        with pytest.raises(ValueError, match="Model .* not found"):
            manager.predict("nonexistent_model_id", X)

    def test_delete_model(self, temp_models_dir):
        """Test model deletion"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("logistic_regression")

        assert model_id in manager.models

        result = manager.delete_model(model_id)

        assert result is True
        assert model_id not in manager.models

    def test_delete_nonexistent_model(self, temp_models_dir):
        """Test deleting non-existent model"""
        manager = ModelManager(models_dir=temp_models_dir)

        result = manager.delete_model("nonexistent_model_id")

        assert result is False

    def test_get_model_info(self, temp_models_dir):
        """Test getting model info"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("random_forest_classifier", n_estimators=10)

        info = manager.get_model_info(model_id)

        assert info is not None
        assert info["model_id"] == model_id
        assert info["model_type"] == "RandomForestClassifierModel"
        assert info["is_trained"] is False

    def test_list_models(self, temp_models_dir):
        """Test listing models"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id_1 = manager.create_model("logistic_regression")
        model_id_2 = manager.create_model("random_forest_classifier")

        models = manager.list_models()

        assert len(models) == 2
        assert any(m["model_id"] == model_id_1 for m in models)
        assert any(m["model_id"] == model_id_2 for m in models)


class TestModelManagerRetrain:
    """Tests for model retraining functionality"""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for model storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_retrain_model(self, temp_models_dir):
        """Test retraining model"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("logistic_regression")

        X1 = np.random.rand(50, 5)
        y1 = np.random.randint(0, 2, 50)
        result1 = manager.train_model(model_id, X1, y1)

        X2 = np.random.rand(100, 5)
        y2 = np.random.randint(0, 2, 100)
        result2 = manager.retrain_model(model_id, X2, y2)

        assert result1["model_id"] == result2["model_id"]
        assert manager.get_model_info(model_id)["training_data_shape"] == X2.shape


class TestModelManagerMetadataPersistence:
    """Tests for model metadata persistence"""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary directory for model storage"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_metadata_saved_after_training(self, temp_models_dir):
        """Test that metadata is saved to file after training"""
        manager = ModelManager(models_dir=temp_models_dir)
        model_id = manager.create_model("logistic_regression")

        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        manager.train_model(model_id, X, y)

        # Check metadata file exists
        metadata_file = os.path.join(temp_models_dir, "models_metadata.json")
        assert os.path.exists(metadata_file)

        # Check metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            assert model_id in metadata
            assert metadata[model_id]["model_type"] == "LogisticRegressionModel"
