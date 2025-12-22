import numpy as np
from app.ml.models import LogisticRegressionModel, RandomForestClassifierModel


class TestLogisticRegressionModel:
    """Tests for LogisticRegression model"""

    def test_model_initialization(self):
        """Test that LogisticRegression model initializes correctly"""
        model = LogisticRegressionModel(regularisation="l2", C=0.5)
        assert model.model_type == "LogisticRegression"
        assert model.is_trained is False
        assert model.model_id is not None
        assert model.training_data_shape is None

    def test_model_training(self, sample_data):
        """Test model training with sample data"""
        X, y = sample_data
        model = LogisticRegressionModel()

        result = model.train(X, y)

        assert result["model_id"] == model.model_id
        assert result["model_type"] == "LogisticRegression"
        assert model.is_trained is True
        assert model.training_data_shape == X.shape

    def test_model_prediction(self, sample_data):
        """Test model prediction after training"""
        X, y = sample_data
        model = LogisticRegressionModel()
        model.train(X, y)

        X_test = np.random.rand(10, 5)
        predictions = model.predict(X_test)

        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)

    def test_model_with_different_hyperparameters(self, sample_data):
        """Test model with different C and regularisation parameters"""
        X, y = sample_data

        for C in [0.1, 1.0, 10.0]:
            for reg in ["l2"]:
                model = LogisticRegressionModel(regularisation=reg, C=C)
                result = model.train(X, y)
                assert result["model_type"] == "LogisticRegression"
                assert model.is_trained is True


class TestRandomForestModel:
    """Tests for RandomForest model"""

    def test_model_initialization(self):
        """Test that RandomForest model initializes correctly"""
        model = RandomForestClassifierModel(n_estimators=50, max_depth=5)
        assert model.model_type == "RandomForestClassifier"
        assert model.is_trained is False
        assert model.model_id is not None

    def test_model_training(self, sample_data):
        """Test RandomForest training"""
        X, y = sample_data
        model = RandomForestClassifierModel(n_estimators=10)

        result = model.train(X, y)

        assert result["model_id"] == model.model_id
        assert model.is_trained is True
        assert model.training_data_shape == X.shape

    def test_model_prediction(self, sample_data):
        """Test RandomForest prediction"""
        X, y = sample_data
        model = RandomForestClassifierModel(n_estimators=10)
        model.train(X, y)

        X_test = np.random.rand(5, 5)
        predictions = model.predict(X_test)

        assert len(predictions) == 5
        assert all(pred in [0, 1] for pred in predictions)

    def test_different_n_estimators(self, sample_data):
        """Test RandomForest with different n_estimators"""
        X, y = sample_data

        # Почему бы и нет?
        for n_est in [5, 20, 50]:
            model = RandomForestClassifierModel(n_estimators=n_est)
            result = model.train(X, y)
            assert result["model_type"] == "RandomForestClassifier"


class TestBaseModelLogic:
    """Tests for BaseModel core functionality"""

    # Больше ничего в голову не пришло))
    def test_model_id_uniqueness(self):
        """Test that each model gets unique ID"""
        model1 = LogisticRegressionModel()
        model2 = LogisticRegressionModel()

        assert model1.model_id != model2.model_id
