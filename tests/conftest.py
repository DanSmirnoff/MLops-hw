import pytest
import numpy as np
from app.ml.models import LogisticRegressionModel, RandomForestClassifierModel


@pytest.fixture
def sample_data():
    """
    Fixture providing sample training data
    """
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def logistic_regression_model():
    """
    Fixture creating a LogisticRegression model
    """
    return LogisticRegressionModel(regularisation="l2", C=1.0)


@pytest.fixture
def random_forest_model():
    """
    Fixture creating a RandomForest model
    """
    return RandomForestClassifierModel(n_estimators=50, max_depth=10)
