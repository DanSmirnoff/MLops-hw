import uuid

import joblib
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class BaseModel:
    def __init__(self, model_class, task_type, **kwargs):
        self.model = model_class(**kwargs)
        # Это я загптшил
        self.model_id = str(uuid.uuid4())
        self.model_name = self.model.__class__.__name__
        self.training_data_shape = None
        self.test_size = 0.2

    def train(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.test_size)
        self.model.fit(X_train, y_train)

        self.training_data_shape = X.shape

        logger.info(f"Model ({self.model_name}) {self.model_id} trained successfully.")

        return {
            "model_id": self.model_id,
            "model_name": self.model_name
        }

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath):
        model = joblib.load(filepath)
        return model


class LinearRegressionModel(BaseModel):
    def __init__(self, fit_intercept=True, copy_X=True):
        super().__init__(LinearRegression, "regression", fit_intercept=fit_intercept, copy_X=copy_X)
