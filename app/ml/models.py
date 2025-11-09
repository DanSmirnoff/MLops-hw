import uuid

import joblib
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BaseModel:
    def __init__(self, model_class, **kwargs):
        self.model = model_class(**kwargs)
        # Это я загптшил каюсь
        self.model_id = str(uuid.uuid4())
        self.model_type = type(self.model).__name__
        self.is_trained = False
        self.training_data_shape = None
        self.test_size = 0.2

    def train(self, X, y):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.test_size)
        self.model.fit(X_train, y_train)

        self.is_trained = True
        self.training_data_shape = X.shape

        logger.info(f"Model ({self.model_type}) {self.model_id} trained successfully.")

        return {
            "model_id": self.model_id,
            "model_type": self.model_type
        }

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self.model, filepath)

    @classmethod
    def load(cls, filepath):
        model = joblib.load(filepath)
        return model


class LogisticRegressionModel(BaseModel):
    def __init__(self, regularisation='l2', C=0.1):
        super().__init__(LogisticRegression, penalty=regularisation, C=C, fit_intercept=True)


class RandomForestClassifierModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        super().__init__(RandomForestClassifier, n_estimators=n_estimators, max_depth=max_depth)


# Наверное стоит вынести в другое место?
MODEL_CLASSES = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest_classifier": RandomForestClassifierModel,
}
