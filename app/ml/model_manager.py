import json
import os
from .models import BaseModel, MODEL_CLASSES


class ModelManager:
    """
    Весь функционал связанный с тем что можно делать с моделями:
    (2, 3, 4 пункты)
    """

    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        os.makedirs(models_dir, exist_ok=True)
        self._load_existing_models()

    def _load_existing_models(self):
        metadata_file = os.path.join(self.models_dir, "models_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            for model_id, model_info in metadata.items():
                model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
                if os.path.exists(model_path):
                    model = BaseModel.load(model_path)
                    model.model_id = model_id
                    model.is_trained = True
                    model.training_data_shape = model_info.get("training_data_shape")
                    self.models[model_id] = model

    def _save_metadata(self):
        metadata = {}
        for model_id, model in self.models.items():
            if model.is_trained:
                metadata[model_id] = {
                    "model_type": type(model).__name__,
                    "training_data_shape": model.training_data_shape,
                }
        metadata_file = os.path.join(self.models_dir, "models_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

    def get_available_model_classes(self):
        return list(MODEL_CLASSES.keys())

    def create_model(self, model_class, **hyperparameters):
        if model_class not in MODEL_CLASSES:
            raise ValueError(f"Unknown model class: {model_class}")
        model = MODEL_CLASSES[model_class](**hyperparameters)
        self.models[model.model_id] = model
        return model.model_id

    def train_model(self, model_id, X, y):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        result = self.models[model_id].train(X, y)
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        self.models[model_id].save(model_path)
        self._save_metadata()
        return result

    def predict(self, model_id, X):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        return self.models[model_id].predict(X).tolist()

    def retrain_model(self, model_id, X, y):
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        return self.train_model(model_id, X, y)

    def delete_model(self, model_id):
        if model_id not in self.models:
            return False
        del self.models[model_id]
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        if os.path.exists(model_path):
            os.remove(model_path)
        self._save_metadata()
        return True

    # 2 функциии для себя
    def get_model_info(self, model_id):
        if model_id not in self.models:
            return None
        model = self.models[model_id]
        return {
            "model_id": model_id,
            "model_type": type(model).__name__,
            "is_trained": model.is_trained,
            "training_data_shape": model.training_data_shape,
        }

    def list_models(self):
        return [self.get_model_info(model_id) for model_id in self.models.keys()]
