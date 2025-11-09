import os

import requests
from test_generate_data import generate_spirals

models_dir = "models"
if os.path.exists(models_dir):
    for filename in os.listdir(models_dir):
        if filename.endswith(".joblib"):
            model_id = filename.replace(".joblib", "")
            print(f"Модель: {model_id}")
            break
else:
    print('mda')

url = f"http://localhost:8000/api/v1/models/{model_id}/retrain"


spirals = generate_spirals(1000)

data = {
    "model_class": "random_forest",
    "hyperparameters": {
        "n_estimators": 150,
        "max_depth": 15
    },
    "features": spirals[0],
    "target": spirals[1]
}


response = requests.post(url, json=data)
result = response.json()


print("Done successfully")
