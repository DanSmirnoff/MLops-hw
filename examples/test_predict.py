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

url = f"http://localhost:8000/api/v1/models/{model_id}/predict"


spirals = generate_spirals(20)

data = {
    "features": spirals[0]
}


response = requests.post(
    url,
    json=data,
    headers={"Content-Type": "application/json"}
)

print("Status:", response.status_code)
print("Response:", response.json())
