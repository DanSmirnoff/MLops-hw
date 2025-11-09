import requests
from test_generate_data import generate_spirals

url = "http://localhost:8000/api/v1/models/create_and_train"


spirals = generate_spirals(1000)

data = {
  "model_class": "random_forest_classifier",
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "features": spirals[0],
  "target": spirals[1]
}

response = requests.post(url, json=data)
print(response.json())
