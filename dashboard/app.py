import json

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="ML Dashboard", layout="wide")
API_BASE = "http://localhost:8000/api/v1"

# Функция генерации спиралей (такая же есть в папке examples)
def generate_spirals(n_points=500, noise=0.5, revolutions=4):
    np.random.seed(42)

    theta1 = np.sqrt(np.random.rand(n_points)) * revolutions * np.pi
    r1 = theta1 + noise * np.random.randn(n_points)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    theta2 = np.sqrt(np.random.rand(n_points)) * revolutions * np.pi
    r2 = theta2 + noise * np.random.randn(n_points)
    x2 = r2 * np.cos(theta2 + np.pi)
    y2 = r2 * np.sin(theta2 + np.pi)

    X = np.vstack([np.column_stack([x1, y1]), 
                   np.column_stack([x2, y2])]).tolist()
    y = np.hstack([np.zeros(n_points), np.ones(n_points)]).tolist()

    return X, y


# api
def health_check():
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_models():
    try:
        response = requests.get(f"{API_BASE}/models")
        return response.json()["models"] if response.status_code == 200 else []
    except:
        return []


def train_model(data):
    try:
        response = requests.post(f"{API_BASE}/models/create_and_train", json=data)
        print(response.status_code)
        print(response.text)
        return response.json() if response.status_code == 200 else None
    except Exception:
        return None


def predict_model(model_id, features):
    try:
        response = requests.post(f"{API_BASE}/models/{model_id}/predict", json={"features": features})
        return response.json() if response.status_code == 200 else None
    except:
        return None


def delete_model(model_id):
    try:
        response = requests.delete(f"{API_BASE}/models/{model_id}")
        return response.status_code == 200
    except:
        return False


def get_smth_nice():
    try:
        response = requests.get(f"{API_BASE}/see_smth_nice")
        return response.json() if response.status_code == 200 else None
    except:
        return None


st.title("А я форточку на работе открыл")


st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Status", "Demo: Spirals", "Training", "Prediction", "Model Management", "Something Nice"])


if page == "Status":
    st.header("Service Status")
    health = health_check()
    if health:
        st.info(f"Service is running (v{health['version']})")
    else:
        st.error("Service unavailable")

    models = get_models()
    st.write(f"Trained models: {len(models)}")

# Где-то давно читал что это бенчмарк для классификаторов, решил свой сделать)
elif page == "Demo: Spirals":
    st.header("Demo: Spiral Classification")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Data Parameters")
        n_points = st.slider("Points per spiral", 100, 1000, 300)
        noise = st.slider("Noise level", 0.1, 2.0, 0.5)
        revolutions = st.slider("Number of revolutions", 1, 8, 3)

        st.subheader("Model Training")
        model_type = st.selectbox("Model type", ["random_forest_classifier", "logistic_regression"])

        if model_type == "random_forest_classifier":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Max depth", 1, 50, 10)
        else:
            C = st.slider("C parameter", 0.1, 10.0, 1.0)
            regularisation = st.selectbox("Regularization", [None, "l1", "l2", "elasticnet"])

        if st.button("Generate and Train"):
            X, y = generate_spirals(n_points, noise, revolutions)

            data = {
                "model_class": model_type,
                "features": X,
                "target": y
            }

            if model_type == "random_forest_classifier":
                data["hyperparameters"] = {"n_estimators": n_estimators, "max_depth": max_depth}
            else:
                data["hyperparameters"] = {"C": C, "regularisation": regularisation}

            result = train_model(data)

            if result:
                st.success("Model trained successfully")
                st.session_state.training_result = result
                st.session_state.demo_data = (X, y)
            else:
                st.error("Training failed")

    with col2:
        st.subheader("Data Visualization")

        if 'demo_data' in st.session_state:
            X, y = st.session_state.demo_data
            X_arr, y_arr = np.array(X), np.array(y)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(X_arr[y_arr==0, 0], X_arr[y_arr==0, 1], c='red', label='Class 0', alpha=0.7, s=20)
            ax.scatter(X_arr[y_arr==1, 0], X_arr[y_arr==1, 1], c='blue', label='Class 1', alpha=0.7, s=20)

            ax.set_title('Two Spirals Dataset')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            st.write(f"Data information:")
            st.write(f"- Total samples: {len(X)}")
            st.write(f"- Class 0: {sum(y_arr==0)} samples")
            st.write(f"- Class 1: {sum(y_arr==1)} samples")

            if 'training_result' in st.session_state:
                st.subheader("Model Testing")

                if st.button("Show Decision Boundary"):
                    model_id = st.session_state.training_result['model_id']

                    x_min, x_max = X_arr[:, 0].min() - 1, X_arr[:, 0].max() + 1
                    y_min, y_max = X_arr[:, 1].min() - 1, X_arr[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))

                    grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()
                    predictions = predict_model(model_id, grid_points)

                    if predictions:
                        Z = np.array(predictions['predictions']).reshape(xx.shape)

                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        ax2.contourf(xx, yy, Z, alpha=0.3)
                        ax2.scatter(X_arr[y_arr==0, 0], X_arr[y_arr==0, 1], c='red', label='Class 0', alpha=0.7, s=20)
                        ax2.scatter(X_arr[y_arr==1, 0], X_arr[y_arr==1, 1], c='blue', label='Class 1', alpha=0.7, s=20)
                        ax2.set_title('Model Decision Boundary')
                        ax2.legend()
                        st.pyplot(fig2)

elif page == "Training":
    st.header("Model Training")

    col1, col2 = st.columns(2)

    with col1:
        model_type = st.selectbox("Model type", ["random_forest_classifier", "logistic_regression"])

        if model_type == "random_forest_classifier":
            n_estimators = st.slider("Number of trees", 10, 500, 100)
            max_depth = st.slider("Max depth", 1, 50, 10)
        else:
            C = st.slider("C parameter", 0.1, 10.0, 1.0)
            regularisation = st.selectbox("Regularization", [None, "l1", "l2", "elasticnet"])

    with col2:
        st.subheader("Training Data")
        features_input = st.text_area("Features", '[[1,2,3], [4,5,6], [7,8,9]]')
        target_input = st.text_area("Target values", '[0, 1, 0]')

    if st.button("Train Model"):
        try:
            features = json.loads(features_input)
            target = json.loads(target_input)

            data = {
                "model_class": model_type,
                "features": features,
                "target": target
            }

            if model_type == "random_forest_classifier":
                data["hyperparameters"] = {"n_estimators": n_estimators, "max_depth": max_depth}
            else:
                data["hyperparameters"] = {"C": C, "regularisation": regularisation}

            result = train_model(data)
            if result:
                st.success(f"Model trained. ID: {result['model_id']}")
                st.write("Training results:", result)
            else:
                st.error("Training failed")

        except Exception as e:
            st.error(f"Error: {e}")

elif page == "Prediction":
    st.header("Prediction")

    models = get_models()
    if not models:
        st.write("No trained models available")
    else:
        model_options = {f"{m['model_id'][:8]}...": m['model_id'] for m in models if m['is_trained']}
        selected_model = st.selectbox("Select model", list(model_options.keys()))
        model_id = model_options[selected_model]

        features_input = st.text_area("Features for prediction", '[[1,2,3], [4,5,6]]')

        if st.button("Predict"):
            try:
                features = json.loads(features_input)
                result = predict_model(model_id, features)
                if result:
                    st.write("Predictions completed")
                    df = pd.DataFrame({
                        'Features': [str(f) for f in features],
                        'Prediction': result['predictions']
                    })
                    st.dataframe(df)
                else:
                    st.error("Prediction failed")

            except Exception as e:
                st.error(f"Error: {e}")

elif page == "Model Management":
    st.header("Model Management")

    models = get_models()
    if not models:
        st.write("No models available")
    else:
        for model in models:
            with st.expander(f"Model {model['model_id'][:8]}..."):
                col1, col2 = st.columns([3, 1])
                col1.write(f"ID: {model['model_id']}")
                col1.write(f"Trained: {'Yes' if model['is_trained'] else 'No'}")
                if 'model_type' in model:
                    col1.write(f"Model type: {model['model_type']}")

                if col2.button("Delete", key=model['model_id']):
                    if delete_model(model['model_id']):
                        st.write("Model deleted")
                        st.rerun()
                    else:
                        st.error("Delete failed")

elif page == "Something Nice":
    st.header("Something Nice")

    nice_response = get_smth_nice()

    if nice_response:
        video_url = nice_response.get('smth_nice', '')
        st.write("Here's something nice for you:")

        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be" in video_url:
            video_id = video_url.split("/")[-1]
        else:
            video_id = ""

        if video_id:
            youtube_html = f"""
            <div style="width: 100%; display: flex; justify-content: center;">
                <iframe width="853" height="480"
                        src="https://www.youtube.com/embed/{video_id}?autoplay=1"
                        frameborder="0"
                        allow="autoplay; encrypted-media"
                        allowfullscreen
                        style="border: none;">
                </iframe>
            </div>
            """
            st.components.v1.html(youtube_html, height=500)
        else:
            st.video(video_url)
    else:
        st.write("Could not load the nice thing. Service might be unavailable.")
