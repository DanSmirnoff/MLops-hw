from fastapi import APIRouter, HTTPException, status
import numpy as np
import logging

from .models import (
    HealthResponse, ModelClassResponse, TrainRequest, TrainResponse,
    PredictRequest, PredictResponse, ModelsListResponse, ModelInfo,
    SmthNiceResponse
)
from app.ml.model_manager import ModelManager


logger = logging.getLogger(__name__)

router = APIRouter()
model_manager = ModelManager()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    logger.info("Health check requested")
    return HealthResponse(status="healthy", version="0.1.0")


@router.get("/models/available", response_model=ModelClassResponse)
async def get_available_models():
    logger.info("Available models requested")
    available_models = model_manager.get_available_model_classes()
    logger.debug(f"Returning available models: {available_models}")
    return ModelClassResponse(available_models=available_models)


@router.post("/models/create_and_train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    logger.info(f"Training request for model class: {request.model_class}")

    try:
        hyperparams = request.hyperparameters.dict() if request.hyperparameters else {}

        # Важная строка!!! Позволяет использовать разные гиперпараметры:
        # Если гиперпараметр None, то он тут игнорируется
        hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

        model_id = model_manager.create_model(request.model_class, **hyperparams)

        X = np.array(request.features)
        y = np.array(request.target)

        logger.debug(f"Training data shape: {X.shape}, target shape: {y.shape}")

        result = model_manager.train_model(model_id, X, y)

        logger.info(f"Training completed successfully for model {model_id}")

        return TrainResponse(**result)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models/{model_id}/predict", response_model=PredictResponse)
async def predict(model_id, request: PredictRequest):
    logger.info(f"Prediction request for model: {model_id}")
    try:
        X = np.array(request.features)
        logger.debug(f"Prediction data shape: {X.shape}")
        predictions = model_manager.predict(model_id, X)
        logger.info(f"Prediction completed successfully for model {model_id}")
        return PredictResponse(predictions=predictions, model_id=model_id)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models/{model_id}/retrain", response_model=TrainResponse)
async def retrain_model(model_id, request):
    logger.info(f"Retraining request for model: {model_id}")

    try:
        X = np.array(request.features)
        y = np.array(request.target)

        logger.debug(f"Retraining data shape: {X.shape}, target shape: {y.shape}")

        result = model_manager.retrain_model(model_id, X, y)

        logger.info(f"Retraining completed successfully for model {model_id}")

        return TrainResponse(**result)

    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/models/{model_id}")
async def delete_model(model_id):
    logger.info(f"Delete request for model: {model_id}")

    success = model_manager.delete_model(model_id)
    if not success:
        logger.warning(f"Attempted to delete non-existent model {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    return {"message": f"Model {model_id} deleted successfully"}


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    logger.info("Models list requested")
    models_info = model_manager.list_models()
    logger.debug(f"Returning {len(models_info)} models")
    return ModelsListResponse(models=models_info)


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id):
    logger.info(f"Model info requested for: {model_id}")

    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        logger.warning(f"Model info requested for non-existent model {model_id}")
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(**model_info)


@router.get("/see_smth_nice")
async def get_docs_link():
    logger.info("Something nice link requested")

    return SmthNiceResponse(smth_nice="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
