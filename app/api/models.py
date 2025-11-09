from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelClassResponse(BaseModel):
    available_models: List[str]


class Hyperparameters(BaseModel):
    # Всё по дефолту None потому что они выкидываются потом и благодаря этому
    # мы можем задавать разные гиперпараметры для разных моделей
    n_estimators: Optional[int] = Field(None, ge=1, le=1000)
    max_depth: Optional[int] = Field(None, ge=1, le=100)
    C: Optional[float] = Field(None, ge=0.1, le=10.0)
    regularisation: Optional[str] = Field(None, pattern="^(l1|l2|elasticnet)$")


class TrainRequest(BaseModel):
    model_class: str
    hyperparameters: Optional[Hyperparameters] = None
    features: List[List[float]]
    target: List[int]


class TrainResponse(BaseModel):
    model_id: str
    model_type: str


class PredictRequest(BaseModel):
    features: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]
    model_id: str


class ModelInfo(BaseModel):
    model_id: str
    model_type: str
    is_trained: bool


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]


class SmthNiceResponse(BaseModel):
    smth_nice: str
