from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    MINIO_ENDPOINT: str = Field("localhost:9000")
    MINIO_ACCESS_KEY: str = Field("minioadmin")
    MINIO_SECRET_KEY: str = Field("minioadmin")
    MINIO_BUCKET: str = Field("ml-models")
    MINIO_SECURE: bool = Field(False)

    DVC_REMOTE: str = Field("minio")
    DVC_BUCKET: str = Field("ml-datasets")

    class Config:
        env_file = ".env"


settings = Settings()
