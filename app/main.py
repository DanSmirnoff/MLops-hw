import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.endpoints import router as api_router
from app.core.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    logger.info("Starting ML Service API")
    yield
    logger.info("Shutting down ML Service API")


app = FastAPI(
    title="ML Service API",
    description="Buratino utonul",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


app.include_router(api_router, prefix="/api/v1")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
