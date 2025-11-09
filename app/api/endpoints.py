from fastapi import APIRouter
from loguru import logger

router = APIRouter()

@router.get("/see_smth_nice")
async def get_docs_link():

    logger.info("Something nice link requested")
    return {
        "message": "Something nice:",
        "smth_nice_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }
