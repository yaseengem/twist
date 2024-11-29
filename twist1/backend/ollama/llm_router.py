from fastapi import APIRouter
from .endpoints import status, models

llm_router = APIRouter()
llm_router.include_router(status.router, prefix="/status", tags=["status"])
# llm_router.include_router(models.router, prefix="/models", tags=["models"])