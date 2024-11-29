from fastapi import APIRouter
from .endpoints.status import get_status

llm_router = APIRouter()

llm_router.get("/status")(get_status)