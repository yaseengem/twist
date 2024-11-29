from fastapi import APIRouter
from .endpoints.status import get_status

api_router = APIRouter()

api_router.get("/status")(get_status)