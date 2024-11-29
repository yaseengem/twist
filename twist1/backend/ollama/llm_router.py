
from fastapi import APIRouter

llm_router = APIRouter()

@llm_router.get("/status")
def get_status():
    return {"status": "LLM is running"}