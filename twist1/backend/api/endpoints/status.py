
from fastapi import APIRouter

def get_status():
    return {"status": "API is running"}