from fastapi import FastAPI
from api.api_router import api_router
from ollama.llm_router import llm_router

app = FastAPI()

app.include_router(api_router, prefix="/api")
app.include_router(llm_router, prefix="/ollama")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
