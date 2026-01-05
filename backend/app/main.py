from fastapi import FastAPI

from backend.app.api import search_v2 as search_api

app = FastAPI(title="RAG Graph Service")

app.include_router(search_api.router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "RAG Graph Service running. Use /api/search"}
