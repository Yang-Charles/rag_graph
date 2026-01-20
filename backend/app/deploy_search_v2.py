from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import os
import logging
from backend.app.api import search_v2


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application
    """
    logger.info("Starting up search_v2 service...")
    
    # Initialize services here if needed
    # For example, connecting to databases, loading models, etc.
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down search_v2 service...")


def create_app():
    """
    Create and configure FastAPI application
    """
    app = FastAPI(
        title="RAG Graph Search Service v2",
        description="Multi-modal search service with semantic, full-text, image and knowledge graph capabilities",
        version="2.0.0",
        lifespan=lifespan
    )
    
    # Include the search router
    app.include_router(search_v2.router, prefix="/api/v2", tags=["search"])
    
    @app.get("/")
    def root():
        return {"message": "RAG Graph Search Service v2 is running", "status": "ok"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "search_v2"}
    
    return app


if __name__ == "__main__":
    # Get configuration from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting search_v2 service on {host}:{port}")
    
    app = create_app()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        workers=int(os.getenv("WORKERS", 1))
    )