#!/usr/bin/env python3
"""
Start script for the RAG Graph Search Service v2
"""

import os
import sys
import argparse
from deploy_search_v2 import create_app
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Start RAG Graph Search Service v2")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=os.getenv("PORT", 8000), help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=(os.getenv("RELOAD", "false").lower() == "true"), 
                       help="Enable auto-reload (development)")
    parser.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", 1)), help="Number of worker processes")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"), help="Logging level")
    
    args = parser.parse_args()
    
    logger.info(f"Starting RAG Graph Search Service v2 on {args.host}:{args.port}")
    logger.info(f"Configuration: reload={args.reload}, workers={args.workers}, log_level={args.log_level}")
    
    app = create_app()
    
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Error occurred while running the server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()