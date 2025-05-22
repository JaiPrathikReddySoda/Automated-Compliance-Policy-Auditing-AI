from __future__ import annotations

import logging
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Policy Auditor API",
        description="API for querying policy documents using DeBERTa-v3",
        version="1.0.0",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    app.include_router(router, prefix="/api/v1")
    
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize services on startup."""
        logger.info("Starting up Policy Auditor API")
    
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Clean up on shutdown."""
        logger.info("Shutting down Policy Auditor API")
    
    return app


app = create_app() 