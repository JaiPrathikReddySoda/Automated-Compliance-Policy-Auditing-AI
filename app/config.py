from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    # Model settings
    MODEL_PATH: Path = Path("models/deberta-v3-base-quantized")
    MODEL_NAME: str = "microsoft/deberta-v3-base"
    
    # Vector store settings
    INDEX_PATH: Path = Path("data/faiss_index")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_DIM: int = 768
    HNSW_M: int = 64
    HNSW_EF_CONSTRUCTION: int = 128
    
    # API settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Security
    CORS_ORIGINS: list[str] = ["*"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings() 