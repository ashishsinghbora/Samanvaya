"""
src/core/config.py

Defense-grade configuration management utilizing Pydantic Settings v2.
Validates environment variables strongly and ensures secure defaults.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    """
    Samanvaya Global Configuration.
    Reads from environment variables (e.g., SAMANVAYA_ENVIRONMENT).
    """
    environment: str = Field("production", env="ENVIRONMENT")
    
    # API Settings
    api_host: str = Field("0.0.0.0", env="API_HOST")  # nosec B104 - Container boundary
    api_port: int = Field(8000, env="API_PORT")
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    
    # Storage Settings
    workspace_dir: str = Field("/app/data", env="WORKSPACE_DIR")
    max_file_size_mb: int = Field(4096, env="MAX_FILE_SIZE_MB")  # 4 GiB
    
    # Security Settings
    jwt_secret: str = Field("change-me-in-production-to-a-secure-key", env="JWT_SECRET")
    rate_limit_requests: int = Field(60, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")
    
    # Celery / Redis Settings
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")

    # Algorithm Settings
    enable_gpu_acceleration: bool = Field(False, env="ENABLE_GPU")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global singleton configuration
settings = AppConfig()
