"""
Application configuration using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    GEMINI_API_KEY: str

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    WORKERS: int = 1  # Single worker for GPU management

    # Storage settings
    OUTPUT_DIR: str = "outputs"
    BASE_URL: str = "https://yourcdn.com"  # Base URL for generated images

    # --- MODIFIED: CONCURRENCY SETTING ---
    # Controls how many upscaling tasks can run in parallel.
    # Reduced to 2 since we now process half the number of images.
    UPSCALER_CONCURRENCY_LIMIT: int = 2

    # Processing settings
    MAX_IMAGE_SIZE: int = 4096  # Maximum image dimension

    # Model settings
    DEVICE: str = "cuda"  # or "cpu"
    USE_FP16: bool = True  # Use half precision for faster inference

    # Gemini settings
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-image"
    GEMINI_TEMPERATURE: float = 0.4

    # --- REMOVED: RMBG settings ---

    # --- REMOVED: BiRefNet settings ---

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()