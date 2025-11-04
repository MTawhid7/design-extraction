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

    # --- NEW CONCURRENCY SETTING ---
    # Controls how many upscaling tasks can run in parallel.
    # Set to 1 for GPUs with low VRAM (< 16GB).
    # Set to 4 for powerful GPUs (>= 24GB) like the NVIDIA L4.
    UPSCALER_CONCURRENCY_LIMIT: int = 1

    # Processing settings
    MAX_IMAGE_SIZE: int = 4096  # Maximum image dimension

    # Model settings
    DEVICE: str = "cuda"  # or "cpu"
    USE_FP16: bool = True  # Use half precision for faster inference

    # Gemini settings
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-image"
    GEMINI_TEMPERATURE: float = 0.4

    # RMBG settings
    RMBG_MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)
    RMBG_CORE_THRESHOLD: float = 0.90
    RMBG_USE_GUIDED_FILTER: bool = True
    RMBG_GUIDED_FILTER_RADIUS: int = 5
    RMBG_GUIDED_FILTER_EPS: float = 0.02
    RMBG_USE_NOISE_REMOVAL: bool = True
    RMBG_MIN_COMPONENT_AREA: int = 50

    # BiRefNet settings
    BIREFNET_MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)
    BIREFNET_CORE_THRESHOLD_PERCENTILE: float = 98.0
    BIREFNET_BOOST_MAX_PERCENTILE: float = 95.0
    BIREFNET_CORE_THRESHOLD_CLAMP: float = 0.98
    BIREFNET_BOOST_MAX_CLAMP: float = 0.97

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()