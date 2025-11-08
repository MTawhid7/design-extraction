# app/modules/remover/isnet/config.py
"""
Configuration for IS-Net V6 - ULTIMATE PRINT-QUALITY (2025 Optimized)
Research-backed improvements with minimal processing philosophy.
"""
from pydantic_settings import BaseSettings


class ISNetSettings(BaseSettings):
    """Configuration for IS-Net V6 Pipeline - Ultimate Print Quality"""

    # Model settings
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # === STAGE 1: Smart Noise Removal ===
    USE_NOISE_REMOVAL: bool = True
    MIN_COMPONENT_SIZE: int = 50        # Smaller = preserve more detail
    NOISE_THRESHOLD: float = 0.01       # Very conservative

    # === STAGE 2: Adaptive Contrast Enhancement ===
    USE_ADAPTIVE_CONTRAST: bool = True
    GAMMA_CORRECTION: float = 1.0       # 1.0 = neutral, <1 brighten, >1 darken
    STRETCH_PERCENTILE_LOW: float = 1.0    # Work on ENTIRE alpha range
    STRETCH_PERCENTILE_HIGH: float = 99.0
    STRETCH_STRENGTH: float = 0.5       # Subtle: 0.3-0.7 ideal

    # === STAGE 3: Gentle Gaussian Smoothing ===
    USE_GAUSSIAN_SMOOTH: bool = True
    GAUSSIAN_SIGMA: float = 0.5         # 0.5-2.0 for subtle smoothing

    # === STAGE 4: Feathering (OPTIONAL) ===
    USE_EDGE_FEATHERING: bool = False
    FEATHER_RADIUS: int = 1             # Subtle: 1-2

    # Debugging - Set to True to save intermediate images to 'outputs' folder
    SHOW_DEBUG_IMAGES: bool = False


settings = ISNetSettings()