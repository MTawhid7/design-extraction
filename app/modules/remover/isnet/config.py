# app/modules/remover/isnet/config.py
"""
Configuration for the IS-Net processing pipeline.
Updated to match IS-Net_V3 notebook with SMOOTH OUTLINE support.
"""
from pydantic_settings import BaseSettings

class ISNetSettings(BaseSettings):
    """Configuration for IS-Net (DIS) Pipeline - V3 with Smooth Outline"""

    # Model settings
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # Noise removal
    USE_NOISE_REMOVAL: bool = True
    NOISE_REMOVAL_THRESHOLD: float = 0.01
    MIN_COMPONENT_AREA: int = 50

    # Enhancement settings
    USE_CONTRAST_STRETCHING: bool = True
    CONTRAST_EXCLUDE_THRESHOLD: float = 0.05
    USE_GAMMA_CORRECTION: bool = True
    GAMMA_VALUE: float = 0.75

    # Guided Filtering settings
    GUIDANCE_PERCENTILE: int = 50

    # Core processing parameters
    CORE_BILATERAL_D: int = 9
    CORE_BILATERAL_SIGMA_COLOR: int = 75
    CORE_BILATERAL_SIGMA_SPACE: int = 75

    # ✨ SMOOTH OUTLINE SETTINGS (NEW)
    USE_SMOOTH_OUTLINE: bool = True
    # Controls the spread/softness of the outline. Smaller values = sharper edge.
    OUTLINE_BLUR_KERNEL_SIZE: int = 7
    # Controls the opacity of the blurred outline.
    OUTLINE_INTENSITY: float = 0.8

    # Morphological closing
    USE_MORPHOLOGICAL_CLOSING: bool = True
    CLOSING_KERNEL_SIZE: int = 3

    # Debugging - Set to True to save intermediate images to 'outputs' folder
    SHOW_DEBUG_IMAGES: bool = False

settings = ISNetSettings()