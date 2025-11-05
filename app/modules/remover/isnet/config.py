# app/modules/remover/isnet/config.py
"""
Configuration for the IS-Net processing pipeline.
"""
from pydantic_settings import BaseSettings

class ISNetSettings(BaseSettings):
    """Configuration for IS-Net (DIS) Pipeline - Adapted from Research Notebook"""

    # Model settings
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # Noise removal (more conservative for IS-Net)
    USE_NOISE_REMOVAL: bool = True
    NOISE_REMOVAL_THRESHOLD: float = 0.01
    MIN_COMPONENT_AREA: int = 50

    # Contrast enhancement
    USE_CONTRAST_STRETCHING: bool = True
    CONTRAST_EXCLUDE_THRESHOLD: float = 0.05

    # Gamma correction for brightness boost
    USE_GAMMA_CORRECTION: bool = True
    GAMMA_VALUE: float = 0.75

    # Adaptive thresholding (replaces fixed 0.92)
    USE_ADAPTIVE_THRESHOLDING: bool = True
    CORE_PERCENTILE: int = 75
    TRANSITION_PERCENTILE: int = 40
    EDGE_PERCENTILE: int = 15

    # Legacy threshold (used if adaptive is disabled)
    CORE_THRESHOLD_FIXED: float = 0.65

    # Edge refinement
    USE_EDGE_REFINEMENT: bool = True
    EDGE_DILATION_KERNEL: int = 3
    EDGE_BLUR_KERNEL: int = 5

    # Bilateral filter (better edge preservation)
    USE_BILATERAL_FILTER: bool = True
    BILATERAL_D: int = 9
    BILATERAL_SIGMA_COLOR: int = 75
    BILATERAL_SIGMA_SPACE: int = 75

    # Morphological closing (fill small holes)
    USE_MORPHOLOGICAL_CLOSING: bool = True
    CLOSING_KERNEL_SIZE: int = 3

    # Unsharp masking (edge sharpening)
    USE_UNSHARP_MASK: bool = True
    UNSHARP_STRENGTH: float = 0.3
    UNSHARP_BLUR_SIZE: int = 5

    # Debugging - Set to True to save intermediate images to 'outputs' folder
    SHOW_DEBUG_IMAGES: bool = False

settings = ISNetSettings()