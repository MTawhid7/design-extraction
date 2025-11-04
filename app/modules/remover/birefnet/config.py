
from pydantic_settings import BaseSettings

class BirefnetSettings(BaseSettings):
    """Configuration for the BiRefNet processing pipeline."""
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # --- DYNAMIC ANALYSIS PARAMETERS ---
    # Percentiles used to dynamically calculate thresholds based on the image's alpha matte.
    CORE_THRESHOLD_PERCENTILE: float = 98.0
    BOOST_MAX_PERCENTILE: float = 95.0
    # Safety clamps to prevent extreme values from distorting the result.
    CORE_THRESHOLD_CLAMP: float = 0.98
    BOOST_MAX_CLAMP: float = 0.97

    # --- MULTI-STAGE BOOST PARAMETERS ---
    # Stage 1: Aggressive boost for very faint details.
    HARD_BOOST_INPUT_MIN: float = 0.10
    HARD_BOOST_INPUT_MAX: float = 0.40
    HARD_BOOST_GAMMA: float = 0.7

    # Stage 2: Gentler, dynamic boost for mid-tones.
    SOFT_BOOST_INPUT_MIN: float = 0.10
    SOFT_BOOST_GAMMA: float = 0.8


settings = BirefnetSettings()