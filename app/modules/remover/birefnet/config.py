from pydantic_settings import BaseSettings

class BirefnetSettings(BaseSettings):
    """Configuration for the BiRefNet processing pipeline, aligned with research notebook."""
    # --- INCREASED FOR HIGHER QUALITY ---
    MODEL_INPUT_SIZE: tuple[int, int] = (2048, 2048)

    # --- DYNAMIC ANALYSIS PARAMETERS ---
    CORE_THRESHOLD_PERCENTILE: float = 98.0
    BOOST_MAX_PERCENTILE: float = 95.0
    CORE_THRESHOLD_CLAMP: float = 0.98
    BOOST_MAX_CLAMP: float = 0.97

    # --- MULTI-STAGE BOOST PARAMETERS (Fine-tuned) ---
    HARD_BOOST_INPUT_MIN: float = 0.10
    HARD_BOOST_INPUT_MAX: float = 0.40
    HARD_BOOST_GAMMA: float = 0.7
    SOFT_BOOST_INPUT_MIN: float = 0.10
    SOFT_BOOST_GAMMA: float = 0.8

settings = BirefnetSettings()