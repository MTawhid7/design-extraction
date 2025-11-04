from pydantic_settings import BaseSettings

class RmbgSettings(BaseSettings):
    """Configuration for the RMBG 2.0 processing pipeline, aligned with research notebook."""
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # --- INTELLIGENT NOISE REMOVAL (Fine-tuned) ---
    USE_NOISE_REMOVAL: bool = True
    NOISE_REMOVAL_THRESHOLD: float = 0.02
    MIN_COMPONENT_AREA: int = 25

    # --- POST-PROCESSING (Core & Penumbra) ---
    # Defines the solid, opaque "core" of the image.
    CORE_THRESHOLD: float = 0.95
    # --- ADDED FROM NOTEBOOK ---
    # Defines the semi-transparent "penumbra" (soft edges). Any pixel below this is considered noise.
    PENUMBRA_LOWER_BOUND: float = 0.05

    # --- FINAL POLISHING (Fine-tuned) ---
    USE_GUIDED_FILTER: bool = True
    GUIDED_FILTER_RADIUS: int = 3
    GUIDED_FILTER_EPS: float = 0.01

settings = RmbgSettings()