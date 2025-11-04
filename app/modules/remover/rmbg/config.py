
from pydantic_settings import BaseSettings

class RmbgSettings(BaseSettings):
    """Configuration for the RMBG 2.0 processing pipeline."""
    # The native training size for RMBG 2.0
    MODEL_INPUT_SIZE: tuple[int, int] = (1024, 1024)

    # --- INTELLIGENT NOISE REMOVAL ---
    USE_NOISE_REMOVAL: bool = True
    # Low threshold to create the initial binary mask for component analysis.
    NOISE_REMOVAL_THRESHOLD: float = 0.05
    # Any disconnected component with an area smaller than this will be erased.
    MIN_COMPONENT_AREA: int = 50

    # --- POST-PROCESSING (Core & Penumbra) ---
    # Defines the solid, opaque "core" of the image. A lower value is more forgiving.
    CORE_THRESHOLD: float = 0.90

    # --- FINAL POLISHING ---
    USE_GUIDED_FILTER: bool = True
    # A larger radius creates a more pronounced smoothing effect along the edges.
    GUIDED_FILTER_RADIUS: int = 5
    # A higher epsilon smooths over more minor texture variations.
    GUIDED_FILTER_EPS: float = 0.02


settings = RmbgSettings()