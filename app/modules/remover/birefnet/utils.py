
import numpy as np

def alpha_levels_adjustment(alpha: np.ndarray, input_min: float, input_max: float, gamma: float = 1.0) -> np.ndarray:
    """Remaps the alpha channel values to enhance contrast."""
    # Ensure we don't divide by zero
    if abs(input_max - input_min) < 1e-6:
        return np.where(alpha >= input_min, 1.0, 0.0).astype(np.float32)

    # Clip values to the specified input range
    alpha_clipped = np.clip(alpha, input_min, input_max)

    # Normalize the clipped values to a 0-1 range
    alpha_normalized = (alpha_clipped - input_min) / (input_max - input_min)

    # Apply gamma correction
    alpha_gamma_corrected = np.power(alpha_normalized, gamma)

    # For pixels that were originally below the minimum threshold, retain their original value
    # instead of crushing them to zero. This preserves very faint details.
    final_alpha = np.where(alpha < input_min, alpha, alpha_gamma_corrected)

    return final_alpha.astype(np.float32)

def get_dynamic_thresholds(
    alpha_matte: np.ndarray,
    core_percentile: float, core_clamp: float,
    boost_percentile: float, boost_clamp: float
) -> tuple[float, float]:
    """
    Analyzes the alpha matte to calculate dynamic thresholds for processing.
    """
    # Consider only non-trivial alpha values for a more accurate analysis
    non_zero_alpha = alpha_matte[alpha_matte > 0.01]

    if len(non_zero_alpha) == 0:
        # If the image is entirely black, return default safe values
        return core_clamp, boost_clamp

    core_threshold = min(np.percentile(non_zero_alpha, core_percentile), core_clamp)
    soft_boost_max = min(np.percentile(non_zero_alpha, boost_percentile), boost_clamp)

    return core_threshold, soft_boost_max