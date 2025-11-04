
import cv2
import numpy as np
import structlog

log = structlog.get_logger(__name__)

def remove_noise_with_components(alpha_matte: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """
    Intelligently removes noise by deleting small, disconnected components.
    """
    binary_mask = (alpha_matte > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create a clean mask, starting with all zeros
    clean_mask = np.zeros_like(binary_mask)

    # Iterate from 1 to exclude the background label (0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 1

    # Apply the clean mask to the original alpha matte
    cleaned_alpha = alpha_matte * clean_mask
    return cleaned_alpha

def apply_guided_filter(alpha: np.ndarray, guide_image: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Applies a guided filter to refine the edges of the alpha matte.
    """
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    guide_uint8 = (guide_image * 255).astype(np.uint8)

    # Ensure guide image is 3-channel BGR for the filter
    if guide_uint8.ndim == 3 and guide_uint8.shape[2] == 3:
        guide_bgr = cv2.cvtColor(guide_uint8, cv2.COLOR_RGB2BGR)
    else:
        guide_bgr = guide_uint8

    try:
        # The guided filter function expects a BGR guide image
        filtered = cv2.ximgproc.guidedFilter(
            guide=guide_bgr,
            src=alpha_uint8,
            radius=radius,
            eps=eps * 255 * 255
        )
        return filtered.astype(np.float32) / 255.0
    except (AttributeError, cv2.error):
        log.warning(
            "Guided Filter not available (requires opencv-contrib-python). Skipping.",
            exc_info=True
        )
        return alpha