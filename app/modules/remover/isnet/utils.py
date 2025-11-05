# app/modules/remover/isnet/utils.py
"""
Helper functions for the IS-Net image processing pipeline.
Exact implementations from the research notebook.
"""
import cv2
import numpy as np
from PIL import Image
import uuid
from pathlib import Path

from .config import settings


def save_debug_image(image_data: np.ndarray, step_name: str):
    """
    Saves an image to the output directory for debugging if the setting is enabled.
    Note: This function is not from the notebook - it's added for production use.
    The notebook used matplotlib visualization instead.
    """
    if not settings.SHOW_DEBUG_IMAGES:
        return

    try:
        filename = f"debug_{uuid.uuid4().hex[:8]}_{step_name}.png"
        save_path = Path("outputs") / filename

        if image_data.dtype == np.float64 or image_data.dtype == np.float32:
            image_data = (image_data * 255).astype(np.uint8)

        Image.fromarray(image_data).save(save_path)
    except Exception:
        pass


def remove_noise_with_components(alpha_matte, threshold, min_area):
    """Remove small disconnected components (noise/artifacts)."""
    binary_mask = (alpha_matte > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    clean_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 1

    cleaned_alpha = alpha_matte * clean_mask
    return cleaned_alpha


def apply_contrast_stretching(alpha_matte, exclude_threshold=0.05):
    """Stretch alpha values to use full [0, 1] range."""
    # Exclude background noise
    foreground_mask = alpha_matte > exclude_threshold
    foreground_values = alpha_matte[foreground_mask]

    if len(foreground_values) == 0:
        return alpha_matte

    alpha_min = foreground_values.min()
    alpha_max = foreground_values.max()

    # Stretch values
    stretched = np.zeros_like(alpha_matte)
    stretched[foreground_mask] = (alpha_matte[foreground_mask] - alpha_min) / (alpha_max - alpha_min + 1e-8)
    stretched = np.clip(stretched, 0, 1)

    return stretched


def apply_gamma_correction(alpha_matte, gamma=0.75):
    """Apply gamma correction to brighten mid-tones."""
    return np.power(alpha_matte, gamma)


def refine_edges(alpha_matte, dilation_kernel=3, blur_kernel=5):
    """Refine edges using morphological operations and Gaussian blur."""
    alpha_uint8 = (alpha_matte * 255).astype(np.uint8)

    # Slight dilation to include edge pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    dilated = cv2.dilate(alpha_uint8, kernel, iterations=1)

    # Gaussian blur for smooth transitions
    blurred = cv2.GaussianBlur(dilated, (blur_kernel, blur_kernel), 0)

    return blurred.astype(np.float32) / 255.0


def apply_bilateral_filter(alpha_matte, d=9, sigma_color=75, sigma_space=75):
    """Bilateral filter for edge-preserving smoothing."""
    alpha_uint8 = (alpha_matte * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(alpha_uint8, d, sigma_color, sigma_space)
    return filtered.astype(np.float32) / 255.0


def apply_morphological_closing(alpha_matte, kernel_size=3):
    """Fill small holes in the design."""
    alpha_uint8 = (alpha_matte * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(alpha_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed.astype(np.float32) / 255.0


def apply_unsharp_mask(alpha_matte, strength=0.3, blur_size=5):
    """Sharpen edges using unsharp masking."""
    blurred = cv2.GaussianBlur(alpha_matte, (blur_size, blur_size), 0)
    sharpened = alpha_matte + strength * (alpha_matte - blurred)
    return np.clip(sharpened, 0, 1)


def compute_adaptive_thresholds(alpha_matte, exclude_threshold=0.05):
    """Compute adaptive thresholds based on percentiles."""
    foreground_mask = alpha_matte > exclude_threshold
    foreground_values = alpha_matte[foreground_mask]

    if len(foreground_values) == 0:
        return 0.65, 0.40, 0.15

    core_threshold = np.percentile(foreground_values, settings.CORE_PERCENTILE)
    transition_threshold = np.percentile(foreground_values, settings.TRANSITION_PERCENTILE)
    edge_threshold = np.percentile(foreground_values, settings.EDGE_PERCENTILE)

    return core_threshold, transition_threshold, edge_threshold