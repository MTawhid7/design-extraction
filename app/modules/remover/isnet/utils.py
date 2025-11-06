# app/modules/remover/isnet/utils.py
"""
Helper functions for the IS-Net image processing pipeline.
Updated to match IS-Net_V3 notebook implementation exactly.
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


def create_blurred_layer(alpha_matte, kernel_size, intensity):
    """
    Applies a Gaussian blur and intensity to create a soft layer.
    This is the new function from IS-Net_V3 for smooth outlines.
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1

    # Apply blur
    blurred_matte = cv2.GaussianBlur(alpha_matte, (kernel_size, kernel_size), 0)

    # Apply intensity and clip
    soft_layer = np.clip(blurred_matte * intensity, 0, 1)

    return soft_layer