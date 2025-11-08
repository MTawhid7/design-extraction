# app/modules/remover/isnet/utils.py
"""
Helper functions for IS-Net V6 - ULTIMATE PRINT-QUALITY Pipeline.
Research-backed minimal processing approach.
"""
import cv2
import numpy as np
from PIL import Image
import uuid
from pathlib import Path
from scipy.ndimage import gaussian_filter

from .config import settings



def smart_noise_removal(alpha_matte, min_size=50, threshold=0.01):
    """
    Conservative connected component analysis.
    Remove only tiny isolated artifacts.
    """
    binary = (alpha_matte > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean_mask = np.zeros_like(binary)

    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_mask[labels == i] = 1

    return alpha_matte * clean_mask


def adaptive_contrast_enhancement(alpha_matte, gamma=1.0, low_pct=1.0, high_pct=99.0, strength=0.5):
    """
    Proper contrast stretching that preserves gradients.

    Calculate percentiles on ENTIRE alpha (not just foreground)
    to avoid crushing semi-transparent regions.
    """
    # Gamma correction first (optional)
    if gamma != 1.0:
        alpha_matte = np.power(alpha_matte, gamma)

    # Calculate percentiles on ENTIRE alpha range
    non_zero = alpha_matte[alpha_matte > 0]

    if len(non_zero) == 0:
        return alpha_matte

    p_low = np.percentile(non_zero, low_pct)
    p_high = np.percentile(non_zero, high_pct)

    # Stretch only non-zero regions
    stretched = alpha_matte.copy()
    mask = alpha_matte > 0

    stretched[mask] = np.clip(
        (alpha_matte[mask] - p_low) / (p_high - p_low + 1e-8),
        0, 1
    )

    # Blend with original
    result = strength * stretched + (1 - strength) * alpha_matte

    return result


def apply_gaussian_smoothing(alpha_matte, sigma=0.5):
    """Gentle Gaussian blur for subtle smoothing."""
    return gaussian_filter(alpha_matte, sigma=sigma)


def feather_edges(alpha_matte, radius=1):
    """Gentle Gaussian blur for soft edges (optional)."""
    if radius <= 0:
        return alpha_matte

    sigma = radius / 3.0
    return gaussian_filter(alpha_matte, sigma=sigma)