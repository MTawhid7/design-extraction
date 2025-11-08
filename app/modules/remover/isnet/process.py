# app/modules/remover/isnet/process.py
"""
IS-Net V6 - ULTIMATE PRINT-QUALITY (2025 Optimized)
====================================================
Research-backed improvements:
✓ Fixed contrast stretch algorithm
✓ Gradient-preserving morphology
✓ Edge-aware denoising
✓ Minimal processing philosophy
"""
import torch
import numpy as np
import structlog
from PIL import Image
from torchvision import transforms

from .config import settings
from . import utils

log = structlog.get_logger(__name__)


def run(input_image: Image.Image, model_manager: "ModelManager") -> Image.Image:
    """
    Performs background removal using IS-Net V6 pipeline.

    Philosophy: Research-backed minimal processing
    - Smart noise removal (conservative)
    - Adaptive contrast enhancement (gradient-preserving)
    - Gentle Gaussian smoothing
    - Optional edge feathering
    """
    log.info("IS-Net V6: Starting ULTIMATE PRINT-QUALITY processing.")
    original_size = input_image.size

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: IS-Net Inference
    # ═══════════════════════════════════════════════════════════════════
    log.info("IS-Net V6 [Stage 1/4]: IS-Net Inference...")

    # Use EXACT normalization from V6 notebook
    preprocess = transforms.Compose([
        transforms.Resize(settings.MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(model_manager.device)
    if model_manager.device.type == 'cuda':
        input_tensor = input_tensor.half()

    with torch.no_grad():
        outputs = model_manager.isnet_model(input_tensor)
        pred = torch.sigmoid(outputs[0][0].squeeze())

    # Resize to original size and convert to numpy
    pred_resized = transforms.Resize(original_size[::-1])(pred.clamp(0, 1).unsqueeze(0))
    base_alpha = pred_resized.squeeze().cpu().numpy().astype(np.float32)

    utils.save_debug_image(base_alpha, "1_isnet_raw_output")
    log.info("IS-Net V6 [Stage 1/4]: Inference complete.")

    current_alpha = base_alpha.copy()

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 2: Smart Noise Removal
    # ═══════════════════════════════════════════════════════════════════
    log.info("IS-Net V6 [Stage 2/4]: Smart Noise Removal...")

    if settings.USE_NOISE_REMOVAL:
        current_alpha = utils.smart_noise_removal(
            current_alpha,
            settings.MIN_COMPONENT_SIZE,
            settings.NOISE_THRESHOLD
        )
        utils.save_debug_image(current_alpha, "2_noise_removed")
        log.info("IS-Net V6 [Stage 2/4]: Noise removal complete.")
    else:
        log.info("IS-Net V6 [Stage 2/4]: Noise removal skipped.")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 3: Adaptive Contrast Enhancement
    # ═══════════════════════════════════════════════════════════════════
    log.info("IS-Net V6 [Stage 3/4]: Adaptive Contrast Enhancement...")

    if settings.USE_ADAPTIVE_CONTRAST:
        current_alpha = utils.adaptive_contrast_enhancement(
            current_alpha,
            settings.GAMMA_CORRECTION,
            settings.STRETCH_PERCENTILE_LOW,
            settings.STRETCH_PERCENTILE_HIGH,
            settings.STRETCH_STRENGTH
        )
        utils.save_debug_image(current_alpha, "3_contrast_enhanced")
        log.info("IS-Net V6 [Stage 3/4]: Contrast enhancement complete.")
    else:
        log.info("IS-Net V6 [Stage 3/4]: Contrast enhancement skipped.")

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 4: Gentle Gaussian Smoothing
    # ═══════════════════════════════════════════════════════════════════
    if settings.USE_GAUSSIAN_SMOOTH:
        log.info(f"IS-Net V6 [Stage 4/4]: Gentle Gaussian Smoothing (σ={settings.GAUSSIAN_SIGMA})...")
        current_alpha = utils.apply_gaussian_smoothing(
            current_alpha,
            settings.GAUSSIAN_SIGMA
        )
        utils.save_debug_image(current_alpha, "4_gaussian_smoothed")
        log.info("IS-Net V6 [Stage 4/4]: Gaussian smoothing complete.")
    else:
        log.info("IS-Net V6 [Stage 4/4]: Gaussian smoothing skipped.")

    # ═══════════════════════════════════════════════════════════════════
    # Optional: Edge Feathering
    # ═══════════════════════════════════════════════════════════════════
    if settings.USE_EDGE_FEATHERING:
        log.info(f"IS-Net V6 [Optional]: Edge Feathering (radius={settings.FEATHER_RADIUS})...")
        current_alpha = utils.feather_edges(current_alpha, settings.FEATHER_RADIUS)
        utils.save_debug_image(current_alpha, "5_edge_feathered")
        log.info("IS-Net V6 [Optional]: Edge feathering complete.")

    final_alpha = current_alpha

    # ═══════════════════════════════════════════════════════════════════
    # Create Final RGBA Image
    # ═══════════════════════════════════════════════════════════════════
    utils.save_debug_image(final_alpha, "6_final_alpha")
    log.info("IS-Net V6: Creating final RGBA image.")

    rgb_array = np.array(input_image)
    alpha_uint8 = (np.clip(final_alpha, 0, 1) * 255).astype(np.uint8)
    rgba_array = np.dstack((rgb_array, alpha_uint8))
    final_rgba_image = Image.fromarray(rgba_array, mode='RGBA')

    log.info("IS-Net V6: ULTIMATE PRINT-QUALITY processing complete.")
    return final_rgba_image