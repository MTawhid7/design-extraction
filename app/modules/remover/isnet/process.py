# app/modules/remover/isnet/process.py
"""
IS-Net background removal process.
Updated to match IS-Net_V3 notebook with SMOOTH OUTLINE implementation.
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
    Performs background removal on an image using the IS-Net model
    with the V3 pipeline for smooth outlines.
    """
    log.info("IS-Net V3: Starting process with smooth outline support.")
    original_size = input_image.size

    # Use EXACT normalization from notebook
    preprocess = transforms.Compose([
        transforms.Resize(settings.MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    log.info("IS-Net V3: Running inference...")
    input_tensor = preprocess(input_image).unsqueeze(0).to(model_manager.device)
    if model_manager.device.type == 'cuda':
        input_tensor = input_tensor.half()

    with torch.no_grad():
        outputs = model_manager.isnet_model(input_tensor)
        pred = torch.sigmoid(outputs[0][0].squeeze())

    # Resize to original size and convert to numpy
    base_alpha = transforms.Resize(original_size[::-1])(pred.clamp(0, 1).unsqueeze(0)).squeeze().cpu().numpy().astype(np.float32)
    utils.save_debug_image(base_alpha, "1_isnet_raw")
    log.info("IS-Net V3: Inference complete.")

    # --- STEP 1 & 2: PRE-PROCESSING & MASK CREATION ---
    log.info("IS-Net V3: Pre-processing and creating guidance masks...")

    if settings.USE_NOISE_REMOVAL:
        base_alpha = utils.remove_noise_with_components(
            base_alpha, settings.NOISE_REMOVAL_THRESHOLD, settings.MIN_COMPONENT_AREA)
        utils.save_debug_image(base_alpha, "2_noise_removed")

    if settings.USE_CONTRAST_STRETCHING:
        base_alpha = utils.apply_contrast_stretching(base_alpha, settings.CONTRAST_EXCLUDE_THRESHOLD)
        utils.save_debug_image(base_alpha, "3_contrast_stretched")

    if settings.USE_GAMMA_CORRECTION:
        base_alpha = utils.apply_gamma_correction(base_alpha, settings.GAMMA_VALUE)
        utils.save_debug_image(base_alpha, "4_gamma_corrected")

    # Create guidance masks
    foreground_values = base_alpha[base_alpha > 0.05]
    guidance_threshold = np.percentile(foreground_values, settings.GUIDANCE_PERCENTILE) if len(foreground_values) > 0 else 0.5
    core_mask = base_alpha > guidance_threshold
    edge_mask = ~core_mask

    log.info("IS-Net V3: Base alpha and masks prepared.")

    # --- STEP 3: CREATE SHARP DETAIL LAYER ---
    log.info("IS-Net V3: Creating the Sharp Detail Layer...")

    core_alpha = base_alpha * core_mask
    core_alpha_smoothed = utils.apply_bilateral_filter(
        core_alpha, settings.CORE_BILATERAL_D,
        settings.CORE_BILATERAL_SIGMA_COLOR,
        settings.CORE_BILATERAL_SIGMA_SPACE
    )
    processed_core = np.clip(core_alpha_smoothed * 1.25, 0, 1)
    preserved_edges = base_alpha * edge_mask
    sharp_detail_alpha = processed_core + preserved_edges

    utils.save_debug_image(sharp_detail_alpha, "5_sharp_detail_layer")
    log.info("IS-Net V3: Sharp detail layer created.")

    # --- STEP 4: CREATE AND COMPOSITE SMOOTH OUTLINE ---
    final_alpha = sharp_detail_alpha

    if settings.USE_SMOOTH_OUTLINE:
        log.info("IS-Net V3: Creating and Compositing Smooth Outline...")

        # Create the soft background layer by blurring the sharp version
        outline_layer = utils.create_blurred_layer(
            sharp_detail_alpha,
            settings.OUTLINE_BLUR_KERNEL_SIZE,
            settings.OUTLINE_INTENSITY
        )
        utils.save_debug_image(outline_layer, "6_smooth_outline_layer")

        # Composite the sharp layer ON TOP of the soft outline layer.
        # np.maximum ensures the sharp details are preserved perfectly.
        final_alpha = np.maximum(sharp_detail_alpha, outline_layer)
        utils.save_debug_image(final_alpha, "7_final_composite_with_outline")
        log.info("IS-Net V3: Smooth outline composited successfully.")

    # --- STEP 5: FINAL POLISH ---
    log.info("IS-Net V3: Final Polishing...")

    if settings.USE_MORPHOLOGICAL_CLOSING:
        final_alpha = utils.apply_morphological_closing(final_alpha, settings.CLOSING_KERNEL_SIZE)
        utils.save_debug_image(final_alpha, "8_morphological_closing")

    utils.save_debug_image(final_alpha, "9_final_alpha")

    # Create final RGBA image
    log.info("IS-Net V3: Creating final RGBA image.")
    rgb_array = np.array(input_image)
    alpha_uint8 = (final_alpha * 255).astype(np.uint8)
    rgba_array = np.dstack((rgb_array, alpha_uint8))
    final_rgba_image = Image.fromarray(rgba_array, mode='RGBA')

    log.info("IS-Net V3: Process completed with smooth outline.")
    return final_rgba_image