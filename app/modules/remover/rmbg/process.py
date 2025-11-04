# app/modules/remover/rmbg/process.py

import torch
import numpy as np
import structlog
from PIL import Image
from app.core.model_manager import ModelManager # <-- CORRECTED IMPORT PATH
from .config import settings
from . import utils as rmbg_utils

log = structlog.get_logger(__name__)

def run(input_image: Image.Image, model_manager: ModelManager) -> Image.Image:
    """
    Performs background removal on an image using the RMBG-2.0 model.

    Args:
        input_image: The PIL Image object (RGB).
        model_manager: The manager holding the pre-loaded models.

    Returns:
        A PIL Image object with a transparent background (RGBA).
    """
    log.info("RMBG: Starting process.")
    original_size = input_image.size

    # --- STEP 1: GET SOFT MASK WITH RMBG 2.0 ---
    log.info("RMBG: Generating soft mask...")
    inputs = model_manager.rmbg_processor(images=input_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(model_manager.device)

    if model_manager.device.type == 'cuda':
        pixel_values = pixel_values.half()

    with torch.no_grad():
        outputs = model_manager.rmbg_model(pixel_values)
        raw_mask = outputs[-1].squeeze()

    original_alpha_tensor = torch.sigmoid(raw_mask).cpu().float()

    # Resize mask to original image size
    alpha_pil = Image.fromarray(
        (original_alpha_tensor.numpy() * 255).astype(np.uint8)
    ).resize(original_size, Image.LANCZOS)

    original_alpha = np.array(alpha_pil) / 255.0
    log.info("RMBG: Soft mask generated.")

    # --- STEP 2: INTELLIGENT NOISE REMOVAL ---
    cleaned_alpha = original_alpha
    if settings.USE_NOISE_REMOVAL:
        log.info("RMBG: Applying noise removal...")
        cleaned_alpha = rmbg_utils.remove_noise_with_components(
            original_alpha,
            settings.NOISE_REMOVAL_THRESHOLD,
            settings.MIN_COMPONENT_AREA
        )

    # --- STEP 3: CORE POST-PROCESSING ---
    log.info("RMBG: Applying core threshold...")
    processed_alpha = np.where(cleaned_alpha > settings.CORE_THRESHOLD, 1.0, cleaned_alpha)

    # --- STEP 4: FINAL POLISH WITH GUIDED FILTER ---
    alpha_final = processed_alpha
    if settings.USE_GUIDED_FILTER:
        log.info("RMBG: Applying guided filter...")
        guide_image = np.array(input_image) / 255.0
        alpha_final = rmbg_utils.apply_guided_filter(
            processed_alpha,
            guide_image,
            radius=settings.GUIDED_FILTER_RADIUS,
            eps=settings.GUIDED_FILTER_EPS
        )

    # --- STEP 5: CREATE FINAL RGBA IMAGE ---
    log.info("RMBG: Creating final RGBA image.")
    rgb_array = np.array(input_image)
    alpha_uint8 = (alpha_final * 255).astype(np.uint8)
    rgba_array = np.dstack((rgb_array, alpha_uint8))
    final_rgba_image = Image.fromarray(rgba_array, mode='RGBA')

    log.info("RMBG: Process completed.")
    return final_rgba_image