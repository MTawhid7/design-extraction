# app/modules/remover/isnet/process.py
"""
IS-Net background removal process.
The model import is handled by model_manager.py.
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
    Performs background removal on an image using the IS-Net model and an advanced post-processing pipeline.
    """
    log.info("IS-Net: Starting process.")
    original_size = input_image.size

    # Use EXACT normalization from notebook
    preprocess = transforms.Compose([
        transforms.Resize(settings.MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    ])

    log.info("IS-Net: Running inference...")
    input_tensor = preprocess(input_image).unsqueeze(0).to(model_manager.device)
    if model_manager.device.type == 'cuda':
        input_tensor = input_tensor.half()

    with torch.no_grad():
        outputs = model_manager.isnet_model(input_tensor)
        # Use EXACT prediction extraction from notebook
        pred = outputs[0][0].squeeze()
        pred = torch.sigmoid(pred)

    # Clamp and resize - EXACT match to notebook
    pred_normalized = pred.clamp(0, 1)
    original_alpha = transforms.Resize(original_size[::-1])(pred_normalized.unsqueeze(0)).squeeze().cpu().numpy()

    utils.save_debug_image(original_alpha, "1_isnet_raw")
    log.info("IS-Net: Inference complete.")

    cleaned_alpha = original_alpha
    if settings.USE_NOISE_REMOVAL:
        cleaned_alpha = utils.remove_noise_with_components(
            original_alpha, settings.NOISE_REMOVAL_THRESHOLD, settings.MIN_COMPONENT_AREA)
        utils.save_debug_image(cleaned_alpha, "2_noise_removed")

    contrast_alpha = cleaned_alpha
    if settings.USE_CONTRAST_STRETCHING:
        contrast_alpha = utils.apply_contrast_stretching(cleaned_alpha, settings.CONTRAST_EXCLUDE_THRESHOLD)
        utils.save_debug_image(contrast_alpha, "3_contrast_stretched")

    gamma_alpha = contrast_alpha
    if settings.USE_GAMMA_CORRECTION:
        gamma_alpha = utils.apply_gamma_correction(contrast_alpha, settings.GAMMA_VALUE)
        utils.save_debug_image(gamma_alpha, "4_gamma_corrected")

    edge_refined_alpha = gamma_alpha
    if settings.USE_EDGE_REFINEMENT:
        edge_refined_alpha = utils.refine_edges(
            gamma_alpha, settings.EDGE_DILATION_KERNEL, settings.EDGE_BLUR_KERNEL)
        utils.save_debug_image(edge_refined_alpha, "5_edge_refined")

    # Adaptive thresholding - EXACT match to notebook
    if settings.USE_ADAPTIVE_THRESHOLDING:
        core_thresh, trans_thresh, edge_thresh = utils.compute_adaptive_thresholds(edge_refined_alpha)
    else:
        core_thresh = settings.CORE_THRESHOLD_FIXED
        trans_thresh = 0.40
        edge_thresh = 0.15

    solid_mask = edge_refined_alpha > core_thresh
    transition_mask = (edge_refined_alpha > trans_thresh) & (~solid_mask)
    soft_edge_mask = (edge_refined_alpha > edge_thresh) & (~transition_mask) & (~solid_mask)

    processed_alpha = np.zeros_like(edge_refined_alpha)
    processed_alpha[solid_mask] = 1.0
    processed_alpha[transition_mask] = np.clip(edge_refined_alpha[transition_mask] * 1.2, 0, 1)
    processed_alpha[soft_edge_mask] = edge_refined_alpha[soft_edge_mask]

    utils.save_debug_image(processed_alpha, "6_multi_tier")

    bilateral_alpha = processed_alpha
    if settings.USE_BILATERAL_FILTER:
        bilateral_alpha = utils.apply_bilateral_filter(
            processed_alpha, settings.BILATERAL_D, settings.BILATERAL_SIGMA_COLOR, settings.BILATERAL_SIGMA_SPACE)
        utils.save_debug_image(bilateral_alpha, "7_bilateral_filtered")

    closing_alpha = bilateral_alpha
    if settings.USE_MORPHOLOGICAL_CLOSING:
        closing_alpha = utils.apply_morphological_closing(bilateral_alpha, settings.CLOSING_KERNEL_SIZE)
        utils.save_debug_image(closing_alpha, "8_morphological_closing")

    alpha_final = closing_alpha
    if settings.USE_UNSHARP_MASK:
        alpha_final = utils.apply_unsharp_mask(
            closing_alpha, settings.UNSHARP_STRENGTH, settings.UNSHARP_BLUR_SIZE)
        utils.save_debug_image(alpha_final, "9_final_alpha")

    log.info("IS-Net: Creating final RGBA image.")
    rgb_array = np.array(input_image)
    alpha_uint8 = (alpha_final * 255).astype(np.uint8)
    rgba_array = np.dstack((rgb_array, alpha_uint8))
    final_rgba_image = Image.fromarray(rgba_array, mode='RGBA')

    log.info("IS-Net: Process completed.")
    return final_rgba_image