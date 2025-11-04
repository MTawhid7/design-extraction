import torch
import numpy as np
import structlog
from PIL import Image
from torchvision import transforms
from app.core.model_manager import ModelManager
from .config import settings
from . import utils as birefnet_utils

log = structlog.get_logger(__name__)

def run(input_image: Image.Image, model_manager: ModelManager) -> Image.Image:
    """
    Performs background removal on an image using the BiRefNet model.
    """
    log.info("BiRefNet: Starting process.")
    original_size = input_image.size

    # --- STEP 1: PREPROCESS & RUN INFERENCE ---
    log.info("BiRefNet: Preprocessing image and running inference...")
    transform = transforms.Compose([
        transforms.Resize(settings.MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(input_image).unsqueeze(0).to(model_manager.device)

    if model_manager.device.type == 'cuda':
        image_tensor = image_tensor.half()

    with torch.no_grad():
        # --- CORRECTED INFERENCE CALL ---
        # Transformers models often return a tuple; the main prediction is the last element.
        outputs = model_manager.birefnet_model(image_tensor)
        predicted_matte = outputs[-1].sigmoid()

    # --- The rest of the file is unchanged ---
    alpha_np = predicted_matte.squeeze().cpu().float().numpy()
    alpha_pil = Image.fromarray((alpha_np * 255).astype(np.uint8), mode='L')
    alpha_resized_pil = alpha_pil.resize(original_size, Image.LANCZOS)
    original_alpha = np.array(alpha_resized_pil) / 255.0
    log.info("BiRefNet: Alpha matte resized to original image dimensions.")

    core_threshold, soft_boost_max = birefnet_utils.get_dynamic_thresholds(
        original_alpha,
        settings.CORE_THRESHOLD_PERCENTILE, settings.CORE_THRESHOLD_CLAMP,
        settings.BOOST_MAX_PERCENTILE, settings.BOOST_MAX_CLAMP
    )
    log.info("BiRefNet: Dynamic parameters calculated", core_threshold=f"{core_threshold:.2f}", soft_boost_max=f"{soft_boost_max:.2f}")

    hard_boost_alpha = birefnet_utils.alpha_levels_adjustment(
        original_alpha, settings.HARD_BOOST_INPUT_MIN, settings.HARD_BOOST_INPUT_MAX, settings.HARD_BOOST_GAMMA
    )
    soft_boost_alpha = birefnet_utils.alpha_levels_adjustment(
        original_alpha, settings.SOFT_BOOST_INPUT_MIN, soft_boost_max, settings.SOFT_BOOST_GAMMA
    )
    combined_boosted_alpha = np.maximum(hard_boost_alpha, soft_boost_alpha)

    core_mask = original_alpha >= core_threshold
    solidified_core = np.where(core_mask, 1.0, 0.0)
    alpha_final = np.maximum(combined_boosted_alpha, solidified_core)

    rgb_array = np.array(input_image)
    alpha_uint8 = (alpha_final * 255).astype(np.uint8)
    rgba_array = np.dstack((rgb_array, alpha_uint8))
    final_rgba_image = Image.fromarray(rgba_array, mode='RGBA')

    log.info("BiRefNet: Process completed.")
    return final_rgba_image