# app/modules/upscaler/process.py

import structlog
import numpy as np
from PIL import Image
from app.core.model_manager import ModelManager

log = structlog.get_logger(__name__)


def run(input_image: Image.Image, model_manager: ModelManager) -> Image.Image:
    """
    Upscales an RGBA image using Real-ESRGAN while preserving transparency.

    Args:
        input_image: PIL Image in RGBA mode
        model_manager: Manager holding the pre-loaded Real-ESRGAN model

    Returns:
        Upscaled PIL Image in RGBA mode
    """
    log.info("Real-ESRGAN: Starting upscale", input_size=input_image.size)

    if input_image.mode != "RGBA":
        raise ValueError(f"Expected RGBA image, got {input_image.mode}")

    upsampler = model_manager.realesrgan_model
    if not upsampler:
        raise RuntimeError("Real-ESRGAN model not initialized in ModelManager.")

    # 1. Split RGB and Alpha channels
    rgb_image = input_image.convert("RGB")
    alpha_channel = input_image.split()[3]

    # Convert PIL images to NumPy arrays for processing
    rgb_np = np.array(rgb_image)
    alpha_np = np.array(alpha_channel)

    # --- THIS IS THE FIX ---
    # Convert the RGB NumPy array to BGR before passing it to the upsampler.
    rgb_np_bgr = rgb_np[:, :, ::-1]
    # -----------------------

    # 2. Upscale RGB channels
    log.info("Real-ESRGAN: Upscaling RGB channels...")
    # The enhance method returns a NumPy array (H, W, C) with BGR channel order
    # Pass the corrected BGR array to the function.
    output_rgb_bgr, _ = upsampler.enhance(rgb_np_bgr, outscale=4)

    # Convert BGR back to RGB for PIL (this part was already correct)
    output_rgb_np = output_rgb_bgr[:, :, ::-1]

    # 3. Upscale Alpha channel
    log.info("Real-ESRGAN: Upscaling alpha channel...")
    # To upscale the single-channel alpha, we duplicate it into a 3-channel image
    alpha_3_channel_np = np.stack([alpha_np, alpha_np, alpha_np], axis=-1)
    output_alpha_bgr, _ = upsampler.enhance(alpha_3_channel_np, outscale=4)

    # We only need one channel from the upscaled alpha
    output_alpha_np = output_alpha_bgr[:, :, 0]

    # 4. Create PIL Images from NumPy arrays
    upscaled_rgb_pil = Image.fromarray(output_rgb_np, "RGB")
    upscaled_alpha_pil = Image.fromarray(output_alpha_np, "L")

    # 5. Merge the upscaled RGB and Alpha channels back into a single RGBA image
    upscaled_rgba = Image.merge("RGBA", (*upscaled_rgb_pil.split(), upscaled_alpha_pil))

    log.info("Real-ESRGAN: Upscale complete", output_size=upscaled_rgba.size)

    return upscaled_rgba