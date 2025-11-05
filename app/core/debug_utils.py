# app/core/debug_utils.py

import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import torch

from app.config import settings

def save_debug_image(request_id: int, image_key: str, step_name: str, image_data):
    """
    Saves an image to a debug folder if DEBUG_SAVE_IMAGES is enabled.
    Handles PIL Images, NumPy arrays, and PyTorch Tensors.
    """
    if not settings.DEBUG_SAVE_IMAGES:
        return

    try:
        debug_dir = Path(settings.OUTPUT_DIR) / "debug" / str(request_id)
        debug_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{image_key}_{step_name}.png"
        filepath = debug_dir / filename

        pil_image = None
        if isinstance(image_data, Image.Image):
            pil_image = image_data
        elif isinstance(image_data, np.ndarray):
            # Convert BGR (from OpenCV) to RGB if needed
            if image_data.ndim == 3 and image_data.shape[2] == 3:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_data)
        elif isinstance(image_data, torch.Tensor):
            # Handle tensors (e.g., from model output)
            np_array = image_data.detach().cpu().float().numpy().squeeze()
            if np_array.ndim == 3: # H, W, C
                np_array = np_array.transpose(1, 2, 0)
            # Rescale from 0-1 range to 0-255
            np_array = (np_array * 255).astype(np.uint8)
            pil_image = Image.fromarray(np_array)

        if pil_image:
            pil_image.save(filepath, "PNG")

    except Exception as e:
        # Don't crash the main pipeline if a debug save fails
        print(f"Warning: Failed to save debug image {step_name}. Error: {e}")


def save_debug_heatmap(request_id: int, image_key: str, step_name: str, alpha_matte: np.ndarray):
    """
    Generates a color heatmap from a single-channel alpha matte and saves it.
    """
    if not settings.DEBUG_SAVE_IMAGES:
        return

    try:
        # Normalize to 0-255 and apply a color map (OpenCV is lightweight)
        alpha_uint8 = (alpha_matte * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(alpha_uint8, cv2.COLORMAP_JET)
        save_debug_image(request_id, image_key, f"{step_name}_heatmap", heatmap)
    except Exception as e:
        print(f"Warning: Failed to save debug heatmap {step_name}. Error: {e}")