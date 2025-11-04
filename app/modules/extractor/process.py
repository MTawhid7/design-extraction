"""
Async Gemini extractor optimized for parallel processing.
"""
import structlog
from PIL import Image
from io import BytesIO
from app.core.model_manager import ModelManager
from .config import EXTRACTION_PROMPT

log = structlog.get_logger(__name__)


async def run(input_image: Image.Image, model_manager: ModelManager) -> Image.Image:
    """
    Extracts the main design from an image using the Gemini Vision API.

    Args:
        input_image: PIL Image object (already downloaded)
        model_manager: The manager holding the configured Gemini model

    Returns:
        Extracted design as PIL Image
    """
    log.info("Gemini: Starting extraction", input_size=input_image.size)

    # --- Use the fully configured GenerativeModel from the manager ---
    gemini_model = model_manager.gemini_model
    if not gemini_model:
        raise RuntimeError("Gemini model is not initialized in ModelManager.")

    contents = [EXTRACTION_PROMPT, input_image]

    try:
        log.info("Gemini: Sending request to API...")

        # --- THIS IS THE CORRECTED ASYNC API CALL ---
        # The high-level model object handles passing the configuration correctly.
        response = await gemini_model.generate_content_async(contents=contents)

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    log.info("Gemini: Received image response",
                            mime_type=part.inline_data.mime_type)

                    image_bytes = part.inline_data.data
                    extracted_image = Image.open(BytesIO(image_bytes))

                    log.info("Gemini: Extraction successful",
                            output_size=extracted_image.size,
                            output_mode=extracted_image.mode)

                    # Ensure RGB mode for consistency
                    if extracted_image.mode == 'RGBA':
                        background = Image.new('RGB', extracted_image.size, (255, 255, 255))
                        background.paste(extracted_image, mask=extracted_image.split()[3])
                        return background
                    elif extracted_image.mode != 'RGB':
                        return extracted_image.convert('RGB')

                    return extracted_image

        log.error("Gemini: No valid image in response")
        raise RuntimeError("Failed to extract design: Gemini API did not return an image.")

    except Exception as e:
        log.exception("Gemini: API call failed")
        raise RuntimeError(f"Gemini API call failed: {e}") from e