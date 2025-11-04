"""
Async Gemini extractor optimized for parallel processing.
Updated for google-genai SDK and Gemini 2.5 Flash Image (2025).
"""
import structlog
from PIL import Image
from io import BytesIO
from app.core.model_manager import ModelManager
from .config import EXTRACTION_PROMPT, GENERATION_CONFIG, SAFETY_SETTINGS

log = structlog.get_logger(__name__)


async def run(input_image: Image.Image, model_manager: ModelManager) -> Image.Image:
    """
    Extracts the main design from an image using the Gemini 2.5 Flash Image API.

    Args:
        input_image: PIL Image object (already downloaded)
        model_manager: The manager holding the Gemini client

    Returns:
        Extracted design as PIL Image
    """
    log.info("Gemini: Starting extraction", input_size=input_image.size)

    # Get the client from model manager
    client = model_manager.gemini_client
    if not client:
        raise RuntimeError("Gemini client is not initialized in ModelManager.")

    # Prepare the contents (prompt + image)
    contents = [EXTRACTION_PROMPT, input_image]

    try:
        log.info("Gemini: Sending request to API...")

        # Use the async client for proper async operation
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash-image",  # Use the image generation model
            contents=contents,
            config=GENERATION_CONFIG,
        )

        # Process the response
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                # Check for inline image data
                if part.inline_data is not None:
                    log.info("Gemini: Received image response",
                            mime_type=part.inline_data.mime_type)

                    # Extract image bytes
                    image_bytes = part.inline_data.data
                    extracted_image = Image.open(BytesIO(image_bytes))

                    # --- THIS IS THE CRITICAL FIX ---
                    # Force Pillow to load the image data from the in-memory stream.
                    # This prevents the "broken data stream" error in downstream processes.
                    extracted_image.load()

                    log.info("Gemini: Extraction successful",
                            output_size=extracted_image.size,
                            output_mode=extracted_image.mode)

                    # Ensure RGB mode for consistency
                    if extracted_image.mode == 'RGBA':
                        # Create white background for RGBA images
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