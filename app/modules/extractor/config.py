"""
Configuration for the Gemini Extractor module.
"""
from google import genai

# Extraction prompt for Gemini
EXTRACTION_PROMPT = """
Task:
Given an input image of a t-shirt, extract only the printed design from the shirt. Remove all fabric texture, folds, shadows, or background. Output a clean, sharp, high-resolution version of the design itself.

Detailed Instructions:
1. Detect and isolate the printed or embroidered graphic visible on the t-shirt.
2. Remove all non-design elements, including the t-shirt fabric, wrinkles, shadows, lighting gradients, and background.
3. Preserve accurate colors, edges, and proportions of the original design.
4. Output should be a transparent-background PNG.
5. Maintain maximum sharpness and resolution, suitable for reuse in print or digital design.
6. Do not include any part of the garment, model, or scene—only the design.

Style / Output Requirements:
- Output format: Design-only image (no background or fabric).
- Resolution: Highest available, lossless quality.
- Edge clarity: Perfectly clean, no blending with the shirt.
- Color fidelity: Match the original printed design as closely as possible.
"""

# Generation configuration for Gemini API
GENERATION_CONFIG = genai.types.GenerationConfig(
    response_modalities=["IMAGE"],
    temperature=0.4,
)

# Safety settings for Gemini API
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]