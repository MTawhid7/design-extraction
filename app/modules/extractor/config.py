"""
Configuration for the Gemini Extractor module.
Updated for google-genai SDK and Gemini 2.5 Flash Image (2025).
"""
from google.genai import types

# Extraction prompt for Gemini 2.5 Flash Image
EXTRACTION_PROMPT = """
**Primary Task:** Isolate and extract the graphical design from the provided t-shirt image. The final output must be a clean, high-resolution image of the design itself, completely free of any influence from the original t-shirt.

**Critical Exclusion Criteria (What to Avoid):**
*   **NO T-SHIRT CONTEXT:** Do not include any part of the t-shirt fabric, texture, wrinkles, seams, or collar.
*   **NO GHOSTING:** There should be absolutely no residual shape or outline of the t-shirt.
*   **NO ENVIRONMENTAL ELEMENTS:** Eliminate all background, shadows, lighting effects, or any other element from the original photo.

**Detailed Extraction and Output Requirements:**
1.  **Isolate the Design:** Precisely identify and select only the printed or embroidered graphic.
2.  **Recreate as a Flat Graphic:** The output should appear as a perfectly flat, two-dimensional representation of the design, as if it were a digital vector graphic.
3.  **Preserve Integrity:** Maintain the original colors, proportions, and details of the design with high fidelity.
4.  **Clean Edges:** The edges of the design must be sharp and well-defined.
5.  **Output Format:** Generate a high-resolution PNG with a transparent background.

**Final Check:** Before outputting, verify that the image contains ONLY the design and nothing else.
"""

# Generation configuration for Gemini 2.5 Flash Image API
GENERATION_CONFIG = types.GenerateContentConfig(
    response_modalities=["IMAGE"],  # Request image output
    temperature=0.4,
    # Optional: Configure image output settings
    # image_config=types.ImageConfig(
    #     aspect_ratio="1:1",  # Options: "1:1", "16:9", "9:16", "4:3", "3:4", etc.
    # )
)

# Safety settings for Gemini API
SAFETY_SETTINGS = [
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE"
    ),
]