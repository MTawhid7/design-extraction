"""
Download all required models to the persistent storage volume.
This script provides explicit logging of download locations for verification.
"""
import os
import sys
import torch
import urllib.request
from huggingface_hub import login, snapshot_download
from transformers import AutoModelForImageSegmentation, AutoProcessor

# Define the explicit cache directory for all downloads.
CACHE_DIR = os.getenv("HF_HOME", "/models")

def download_models():
    """Download all models with proper error handling."""
    print("=" * 70)
    print("Model Download Script - Image Processing Service")
    print("=" * 70)
    print(f"--> All models will be saved to: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable not set.")
        return 1

    print(f"--> Found HF_TOKEN, authenticating...")
    login(token=hf_token, add_to_git_credential=False)
    print("    ✓ Authentication successful.")

    # --- Step 1: Download Gated Model (RMBG 2.0) ---
    try:
        print("\n[1/3] Downloading RMBG 2.0 (Gated Model) snapshot...")
        repo_id = "briaai/RMBG-2.0"
        revision = "a6a8895f89cf3150d2046e004766d2b93712c337" # Pin revision for consistency

        # Download complete snapshot first to ensure all files are in cache
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=CACHE_DIR,
            token=hf_token,
            # Allow patterns to ensure custom code and config files are fetched
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"]
        )

        # Then load with from_pretrained to populate transformers' internal cache mapping if needed
        AutoModelForImageSegmentation.from_pretrained(
            repo_id, trust_remote_code=True, token=hf_token, cache_dir=CACHE_DIR, revision=revision
        )
        AutoProcessor.from_pretrained(
            repo_id, trust_remote_code=True, token=hf_token, cache_dir=CACHE_DIR, revision=revision
        )
        print("      ✓ RMBG 2.0 snapshot and model loaded successfully.")
        print(f"        (Verified in cache directory: {os.path.join(CACHE_DIR, 'models--briaai--RMBG-2.0')})")
    except Exception as e:
        print(f"      ERROR: Failed to download RMBG 2.0. Have you accepted the license?")
        print(f"      Details: {e}")
        return 1

    # --- Step 2: Download Public Model (BiRefNet HR) ---
    try:
        print("\n[2/3] Downloading BiRefNet HR model snapshot...")
        repo_id = "ZhengPeng7/BiRefNet_HR-matting"
        # --- THIS IS A MORE STABLE REVISION HASH FOR THIS MODEL ---
        revision = "4548a3861993fb5a6f174dd2b5b52b9dbc226769"

        # Download complete snapshot first to ensure all files are in cache
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=CACHE_DIR,
            # Allow patterns to ensure custom code and config files are fetched
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"]
        )

        # Pre-load to populate transformers' internal cache mapping
        AutoModelForImageSegmentation.from_pretrained(
            repo_id, trust_remote_code=True, cache_dir=CACHE_DIR, revision=revision
        )
        print("      ✓ BiRefNet-HR snapshot and model loaded successfully.")
        print(f"        (Verified in cache directory: {os.path.join(CACHE_DIR, 'models--ZhengPeng7--BiRefNet_HR-matting')})")
    except Exception as e:
        print(f"      ERROR: Failed to download BiRefNet-HR: {e}")
        return 1

    # --- Step 3: Download Real-ESRGAN Model from Official GitHub Release ---
    try:
        print("\n[3/3] Downloading Real-ESRGAN model weights from GitHub...")

        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_dir = os.path.join(CACHE_DIR, "xinntao_Real-ESRGAN")
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")

        if not os.path.exists(model_save_path):
            print(f"      Downloading from {model_url}...")
            urllib.request.urlretrieve(model_url, model_save_path)
            # --- EXPLICIT LOGGING OF THE SAVE PATH ---
            print(f"      ✓ Real-ESRGAN downloaded and saved to: {model_save_path}")
        else:
            print(f"      ✓ Real-ESRGAN already exists at: {model_save_path}")

    except Exception as e:
        print(f"      ERROR: Failed to download Real-ESRGAN: {e}")
        return 1

    print("\n" + "=" * 70)
    print("All models downloaded and cached successfully!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(download_models())