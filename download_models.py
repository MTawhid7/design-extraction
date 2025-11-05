"""
Download all required models to the persistent storage volume.
This script provides explicit logging of download locations for verification.
"""
import os
import sys
import torch
import urllib.request
from huggingface_hub import login, snapshot_download

# Define the explicit cache directory for all downloads.
CACHE_DIR = os.getenv("HF_HOME", "/models")

def download_models():
    """Download all models with proper error handling."""
    print("=" * 70)
    print("Model Download Script - Image Processing Service")
    print("=" * 70)
    print(f"--> All models will be saved to: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- NOTE: Hugging Face login is no longer required as we are not downloading gated models ---
    # --- but we keep the logic in case it's needed in the future. ---

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"--> Found HF_TOKEN, authenticating...")
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("    ✓ Authentication successful.")
        except Exception as e:
            print(f"    WARNING: Failed to login to Hugging Face, but this may not be an issue. Details: {e}")
    else:
        print("--> HF_TOKEN not set. Skipping Hugging Face login.")


    # --- Step 1: REMOVED - RMBG 2.0 Download ---

    # --- Step 2: REMOVED - BiRefNet HR Download ---

    # --- Now Step 1: Download Real-ESRGAN Model from Official GitHub Release ---
    try:
        print("\n[1/1] Downloading Real-ESRGAN model weights from GitHub...")

        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_dir = os.path.join(CACHE_DIR, "xinntao_Real-ESRGAN")
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")

        if not os.path.exists(model_save_path):
            print(f"      Downloading from {model_url}...")
            urllib.request.urlretrieve(model_url, model_save_path)
            print(f"      ✓ Real-ESRGAN downloaded and saved to: {model_save_path}")
        else:
            print(f"      ✓ Real-ESRGAN already exists at: {model_save_path}")

    except Exception as e:
        print(f"      ERROR: Failed to download Real-ESRGAN: {e}")
        return 1

    print("\n" + "=" * 70)
    print("All required public models downloaded successfully!")
    print("\nIMPORTANT: Please ensure you have manually added the 'isnet_dis_weights.pth' file")
    print(f"to the '{CACHE_DIR}' directory.")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(download_models())