#!/bin/bash
set -e

echo "=========================================="
echo "Image Processing Service - Starting"
echo "=========================================="
echo ""

CACHE_DIR="/models"

# --- MODIFIED: More robust check for the new model set ---
# We now check for the IS-Net weights and the Real-ESRGAN model file.
MODEL_FILE_ISNET="$CACHE_DIR/isnet-general-use.pth"
MODEL_FILE_ESRGAN="$CACHE_DIR/xinntao_Real-ESRGAN/RealESRGAN_x4plus.pth"

if [ -f "$MODEL_FILE_ISNET" ] && [ -f "$MODEL_FILE_ESRGAN" ]; then
    CACHE_SIZE=$(du -sh $CACHE_DIR | cut -f1)
    echo "✅ Models found in mounted volume (size: ${CACHE_SIZE})"
    export HF_HUB_OFFLINE=1
    echo "   Offline mode: ENABLED"
    echo ""
    echo "Starting FastAPI service..."
    echo "See main.py lifespan event for model loading logs."
    echo "=========================================="
    echo ""
    # Execute the main command (e.g., python main.py)
    exec "$@"
else
    echo "❌ FATAL ERROR: One or more model files were not found in the cache!"
    echo "   Please ensure the following files exist in your './models' directory:"

    if [ ! -f "$MODEL_FILE_ISNET" ]; then
        echo "   - Missing: isnet_dis_weights.pth (must be added manually)"
    fi
    if [ ! -f "$MODEL_FILE_ESRGAN" ]; then
        echo "   - Missing: RealESRGAN_x4plus.pth (download with 'make download-models')"
    fi

    echo ""
    echo "   To fix this:"
    echo "   1. Manually place 'isnet_dis_weights.pth' into the './models' folder."
    echo "   2. Run 'make download-models' to fetch the Real-ESRGAN model."
    echo ""
    exit 1
fi