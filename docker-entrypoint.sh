#!/bin/bash
set -e

echo "=========================================="
echo "Image Processing Service - Starting"
echo "=========================================="
echo ""

CACHE_DIR="/models" # Corrected Path

# --- NEW, MORE ROBUST CHECK ---
# Instead of checking for exact, long file paths which can change,
# we will check for the existence of the main model directories.
MODEL_DIR_1="$CACHE_DIR/models--briaai--RMBG-2.0"
MODEL_DIR_2="$CACHE_DIR/models--ZhengPeng7--BiRefNet_HR-matting"
MODEL_FILE_3="$CACHE_DIR/xinntao_Real-ESRGAN/RealESRGAN_x4plus.pth"

if [ -d "$MODEL_DIR_1" ] && [ -d "$MODEL_DIR_2" ] && [ -f "$MODEL_FILE_3" ]; then
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
    echo "❌ FATAL ERROR: One or more model directories were not found in the cache!"
    echo "   Checked for:"
    echo "   - Directory: $MODEL_DIR_1"
    echo "   - Directory: $MODEL_DIR_2"
    echo "   - File:      $MODEL_FILE_3"
    echo ""
    echo "   Please run the following command ONCE to download the models:"
    echo "   make download-models"
    echo ""
    exit 1
fi