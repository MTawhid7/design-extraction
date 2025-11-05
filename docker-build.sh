#!/bin/bash

# Docker build and setup script for Image Processing Service
# Optimized for persistent model storage and instant startup

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

echo "=========================================="
echo "Image Processing Service - Setup"
echo "=========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

# Check NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    print_warn "NVIDIA Docker runtime not detected. GPU support may not work."
    print_warn "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi

# Parse arguments
BUILD_MODE=${1:-production}
CACHE_MODE=${2:-default}
DOWNLOAD_MODELS=${3:-ask}

# Check .env
if [ ! -f ".env" ]; then
    print_warn ".env file not found. Creating template..."
    cat > .env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key_here
BASE_URL=https://yourcdn.com
LOG_LEVEL=INFO
EOF
    print_warn "Please edit .env and add your GEMINI_API_KEY"
fi

# Create required directories
print_step "Creating required directories..."
mkdir -p outputs logs models

print_info "Directory structure:"
echo "  ./outputs  - Processed images"
echo "  ./logs     - Application logs"
echo "  ./models   - Persistent model cache (critical!)"
echo ""

# Build Docker image
print_step "Building Docker image..."

BUILD_ARGS=""
if [ "$CACHE_MODE" == "no-cache" ]; then
    BUILD_ARGS="--no-cache"
    print_info "Building with --no-cache"
elif [ "$CACHE_MODE" == "force" ]; then
    BUILD_ARGS="--pull --no-cache"
    print_info "Building with --pull --no-cache"
fi

docker build $BUILD_ARGS -t image-processor:latest .

if [ $? -eq 0 ]; then
    print_info "✓ Docker image built successfully"
else
    print_error "✗ Docker build failed"
    exit 1
fi

# Check image size
IMAGE_SIZE=$(docker images image-processor:latest --format "{{.Size}}" | head -1)
print_info "Image size: $IMAGE_SIZE"

# --- CORRECTED MODEL CHECK LOGIC ---
echo ""
print_step "Checking model cache for required files..."

# Define the paths to the essential model files
MODEL_FILE_ISNET="./models/isnet-general-use.pth"
MODEL_FILE_ESRGAN="./models/xinntao_Real-ESRGAN/RealESRGAN_x4plus.pth"

# Check if ALL required models exist on the host
if [ -f "$MODEL_FILE_ISNET" ] && [ -f "$MODEL_FILE_ESRGAN" ]; then
    CACHE_SIZE=$(du -sh models 2>/dev/null | cut -f1 || echo "unknown")
    print_info "✓ All required models found in ./models (size: $CACHE_SIZE)"
    MODELS_EXIST=true
else
    print_warn "One or more required models are missing from ./models."
    # List what is missing for the user
    if [ ! -f "$MODEL_FILE_ISNET" ]; then
        print_warn "  - MISSING: isnet-general-use.pth (You must add this manually)"
    fi
    if [ ! -f "$MODEL_FILE_ESRGAN" ]; then
        print_warn "  - MISSING: RealESRGAN_x4plus.pth (Can be downloaded now)"
    fi
    MODELS_EXIST=false
fi

# Download models if the check above failed
if [ "$MODELS_EXIST" = false ]; then
    echo ""

    # Only ask to download if the downloadable model is the one missing
    if [ ! -f "$MODEL_FILE_ESRGAN" ]; then
        if [ "$DOWNLOAD_MODELS" == "yes" ]; then
            DOWNLOAD_NOW=true
        elif [ "$DOWNLOAD_MODELS" == "no" ]; then
            DOWNLOAD_NOW=false
        else
            # Ask user
            read -p "$(echo -e ${YELLOW}[?]${NC}) Download missing public models now (Real-ESRGAN)? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                DOWNLOAD_NOW=true
            else
                DOWNLOAD_NOW=false
            fi
        fi

        if [ "$DOWNLOAD_NOW" = true ]; then
            print_step "Downloading models... This will take a few minutes."
            print_info "Models will be saved to ./models/ and reused."
            echo ""

            docker run --rm \
                --gpus all \
                --env-file .env \
                -e HF_HUB_OFFLINE=0 \
                -v $(pwd)/models:/models \
                --entrypoint="" \
                image-processor:latest \
                python3 /app/download_models.py

            if [ $? -eq 0 ]; then
                print_info "✓ Public models downloaded successfully"
            else
                print_error "✗ Model download failed"
                exit 1
            fi
        else
            print_warn "Skipping model download."
        fi
    fi
fi

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

# Re-check to give the user final status and next steps
if [ -f "$MODEL_FILE_ISNET" ] && [ -f "$MODEL_FILE_ESRGAN" ]; then
    print_info "System ready! Next steps:"
    echo ""
    echo "1. Ensure GEMINI_API_KEY is set in .env"
    echo "2. Start the service: make up"
    echo "3. Check logs: make logs"
    echo "4. Test the service: make test"
    echo ""
    print_info "Models are cached in ./models/ and will load instantly!"
else
    print_error "Setup is incomplete. Required models are still missing."
    print_error "Please check the ./models directory and re-run this script."
    echo ""
fi

# Show model caching info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Model Caching Strategy:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_info "Models are stored in ./models/ (NOT in Docker image)"
print_info "Benefits:"
echo "  • Download once, use forever"
echo "  • Container rebuilds don't lose models"
echo "  • Fast startup (models already downloaded)"
echo "  • Easy backup (just backup ./models/ folder)"
echo ""