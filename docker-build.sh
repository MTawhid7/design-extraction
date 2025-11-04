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

# Check if models exist
echo ""
print_step "Checking model cache..."

if [ -d "models" ] && [ "$(ls -A models 2>/dev/null)" ]; then
    CACHE_SIZE=$(du -sh models 2>/dev/null | cut -f1 || echo "unknown")
    print_info "✓ Models found in ./models (size: $CACHE_SIZE)"
    MODELS_EXIST=true
else
    print_warn "Models not found in ./models"
    MODELS_EXIST=false
fi

# Download models if needed
if [ "$MODELS_EXIST" = false ]; then
    echo ""

    if [ "$DOWNLOAD_MODELS" == "yes" ]; then
        DOWNLOAD_NOW=true
    elif [ "$DOWNLOAD_MODELS" == "no" ]; then
        DOWNLOAD_NOW=false
    else
        # Ask user
        read -p "$(echo -e ${YELLOW}[?]${NC}) Download models now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DOWNLOAD_NOW=true
        else
            DOWNLOAD_NOW=false
        fi
    fi

    if [ "$DOWNLOAD_NOW" = true ]; then
        print_step "Downloading models... This will take 5-10 minutes."
        print_info "Models will be saved to ./models/ and reused forever"
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
            print_info "✓ Models downloaded successfully"
            CACHE_SIZE=$(du -sh models/.cache 2>/dev/null | cut -f1 || echo "unknown")
            print_info "  Cache size: $CACHE_SIZE"
        else
            print_error "✗ Model download failed"
            exit 1
        fi
    else
        print_warn "Skipping model download"
        print_info "Models will be downloaded on first container start (slower)"
        print_info "Or download manually later with:"
        echo ""
        echo "  docker run --rm --gpus all \\"
        echo "    -v \$(pwd)/models:/models \\"
        echo "    image-processor:latest \\"
        echo "    python3 /app/download_models.py"
        echo ""
    fi
fi

# Final summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""

if [ "$MODELS_EXIST" = true ] || [ "$DOWNLOAD_NOW" = true ]; then
    print_info "System ready! Next steps:"
    echo ""
    echo "1. Ensure GEMINI_API_KEY is set in .env"
    echo ""
    echo "2. Start the service:"
    echo "   docker-compose up -d"
    echo ""
    echo "3. Check logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "4. Test the service:"
    echo "   curl http://localhost:8008/health"
    echo "   python test_client.py"
    echo ""
    print_info "Models are cached in ./models/ and will load instantly!"
else
    print_warn "Models not downloaded yet. Service will work but slower on first start."
    echo ""
    echo "Recommended workflow:"
    echo ""
    echo "1. Download models first (one time, ~5-10 min):"
    echo "   docker run --rm --gpus all \\"
    echo "     -v \$(pwd)/models:/models \\"
    echo "     image-processor:latest \\"
    echo "     python3 /app/download_models.py"
    echo ""
    echo "2. Then start service (instant startup):"
    echo "   docker-compose up -d"
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

print_info "Quick commands:"
echo "  Build:    bash docker-build.sh [production|dev] [default|no-cache]"
echo "  Start:    docker-compose up -d"
echo "  Stop:     docker-compose down"
echo "  Logs:     docker-compose logs -f"
echo "  Shell:    docker exec -it image-processor bash"
echo ""