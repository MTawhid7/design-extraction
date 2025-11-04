# Optimized Dockerfile for Image Processing Service
# Base: CUDA 12.4 + cuDNN (via the devel image)

# =============================================================================
# Stage 1: Base image
# =============================================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies, including git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3.10 python3-pip python3.10-venv libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base AS dependencies

COPY requirements.txt .
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM dependencies AS production

ENV HF_HOME=/models \
    HF_HUB_OFFLINE=1

COPY app /app/app
COPY main.py /app/
COPY download_models.py /app/

# --- THIS IS THE CRITICAL FIX ---
# DO NOT create the /models directory here. It will be provided by the volume mount.
RUN mkdir -p /app/outputs /app/logs && chmod 777 /app/outputs /app/logs

ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8001/health', timeout=5).raise_for_status()" || exit 1

COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
# Start uvicorn directly on the correct port, respecting the PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]