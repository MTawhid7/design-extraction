"""
FastAPI service for parallel image processing with Gemini, RMBG, BiRefNet, and Real-ESRGAN.
Optimized for L4 GPU with 24GB VRAM.
"""
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog

from app.models.request import ProcessRequest
from app.models.response import ProcessResponse
from app.services.pipeline import ImageProcessingPipeline
from app.core.model_manager import ModelManager
from app.config import settings

log = structlog.get_logger(__name__)

# Global model manager instance
model_manager: ModelManager = None
pipeline: ImageProcessingPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading models on startup and cleanup on shutdown."""
    global model_manager, pipeline

    log.info("Starting service initialization...")

    # Initialize model manager and load all models
    model_manager = ModelManager()
    await model_manager.initialize()

    # Initialize pipeline
    pipeline = ImageProcessingPipeline(model_manager)

    log.info("Service initialization complete. Ready to process requests.")

    yield

    # Cleanup
    log.info("Shutting down service...")
    if model_manager:
        await model_manager.cleanup()
    log.info("Service shutdown complete.")


app = FastAPI(
    title="Parallel Image Processing Service",
    description="High-performance image processing with Gemini, RMBG, BiRefNet, and Real-ESRGAN",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_initialized if model_manager else False
    }


@app.post("/process", response_model=ProcessResponse)
async def process_design(request: ProcessRequest):
    """
    Process front and back images through the complete pipeline.

    Pipeline stages:
    1. Download images (parallel)
    2. Gemini extraction (parallel)
    3. Background removal with RMBG and BiRefNet (parallel)
    4. Upscaling with Real-ESRGAN (parallel for all outputs)
    5. Save and return URLs
    """
    try:
        log.info("Received process request", request_id=request.id)

        if not pipeline:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Run the complete pipeline
        result = await pipeline.process(request)

        log.info("Process request completed", request_id=request.id)
        return result

    except Exception as e:
        log.exception("Error processing request", request_id=request.id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Parallel Image Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process": "/process"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )