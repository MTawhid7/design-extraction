# app/services/pipeline.py
import asyncio
from typing import Dict, Tuple
from PIL import Image
import structlog
from pathlib import Path
from datetime import datetime

from app.core.model_manager import ModelManager
from app.models.request import ProcessRequest
from app.models.response import ProcessResponse
from app.services.downloader import ImageDownloader
from app.modules.extractor import process as gemini_process
from app.modules.remover.isnet import process as isnet_process
from app.modules.upscaler import process as upscaler_process
from app.config import settings
from app.core.debug_utils import save_debug_image
from app.core.process_lock import isnet_lock, realesrgan_lock

log = structlog.get_logger(__name__)

# --- SOLUTION PART 1: Create synchronous helper functions ---

def _process_backgrounds_safely(request_id: int, front_img: Image.Image, back_img: Image.Image, model_manager: ModelManager) -> Tuple[Image.Image, Image.Image]:
    """Synchronous helper to run IS-Net with a cross-process lock."""
    with isnet_lock:
        log.info("IS-Net lock acquired", request_id=request_id)
        # Process images sequentially within the lock to prevent race conditions
        front_removed = isnet_process.run(front_img, model_manager)
        back_removed = isnet_process.run(back_img, model_manager)
        log.info("IS-Net lock released", request_id=request_id)
    return front_removed, back_removed

def _process_upscaling_safely(request_id: int, bg_removed_images: Dict[str, Image.Image], model_manager: ModelManager) -> Dict[str, Image.Image]:
    """Synchronous helper to run Real-ESRGAN with a cross-process lock."""
    with realesrgan_lock:
        log.info("Real-ESRGAN lock acquired", request_id=request_id)
        upscaled = {}
        # Process images sequentially within the lock
        for key, img in bg_removed_images.items():
            upscaled[key] = upscaler_process.run(img, model_manager)
        log.info("Real-ESRGAN lock released", request_id=request_id)
    return upscaled


class ImageProcessingPipeline:
    """Orchestrates parallel image processing pipeline."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.downloader = ImageDownloader()
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("Pipeline initialized")

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        # This method's logic remains the same
        start_time = datetime.now()
        log.info("Starting pipeline", request_id=request.id)
        try:
            log.info("Stage 1: Downloading images...")
            front_img, back_img = await self._download_images(request)
            save_debug_image(request.id, "front", "0_original", front_img)
            save_debug_image(request.id, "back", "0_original", back_img)

            log.info("Stage 2: Extracting designs with Gemini...")
            front_extracted, back_extracted = await self._extract_designs(front_img, back_img)
            save_debug_image(request.id, "front", "1_gemini_extracted", front_extracted)
            save_debug_image(request.id, "back", "1_gemini_extracted", back_extracted)

            log.info("Stage 3: Removing backgrounds with IS-Net...")
            bg_removed_images = await self._remove_backgrounds(request.id, front_extracted, back_extracted)

            log.info("Stage 4: Upscaling images...")
            upscaled_images = await self._upscale_images(request.id, bg_removed_images)

            log.info("Stage 5: Saving images...")
            urls = await self._save_images(request.id, upscaled_images)

            elapsed = (datetime.now() - start_time).total_seconds()
            log.info("Pipeline complete", request_id=request.id, elapsed_seconds=elapsed)

            return ProcessResponse(
                id=request.id,
                front_output=urls["front"],
                back_output=urls["back"],
                processing_time_seconds=elapsed
            )
        except Exception as e:
            log.exception("Pipeline failed", request_id=request.id)
            raise

    async def _download_images(self, request: ProcessRequest) -> Tuple[Image.Image, Image.Image]:
        front_task = self.downloader.download(request.output.front)
        back_task = self.downloader.download(request.output.back)
        front_img, back_img = await asyncio.gather(front_task, back_task)
        log.info("Images downloaded", front_size=front_img.size, back_size=back_img.size)
        return front_img, back_img

    async def _extract_designs(self, front_img: Image.Image, back_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        front_task = gemini_process.run(front_img, self.model_manager)
        back_task = gemini_process.run(back_img, self.model_manager)
        front_extracted, back_extracted = await asyncio.gather(front_task, back_task)
        log.info("Designs extracted", front_size=front_extracted.size, back_size=back_extracted.size)
        return front_extracted, back_extracted

    # --- SOLUTION PART 2: Modify the async methods to call the helpers ---
    async def _remove_backgrounds(self, request_id: int, front_extracted: Image.Image, back_extracted: Image.Image) -> Dict[str, Image.Image]:
        """Runs the entire background removal process in a worker thread to handle locking."""
        front_removed, back_removed = await asyncio.to_thread(
            _process_backgrounds_safely, request_id, front_extracted, back_extracted, self.model_manager
        )

        save_debug_image(request_id, "front", "2_isnet_removed", front_removed)
        save_debug_image(request_id, "back", "2_isnet_removed", back_removed)

        bg_removed = {"front": front_removed, "back": back_removed}
        log.info("Background removal complete", outputs=list(bg_removed.keys()))
        return bg_removed

    async def _upscale_images(self, request_id: int, bg_removed_images: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """Runs the entire upscaling process in a worker thread to handle locking."""
        upscaled = await asyncio.to_thread(
            _process_upscaling_safely, request_id, bg_removed_images, self.model_manager
        )

        save_debug_image(request_id, "front", "3_realesrgan_upscaled", upscaled["front"])
        save_debug_image(request_id, "back", "3_realesrgan_upscaled", upscaled["back"])

        log.info("Upscaling complete", outputs=list(upscaled.keys()))
        return upscaled

    async def _save_images(self, request_id: int, images: Dict[str, Image.Image]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        urls = {}
        save_tasks = []
        for key, img in images.items():
            filename = f"design_{request_id}_{timestamp}_{key}_isnet.png"
            filepath = self.output_dir / filename

            # --- THIS IS THE FIX ---
            # Defensively ensure the parent directory exists right before saving.
            # This prevents a crash if the 'outputs' folder is deleted during runtime.
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # -----------------------

            task = asyncio.create_task(asyncio.to_thread(img.save, filepath, "PNG"))
            save_tasks.append((key, filepath, task))

        await asyncio.gather(*[task for _, _, task in save_tasks])

        for key, filepath, _ in save_tasks:
            urls[key] = f"{settings.BASE_URL}/{filepath.name}"
        log.info("Images saved", count=len(urls))
        return urls