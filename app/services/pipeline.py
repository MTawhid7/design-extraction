"""
Main processing pipeline with maximum parallelism.
Orchestrates the complete image processing workflow.
Updated for IS-Net integration (2025).
"""
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

log = structlog.get_logger(__name__)


class ImageProcessingPipeline:
    """Orchestrates parallel image processing pipeline."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.downloader = ImageDownloader()
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upscaler_semaphore = asyncio.Semaphore(settings.UPSCALER_CONCURRENCY_LIMIT)
        log.info("Pipeline initialized", upscaler_concurrency=settings.UPSCALER_CONCURRENCY_LIMIT)

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        """Execute the complete processing pipeline."""
        start_time = datetime.now()
        log.info("Starting pipeline", request_id=request.id)

        try:
            log.info("Stage 1: Downloading images...")
            front_img, back_img = await self._download_images(request)

            log.info("Stage 2: Extracting designs with Gemini...")
            front_extracted, back_extracted = await self._extract_designs(front_img, back_img)

            log.info("Stage 3: Removing backgrounds with IS-Net...")
            bg_removed_images = await self._remove_backgrounds(front_extracted, back_extracted)

            log.info("Stage 4: Upscaling images...")
            upscaled_images = await self._upscale_images(bg_removed_images)

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
        """Download front and back images in parallel."""
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

    async def _remove_backgrounds(self, front_extracted: Image.Image, back_extracted: Image.Image) -> Dict[str, Image.Image]:
        tasks = {
            # --- THIS IS THE FIX ---
            # We must call the 'run' function within the 'isnet_process' module.
            "front": asyncio.create_task(asyncio.to_thread(isnet_process.run, front_extracted, self.model_manager)),
            "back": asyncio.create_task(asyncio.to_thread(isnet_process.run, back_extracted, self.model_manager)),
        }
        results = await asyncio.gather(*tasks.values())
        bg_removed = {key: result for key, result in zip(tasks.keys(), results)}
        log.info("Background removal complete", outputs=list(bg_removed.keys()))
        return bg_removed

    async def _upscale_one_image(self, key: str, image: Image.Image) -> Tuple[str, Image.Image]:
        """A helper function to wrap the upscaling task with the semaphore."""
        async with self.upscaler_semaphore:
            log.info(f"Upscaler lock acquired for: {key}")
            upscaled_image = await asyncio.to_thread(upscaler_process.run, image, self.model_manager)
            log.info(f"Upscaler lock released for: {key}")
            return key, upscaled_image

    async def _upscale_images(self, bg_removed_images: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """Upscale images with concurrency limited by the semaphore."""
        tasks = [self._upscale_one_image(key, img) for key, img in bg_removed_images.items()]
        results = await asyncio.gather(*tasks)
        upscaled = {key: result_img for key, result_img in results}
        log.info("Upscaling complete", outputs=list(upscaled.keys()))
        return upscaled

    async def _save_images(self, request_id: int, images: Dict[str, Image.Image]) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        urls = {}
        save_tasks = []
        for key, img in images.items():
            filename = f"design_{request_id}_{timestamp}_{key}_isnet.png"
            filepath = self.output_dir / filename
            task = asyncio.create_task(asyncio.to_thread(img.save, filepath, "PNG"))
            save_tasks.append((key, filepath, task))

        await asyncio.gather(*[task for _, _, task in save_tasks])

        for key, filepath, _ in save_tasks:
            urls[key] = f"{settings.BASE_URL}/{filepath.name}"
        log.info("Images saved", count=len(urls))
        return urls