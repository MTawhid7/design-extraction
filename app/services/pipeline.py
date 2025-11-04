"""
Main processing pipeline with maximum parallelism.
Orchestrates the complete image processing workflow.
"""
import asyncio
from typing import Dict, List, Tuple
from PIL import Image
import structlog
from pathlib import Path
import uuid
from datetime import datetime

from app.core.model_manager import ModelManager
from app.models.request import ProcessRequest
from app.models.response import ProcessResponse, ProcessedImage
from app.services.downloader import ImageDownloader
from app.modules.extractor import process as gemini_process
from app.modules.remover.rmbg import process as rmbg_process
from app.modules.remover.birefnet import process as birefnet_process
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

    async def process(self, request: ProcessRequest) -> ProcessResponse:
        """
        Execute the complete processing pipeline with maximum parallelism.

        Pipeline flow:
        1. Download front & back images (parallel)
        2. Gemini extraction on both (parallel)
        3. For each extracted image:
           - RMBG background removal
           - BiRefNet background removal
           (parallel)
        4. Upscale all 4 results (parallel)
        5. Save all images and return URLs
        """
        start_time = datetime.now()
        log.info("Starting pipeline", request_id=request.id)

        try:
            # Stage 1: Download images in parallel
            log.info("Stage 1: Downloading images...")
            front_img, back_img = await self._download_images(request)

            # Stage 2: Gemini extraction in parallel
            log.info("Stage 2: Extracting designs with Gemini...")
            front_extracted, back_extracted = await self._extract_designs(front_img, back_img)

            # Stage 3: Background removal (all combinations in parallel)
            log.info("Stage 3: Removing backgrounds...")
            bg_removed_images = await self._remove_backgrounds(front_extracted, back_extracted)

            # Stage 4: Upscale all images in parallel
            log.info("Stage 4: Upscaling images...")
            upscaled_images = await self._upscale_images(bg_removed_images)

            # Stage 5: Save all images and generate URLs
            log.info("Stage 5: Saving images...")
            urls = await self._save_images(request.id, upscaled_images)

            elapsed = (datetime.now() - start_time).total_seconds()
            log.info("Pipeline complete", request_id=request.id, elapsed_seconds=elapsed)

            return ProcessResponse(
                id=request.id,
                front_rmbg=urls["front_rmbg"],
                front_birefnet=urls["front_birefnet"],
                back_rmbg=urls["back_rmbg"],
                back_birefnet=urls["back_birefnet"],
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

        log.info("Images downloaded",
                 front_size=front_img.size,
                 back_size=back_img.size)

        return front_img, back_img

    async def _extract_designs(
        self,
        front_img: Image.Image,
        back_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Extract designs using Gemini in parallel."""

        # Run Gemini extraction in parallel for both images
        front_task = gemini_process.run(front_img, self.model_manager)
        back_task = gemini_process.run(back_img, self.model_manager)

        front_extracted, back_extracted = await asyncio.gather(front_task, back_task)

        log.info("Designs extracted",
                 front_size=front_extracted.size,
                 back_size=back_extracted.size)

        return front_extracted, back_extracted

    async def _remove_backgrounds(
        self,
        front_extracted: Image.Image,
        back_extracted: Image.Image
    ) -> Dict[str, Image.Image]:
        """
        Remove backgrounds using both RMBG and BiRefNet in parallel.
        Creates 4 outputs total (2 images × 2 models).
        """

        # Create all 4 background removal tasks in parallel
        tasks = {
            "front_rmbg": asyncio.create_task(
                asyncio.to_thread(rmbg_process.run, front_extracted, self.model_manager)
            ),
            "front_birefnet": asyncio.create_task(
                asyncio.to_thread(birefnet_process.run, front_extracted, self.model_manager)
            ),
            "back_rmbg": asyncio.create_task(
                asyncio.to_thread(rmbg_process.run, back_extracted, self.model_manager)
            ),
            "back_birefnet": asyncio.create_task(
                asyncio.to_thread(birefnet_process.run, back_extracted, self.model_manager)
            ),
        }

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks.values())

        # Map results back to keys
        bg_removed = {key: result for key, result in zip(tasks.keys(), results)}

        log.info("Background removal complete", outputs=list(bg_removed.keys()))

        return bg_removed

    async def _upscale_images(
        self,
        bg_removed_images: Dict[str, Image.Image]
    ) -> Dict[str, Image.Image]:
        """Upscale all 4 images in parallel."""

        # Create upscaling tasks for all images
        tasks = {
            key: asyncio.create_task(
                asyncio.to_thread(upscaler_process.run, img, self.model_manager)
            )
            for key, img in bg_removed_images.items()
        }

        # Wait for all upscaling to complete
        results = await asyncio.gather(*tasks.values())

        # Map results back to keys
        upscaled = {key: result for key, result in zip(tasks.keys(), results)}

        log.info("Upscaling complete", outputs=list(upscaled.keys()))

        return upscaled

    async def _save_images(
        self,
        request_id: int,
        images: Dict[str, Image.Image]
    ) -> Dict[str, str]:
        """Save all images and return their URLs."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        urls = {}

        # Save all images in parallel
        save_tasks = []
        for key, img in images.items():
            filename = f"design_{request_id}_{timestamp}_{key}.png"
            filepath = self.output_dir / filename

            # Create save task
            task = asyncio.create_task(
                asyncio.to_thread(img.save, filepath, "PNG")
            )
            save_tasks.append((key, filepath, task))

        # Wait for all saves to complete
        await asyncio.gather(*[task for _, _, task in save_tasks])

        # Generate URLs
        for key, filepath, _ in save_tasks:
            # Generate URL based on your storage configuration
            # This is a placeholder - adjust based on your actual storage setup
            urls[key] = f"{settings.BASE_URL}/{filepath.name}"

        log.info("Images saved", count=len(urls))

        return urls