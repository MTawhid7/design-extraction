"""
Async image downloader with connection pooling, retry logic, and exponential backoff.
"""
import httpx
import structlog
import asyncio
import random
from PIL import Image
from io import BytesIO
from typing import Optional

log = structlog.get_logger(__name__)


# --- IMPROVEMENT 3: Create a single, shared client instance ---
# This client will be created once and reused for all downloads, preserving the connection pool.
_client = httpx.AsyncClient(
    timeout=30.0,
    follow_redirects=True,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
    # --- IMPROVEMENT 2: Add a User-Agent header ---
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
)


class ImageDownloader:
    """Async image downloader using a shared, application-wide client."""

    def __init__(self, max_retries: int = 3, initial_backoff: float = 1.0):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        # The client is now managed globally, not per instance.

    async def download(self, url: str) -> Image.Image:
        """
        Download an image from a URL with exponential backoff retry logic.

        Args:
            url: URL of the image to download

        Returns:
            PIL Image object

        Raises:
            ValueError: If download fails or image is invalid
        """
        log.info("Downloading image", url=url)

        for attempt in range(self.max_retries):
            try:
                response = await _client.get(url)
                response.raise_for_status()

                image_data = BytesIO(response.content)
                image = Image.open(image_data)
                image.load()

                log.info("Image downloaded successfully",
                        url=url,
                        size=image.size,
                        mode=image.mode,
                        attempt=attempt + 1)
                return image

            except httpx.RequestError as e:
                log.warning("Network error downloading image",
                         url=url,
                         error=str(e),
                         attempt=attempt + 1)
            except httpx.HTTPStatusError as e:
                log.warning("HTTP error downloading image",
                         url=url,
                         status_code=e.response.status_code,
                         attempt=attempt + 1)
            except Exception as e:
                log.error("Unexpected error downloading image",
                         url=url,
                         error=str(e),
                         attempt=attempt + 1)

            if attempt < self.max_retries - 1:
                # --- IMPROVEMENT 1: Exponential Backoff with Jitter ---
                backoff_time = self.initial_backoff * (2 ** attempt)
                jitter = backoff_time * random.uniform(0.1, 0.5)
                wait_time = backoff_time + jitter
                log.info(f"Download failed. Retrying in {wait_time:.2f} seconds...", url=url)
                await asyncio.sleep(wait_time)

        log.error("Failed to download image after all retries", url=url, attempts=self.max_retries)
        raise ValueError(f"Failed to download image after {self.max_retries} attempts: {url}")

    async def close(self):
        """Close the shared HTTP client (optional, can be done on app shutdown)."""
        if not _client.is_closed:
            await _client.aclose()
            log.info("Shared HTTP client closed")