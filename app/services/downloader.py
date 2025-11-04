"""
Async image downloader with connection pooling and retry logic.
"""
import httpx
import structlog
from PIL import Image
from io import BytesIO
from typing import Optional

log = structlog.get_logger(__name__)


class ImageDownloader:
    """Async image downloader with connection pooling."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=10,
                    max_connections=20
                )
            )
        return self._client

    async def download(self, url: str) -> Image.Image:
        """
        Download an image from a URL with retry logic.

        Args:
            url: URL of the image to download

        Returns:
            PIL Image object

        Raises:
            ValueError: If download fails or image is invalid
        """
        log.info("Downloading image", url=url)

        client = await self._get_client()

        for attempt in range(self.max_retries):
            try:
                response = await client.get(url)
                response.raise_for_status()

                # Parse image
                image_data = BytesIO(response.content)
                image = Image.open(image_data)

                # Ensure image is loaded
                image.load()

                log.info("Image downloaded successfully",
                        url=url,
                        size=image.size,
                        mode=image.mode,
                        attempt=attempt + 1)

                return image

            except httpx.HTTPStatusError as e:
                log.error("HTTP error downloading image",
                         url=url,
                         status_code=e.response.status_code,
                         attempt=attempt + 1)

                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to download image after {self.max_retries} attempts: {url}") from e

            except Exception as e:
                log.error("Error downloading image",
                         url=url,
                         error=str(e),
                         attempt=attempt + 1)

                if attempt == self.max_retries - 1:
                    raise ValueError(f"Failed to download image: {url}") from e

        raise ValueError(f"Failed to download image after {self.max_retries} attempts: {url}")

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            log.info("HTTP client closed")