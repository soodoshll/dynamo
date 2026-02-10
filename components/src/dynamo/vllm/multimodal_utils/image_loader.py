# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import binascii
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import TypeAlias, Union
from urllib.parse import urlparse

import httpx
import torch
from PIL import Image

from .http_client import get_http_client

logger = logging.getLogger(__name__)

# Image output can be either PIL Image or Tensor (from nvimgcodec)
ImageOutput: TypeAlias = Union[Image.Image, torch.Tensor]

# Thread-local storage for nvimgcodec decoders
_thread_local = threading.local()

# Lazy import for nvimgcodec
_nvimgcodec = None
_nvimgcodec_available: bool | None = None  # None = not yet probed

# Global thread pool for nvimgcodec decoding operations
# Default to 8 workers, configurable via DYN_IMAGE_DECODE_WORKERS env var
_IMAGE_DECODE_WORKERS = int(os.environ.get("DYN_IMAGE_DECODE_WORKERS", 8))
_decode_thread_pool = ThreadPoolExecutor(
    max_workers=_IMAGE_DECODE_WORKERS,
    thread_name_prefix="image_decode_",
)


def _is_nvimgcodec_available() -> bool:
    """Check whether nvimgcodec can be imported. Result is cached."""
    global _nvimgcodec_available
    if _nvimgcodec_available is None:
        try:
            _get_nvimgcodec()
            _nvimgcodec_available = True
        except (ImportError, ModuleNotFoundError):
            _nvimgcodec_available = False
    return _nvimgcodec_available


def _get_nvimgcodec():
    """Lazy import nvimgcodec. Raises ImportError if not installed."""
    global _nvimgcodec
    if _nvimgcodec is None:
        from nvidia import nvimgcodec

        _nvimgcodec = nvimgcodec
    return _nvimgcodec


def get_decoder():
    """Get or create a thread-local nvimgcodec decoder instance."""
    if not hasattr(_thread_local, "decoder"):
        nvimgcodec = _get_nvimgcodec()
        _thread_local.decoder = nvimgcodec.Decoder()
        logger.info("nvimgcodec decoder initialized for thread")
    return _thread_local.decoder


class ImageLoader:
    CACHE_SIZE_MAXIMUM = 8
    DEFAULT_MAX_PENDING = 64

    def __init__(
        self,
        cache_size: int = CACHE_SIZE_MAXIMUM,
        http_timeout: float = 30.0,
        use_nvimgcodec: bool = True,
        max_pending: int | None = None,
    ):
        self._http_timeout = http_timeout
        self._image_cache: dict[str, Image.Image] = {}
        self._cache_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=cache_size)

        # DYN_USE_NVIMGCODEC=0 to force PIL path
        env_override = os.environ.get("DYN_USE_NVIMGCODEC")
        if env_override is not None:
            use_nvimgcodec = env_override.lower() not in ("0", "false", "no")

        # Fall back to PIL if nvimgcodec was requested but is not installed
        if use_nvimgcodec and not _is_nvimgcodec_available():
            logger.warning(
                "nvimgcodec requested but not installed — "
                "falling back to PIL for image decoding"
            )
            use_nvimgcodec = False
        self._use_nvimgcodec = use_nvimgcodec

        if max_pending is None:
            max_pending = int(
                os.environ.get("DYN_IMAGE_MAX_PENDING", self.DEFAULT_MAX_PENDING)
            )
        self._pending_semaphore = asyncio.Semaphore(max_pending)
        self._max_pending = max_pending

    def mark_consumed(self, count: int = 1):
        """
        Signal that decoded images have been consumed by the vLLM prefill batch.
        Call this after the prefill batch completes to allow more images to be decoded.

        Args:
            count: Number of images consumed (default: 1)
        """
        for _ in range(count):
            self._pending_semaphore.release()

    def _decode_with_nvimgcodec(self, data: bytes) -> torch.Tensor:
        """
        Decode image bytes using nvimgcodec for GPU-accelerated decoding.

        Returns:
            torch.Tensor in NCHW format (4D) on CUDA device.
            Shape: (1, C, H, W) - batch dimension added so vLLM treats it as
            a batch of images, not as embeddings.
        """
        nvimgcodec = _get_nvimgcodec()
        decoder = get_decoder()
        code_stream = nvimgcodec.CodeStream(data)
        decoded = decoder.decode(code_stream)

        device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.as_tensor(decoded, device=device)
        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)
        # Add batch dimension: CHW -> NCHW (1, C, H, W)
        # This is critical: 3D tensors are interpreted as embeddings by vLLM,
        # but 4D tensors are interpreted as a batch of images.
        tensor = tensor.unsqueeze(0)

        return tensor

    async def load_image(self, image_url: str) -> ImageOutput:
        """Load an image from a URL or data URI."""
        parsed_url = urlparse(image_url)

        # For HTTP(S) URLs, check cache first (PIL path only)
        if not self._use_nvimgcodec and parsed_url.scheme in ("http", "https"):
            image_url_lower = image_url.lower()
            if image_url_lower in self._image_cache:
                logger.debug(f"Image found in cache for URL: {image_url}")
                return self._image_cache[image_url_lower]

        try:
            if parsed_url.scheme == "data":
                # Parse data URL format: data:[<media type>][;base64],<data>
                if not parsed_url.path.startswith("image/"):
                    raise ValueError("Data URL must be an image type")

                # Split the path into media type and data
                media_type, data = parsed_url.path.split(",", 1)
                if ";base64" not in media_type:
                    raise ValueError("Data URL must be base64 encoded")

                try:
                    image_bytes = base64.b64decode(data)
                except binascii.Error as e:
                    raise ValueError(f"Invalid base64 encoding: {e}")
            elif parsed_url.scheme in ("http", "https"):
                http_client = get_http_client(self._http_timeout)

                response = await http_client.get(image_url)
                response.raise_for_status()

                if not response.content:
                    raise ValueError("Empty response content from image URL")

                image_bytes = response.content
            else:
                raise ValueError(f"Invalid image source scheme: {parsed_url.scheme}")

            # Wait if too many decoded images are pending in the vLLM scheduler.
            # Released when the caller invokes mark_consumed() after prefill.
            await self._pending_semaphore.acquire()

            try:
                if self._use_nvimgcodec:
                    # nvimgcodec decoding (GPU-accelerated, returns 4D tensor)
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(
                        _decode_thread_pool,
                        self._decode_with_nvimgcodec,
                        image_bytes,
                    )
                else:
                    # Original PIL path
                    image_data = BytesIO(image_bytes)
                    image = await asyncio.to_thread(Image.open, image_data)

                    # Validate image format and convert to RGB
                    if image.format not in ("JPEG", "PNG", "WEBP"):
                        raise ValueError(f"Unsupported image format: {image.format}")

                    image_converted = image.convert("RGB")

                    # Cache HTTP(S) URLs
                    if parsed_url.scheme in ("http", "https"):
                        image_url_lower = image_url.lower()
                        if self._cache_queue.full():
                            oldest_image_url = await self._cache_queue.get()
                            del self._image_cache[oldest_image_url]

                        self._image_cache[image_url_lower] = image_converted
                        await self._cache_queue.put(image_url_lower)

                    return image_converted

            except Exception:
                # Release semaphore on decode failure to prevent leak
                self._pending_semaphore.release()
                raise

        except httpx.HTTPError as e:
            logger.error(f"HTTP error loading image: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise ValueError(f"Failed to load image: {e}")
