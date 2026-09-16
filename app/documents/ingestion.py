"""Document intake shared by the Legacy Converter and Accounting AI.

Both paths accept the same file types and render PDFs through Poppler; this module
is the single place that turns uploaded bytes into page images and a manifest.
"""

import hashlib
import io
from dataclasses import asdict, dataclass

from PIL import Image, UnidentifiedImageError

from app.documents.pdf_images import is_pdf, pdf_page_count, pdf_to_images


SUPPORTED_DOCUMENT_TYPES = {"image/jpeg", "image/png", "application/pdf"}

INVALID_IMAGE_MESSAGE = "The uploaded file is not a readable image."


class InvalidImageError(ValueError):
    def __init__(self):
        super().__init__(INVALID_IMAGE_MESSAGE)


@dataclass(frozen=True)
class DocumentManifest:
    filename: str
    content_type: str
    detected_format: str
    size_bytes: int
    sha256: str
    page_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_format(data: bytes) -> str:
    if is_pdf(data):
        return "PDF"
    if data.startswith(b"\x89PNG"):
        return "PNG"
    if data.startswith(b"\xff\xd8"):
        return "JPEG"
    return "UNKNOWN"


import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class PageImageCache:
    """Bounded, thread-safe in-memory cache for rendered document page images.

    Keyed strictly by (sha256, dpi) to guarantee that different documents never collide.
    Bounded capacity (default max 16 entries) with automatic eviction prevents memory leaks.
    """
    def __init__(self, capacity: int = 16):
        self._capacity = capacity
        self._cache: OrderedDict[tuple[str, int], list[Image.Image]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, sha256: str, dpi: int) -> list[Image.Image] | None:
        with self._lock:
            key = (sha256, dpi)
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                logger.debug("PAGE_IMAGE_CACHE_HIT: sha256=%s, dpi=%d (total hits=%d)", sha256[:12], dpi, self.hits)
                # Return independent copies so callers cannot mutate cached PIL objects
                return [img.copy() for img in self._cache[key]]
            self.misses += 1
            logger.debug("PAGE_IMAGE_CACHE_MISS: sha256=%s, dpi=%d (total misses=%d)", sha256[:12], dpi, self.misses)
            return None

    def put(self, sha256: str, dpi: int, images: list[Image.Image]):
        with self._lock:
            key = (sha256, dpi)
            self._cache[key] = [img.copy() for img in images]
            self._cache.move_to_end(key)
            while len(self._cache) > self._capacity:
                _, evicted_images = self._cache.popitem(last=False)
                for img in evicted_images:
                    try:
                        img.close()
                    except Exception:
                        pass

    def clear(self):
        with self._lock:
            for imgs in self._cache.values():
                for img in imgs:
                    try:
                        img.close()
                    except Exception:
                        pass
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


_PAGE_IMAGE_CACHE = PageImageCache(capacity=16)


def get_page_image_cache() -> PageImageCache:
    return _PAGE_IMAGE_CACHE


def load_page_images(data: bytes, poppler_path: str | None, dpi: int, use_cache: bool = True) -> list[Image.Image]:
    """Every page as an RGB image: all PDF pages via Poppler, or the single image."""
    sha = sha256_hex(data) if (use_cache and len(data) > 0) else None
    if sha and use_cache:
        cached = _PAGE_IMAGE_CACHE.get(sha, dpi)
        if cached is not None:
            return cached

    if is_pdf(data):
        images = pdf_to_images(data, poppler_path, dpi=dpi)
    else:
        try:
            images = [Image.open(io.BytesIO(data)).convert("RGB")]
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError() from exc

    if sha and use_cache and images:
        _PAGE_IMAGE_CACHE.put(sha, dpi, images)
    return images


def build_manifest(data: bytes, filename: str, content_type: str, poppler_path: str | None) -> DocumentManifest:
    detected = detect_format(data)
    if detected == "PDF":
        pages = pdf_page_count(data, poppler_path)
    elif detected in {"PNG", "JPEG"}:
        pages = 1
    else:
        raise InvalidImageError()
    return DocumentManifest(
        filename=filename,
        content_type=content_type,
        detected_format=detected,
        size_bytes=len(data),
        sha256=sha256_hex(data),
        page_count=pages,
    )
