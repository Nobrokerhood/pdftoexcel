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


def load_page_images(data: bytes, poppler_path: str | None, dpi: int) -> list[Image.Image]:
    """Every page as an RGB image: all PDF pages via Poppler, or the single image."""
    if is_pdf(data):
        return pdf_to_images(data, poppler_path, dpi=dpi)
    try:
        return [Image.open(io.BytesIO(data)).convert("RGB")]
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError() from exc


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
