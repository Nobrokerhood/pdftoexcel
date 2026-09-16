import gc
import logging
import os
import shutil

from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
    PopplerNotInstalledError,
)
from PIL import Image

from app.core.errors import ExternalServiceUnavailableError


logger = logging.getLogger(__name__)

PDF_DEPENDENCY_MISSING_CODE = "PDF_PROCESSING_DEPENDENCY_MISSING"
PDF_DEPENDENCY_MISSING_MESSAGE = (
    "PDF processing is unavailable because the PDF processing dependency "
    "(Poppler) is not installed. JPG and PNG uploads still work."
)
INVALID_PDF_MESSAGE = "The uploaded file is not a readable PDF."

REQUIRED_POPPLER_TOOLS = ("pdfinfo", "pdftoppm")


class PdfDependencyMissingError(ExternalServiceUnavailableError):
    code = PDF_DEPENDENCY_MISSING_CODE

    def __init__(self):
        super().__init__(PDF_DEPENDENCY_MISSING_MESSAGE)


class InvalidPdfError(ValueError):
    def __init__(self):
        super().__init__(INVALID_PDF_MESSAGE)


def is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def _tool_in_dir(directory: str, tool: str) -> bool:
    return any(
        os.path.isfile(os.path.join(directory, name))
        for name in (tool, f"{tool}.exe")
    )


def poppler_available(poppler_path: str | None) -> bool:
    """True when pdfinfo and pdftoppm exist in POPPLER_PATH, or on PATH if it is unset."""
    if poppler_path:
        return all(_tool_in_dir(poppler_path, tool) for tool in REQUIRED_POPPLER_TOOLS)
    return all(shutil.which(tool) for tool in REQUIRED_POPPLER_TOOLS)


def pdf_page_count(pdf_bytes: bytes, poppler_path: str | None) -> int:
    """Cheap pre-flight check: raises PdfDependencyMissingError or InvalidPdfError."""
    if not poppler_available(poppler_path):
        logger.error("Poppler tools not found (POPPLER_PATH set: %s).", bool(poppler_path))
        raise PdfDependencyMissingError()
    try:
        info = pdfinfo_from_bytes(pdf_bytes, poppler_path=poppler_path or None)
    except (PDFInfoNotInstalledError, PopplerNotInstalledError) as exc:
        raise PdfDependencyMissingError() from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise InvalidPdfError() from exc
    pages = int(info.get("Pages", 0) or 0)
    if pages < 1:
        raise InvalidPdfError()
    return pages


def pdf_to_images(pdf_bytes: bytes, poppler_path: str | None, dpi: int) -> list[Image.Image]:
    pdf_page_count(pdf_bytes, poppler_path)
    images = []
    try:
        for page in convert_from_bytes(
            pdf_bytes, dpi=dpi, fmt="jpeg", poppler_path=poppler_path or None
        ):
            images.append(page.convert("RGB"))
            page.close()
        return images
    except (PDFInfoNotInstalledError, PopplerNotInstalledError) as exc:
        raise PdfDependencyMissingError() from exc
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise InvalidPdfError() from exc
    finally:
        gc.collect()
