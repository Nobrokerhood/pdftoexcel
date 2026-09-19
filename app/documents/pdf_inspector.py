"""
Page-Level PDF Inspector & Adaptive Router.
Inspects each PDF page individually to evaluate embedded text reliability vs. scanned images.
Enables hybrid page-by-page routing, preserving page indices and avoiding unnecessary rasterization.
"""

import io
import logging
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Tuple
import pypdf
from PIL import Image
from pdf2image import convert_from_bytes
from app.documents.pdf_images import pdf_page_count, poppler_available, PdfDependencyMissingError, InvalidPdfError

logger = logging.getLogger(__name__)


@dataclass
class PageInspectionResult:
    page_number: int  # 1-indexed
    is_digital: bool
    is_scanned: bool
    text_char_count: int
    printable_ratio: float
    embedded_text: str
    layout_lines: List[str] = field(default_factory=list)
    routing_decision: str = "SCANNED_RENDER"  # "DIGITAL_TEXT" or "SCANNED_RENDER"


def assess_text_reliability(text: str) -> Tuple[bool, float]:
    """
    Evaluates if extracted embedded text is reliable digital vector text.
    Returns (is_reliable, printable_ratio).
    """
    if not text or len(text.strip()) < 50:
        return False, 0.0

    printable_count = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    printable_ratio = printable_count / len(text)

    # Corrupted font glyphs often produce high density of replacement or control characters
    replacement_chars = text.count("\ufffd") + text.count("?")
    if replacement_chars / len(text) > 0.15:
        return False, round(printable_ratio, 3)

    if printable_ratio < 0.85:
        return False, round(printable_ratio, 3)

    # Check for basic whitespace and token structure
    tokens = [t for t in text.split() if len(t) > 0]
    if len(tokens) < 5:
        return False, round(printable_ratio, 3)

    return True, round(printable_ratio, 3)


class PageLevelPdfInspector:
    """Inspects PDF documents page-by-page and directs page processing."""

    def __init__(self, poppler_path: Optional[str] = None):
        self.poppler_path = poppler_path

    def inspect_document(self, pdf_bytes: bytes) -> List[PageInspectionResult]:
        """Inspects all pages in the PDF and returns a list of PageInspectionResults."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            raise InvalidPdfError() from e

        results: List[PageInspectionResult] = []
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Failed to extract text from page %d: %s", page_num, e)
                raw_text = ""

            char_count = len(raw_text.strip())
            is_reliable, ratio = assess_text_reliability(raw_text)
            
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

            decision = "DIGITAL_TEXT" if is_reliable else "SCANNED_RENDER"
            results.append(PageInspectionResult(
                page_number=page_num,
                is_digital=is_reliable,
                is_scanned=not is_reliable,
                text_char_count=char_count,
                printable_ratio=ratio,
                embedded_text=raw_text,
                layout_lines=lines,
                routing_decision=decision
            ))

        return results

    def render_single_page(
        self,
        pdf_bytes: bytes,
        page_number: int,
        dpi: int = 150
    ) -> Image.Image:
        """Renders a single scanned page on demand with bounded memory."""
        if not poppler_available(self.poppler_path):
            raise PdfDependencyMissingError()

        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
            fmt="jpeg",
            poppler_path=self.poppler_path or None
        )
        if not images:
            raise InvalidPdfError()
        img = images[0].convert("RGB")
        images[0].close()
        return img

    def stream_pages(
        self,
        pdf_bytes: bytes,
        dpi: int = 150
    ) -> Iterator[Tuple[PageInspectionResult, Optional[Image.Image]]]:
        """
        Streams pages one-by-one. Yields (inspection_result, optional_image).
        Image is rendered only if page is scanned or required.
        """
        inspections = self.inspect_document(pdf_bytes)
        for insp in inspections:
            rendered_image = None
            if insp.routing_decision == "SCANNED_RENDER":
                rendered_image = self.render_single_page(pdf_bytes, insp.page_number, dpi=dpi)
            yield insp, rendered_image
            # Bounded memory cleanup
            if rendered_image:
                try:
                    rendered_image.close()
                except Exception:
                    pass
            gc.collect()
