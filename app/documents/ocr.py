"""Document representation: canonical page images plus every OCR result.

Canonical coordinate space
--------------------------
Each page has exactly one canonical image (rendered at `dpi`, orientation- and
skew-normalised once). All OCR engines and preprocessing variants read that
image, or a same-size pixel filter of it, and Gemini is shown that same image.
Every bounding box in the representation is therefore in canonical page pixels,
and page width/height travel with it.

Digital PDF pages take their text and REAL positions from the PDF (pdfplumber
words, scaled from points to canonical pixels). Pages whose geometry cannot be
mapped reliably (e.g. rotated pages) are OCR'd from the render instead; nothing
is given synthetic coordinates.
"""

import gc
import hashlib
import io
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

from PIL import Image, ImageOps

from app.documents.ingestion import DocumentManifest, InvalidImageError
from app.documents.ocr_contract import (
    GEOMETRY_ESTIMATED,
    GEOMETRY_REAL,
    OcrLineEvidence,
    OcrResult,
    make_line_id,
)
from app.documents.ocr_engines import (  # re-exported for existing importers
    OCR_UNAVAILABLE_CODE,
    OCR_UNAVAILABLE_MESSAGE,
    OcrUnavailableError,
    PaddleOcrProvider,
    RapidOcrProvider,
)
from app.documents.ocr_orchestrator import OcrOrchestrator, build_default_orchestrator
from app.documents.pdf_images import InvalidPdfError, is_pdf, pdf_page_count, poppler_available
from app.documents.preprocessing import AdaptivePreprocessor

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentOcrService", "DocumentRepresentation", "OcrLine", "OcrPage", "group_rows",
    "OcrUnavailableError", "DocumentUnreadableError", "RapidOcrProvider", "PaddleOcrProvider",
    "OCR_UNAVAILABLE_CODE", "OCR_UNAVAILABLE_MESSAGE", "DocumentTooLargeError",
]

DOCUMENT_UNREADABLE_CODE = "DOCUMENT_UNREADABLE"
DOCUMENT_UNREADABLE_MESSAGE = "No readable text was found in the document. Upload a clearer scan or photo."

CANONICAL_DPI = int(os.getenv("OCR_CANONICAL_DPI", "200"))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "20"))
# Longest side of a canonical page. Phone photos are downscaled to this.
MAX_CANONICAL_SIDE = int(os.getenv("MAX_CANONICAL_SIDE", "2600"))
# Printed text reads at ~0.97+; handwriting rarely does. A heuristic, reported as such.
HANDWRITING_CONFIDENCE = 0.9

# Decompression-bomb guard: refuse images whose pixel count is implausible for a
# document page instead of letting PIL allocate gigabytes.
Image.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(60_000_000)))


class DocumentUnreadableError(ValueError):
    code = DOCUMENT_UNREADABLE_CODE

    def __init__(self):
        super().__init__(DOCUMENT_UNREADABLE_MESSAGE)


class DocumentTooLargeError(ValueError):
    code = "DOCUMENT_TOO_LARGE"


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in canonical page pixels
    page: int
    engine: str
    index: int = 0
    line_id: str = ""
    variant: str = ""

    @property
    def x0(self) -> int:
        return self.bbox[0]

    @property
    def x1(self) -> int:
        return self.bbox[2]

    @property
    def y_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "OcrLine":
        return cls(
            text=data["text"],
            confidence=float(data["confidence"]),
            bbox=tuple(int(v) for v in data["bbox"]),
            page=int(data["page"]),
            engine=data.get("engine", ""),
            index=int(data.get("index", 0)),
            line_id=str(data.get("line_id", "")),
            variant=str(data.get("variant", "")),
        )

    @classmethod
    def from_evidence(cls, ev: OcrLineEvidence, index: int) -> "OcrLine":
        return cls(text=ev.text, confidence=ev.confidence, bbox=ev.bbox, page=ev.page,
                   engine=ev.engine, index=index, line_id=ev.line_id, variant=ev.variant)


@dataclass
class OcrPage:
    page_number: int
    width: int
    height: int
    lines: list[OcrLine]
    engine: str
    preprocessing: str = "none"
    script: str = "PRINTED"
    geometry: str = GEOMETRY_REAL
    # Every OCR result produced for this page (all engines, all variants).
    evidence: list[OcrResult] = field(default_factory=list)
    routing: dict = field(default_factory=dict)

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return round(sum(line.confidence for line in self.lines) / len(self.lines), 3)

    @property
    def text(self) -> str:
        return "\n".join(" ".join(line.text for line in row) for row in group_rows(self.lines))

    def result(self, engine: str, variant: str | None = None) -> OcrResult | None:
        for res in self.evidence:
            if res.engine == engine and (variant is None or res.variant == variant):
                return res
        return None

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "engine": self.engine,
            "preprocessing": self.preprocessing,
            "script": self.script,
            "geometry": self.geometry,
            "mean_confidence": self.mean_confidence,
            "lines": [line.to_dict() for line in self.lines],
            "evidence": [res.to_dict() for res in self.evidence],
            "routing": self.routing,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OcrPage":
        return cls(
            page_number=int(data["page_number"]),
            width=int(data["width"]),
            height=int(data["height"]),
            lines=[OcrLine.from_dict(line) for line in data.get("lines", [])],
            engine=data.get("engine", ""),
            preprocessing=data.get("preprocessing", "none"),
            script=data.get("script", "PRINTED"),
            geometry=data.get("geometry", GEOMETRY_REAL),
            evidence=[OcrResult.from_dict(r) for r in data.get("evidence", [])],
            routing=dict(data.get("routing", {})),
        )


@dataclass
class DocumentRepresentation:
    manifest: dict
    pages: list[OcrPage]
    engine: str
    dpi: int
    warnings: list[str] = field(default_factory=list)
    # Canonical page images as JPEG bytes (transient, never persisted). These are
    # the exact pixels OCR read, so Gemini and crops share OCR's coordinates.
    page_jpegs: list[bytes] = field(default_factory=list, repr=False)

    @property
    def all_lines(self) -> list[OcrLine]:
        return [line for page in self.pages for line in page.lines]

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def mean_confidence(self) -> float:
        lines = self.all_lines
        if not lines:
            return 0.0
        return round(sum(line.confidence for line in lines) / len(lines), 3)

    def page_image(self, page_number: int) -> Image.Image | None:
        idx = page_number - 1
        if 0 <= idx < len(self.page_jpegs) and self.page_jpegs[idx]:
            return Image.open(io.BytesIO(self.page_jpegs[idx])).convert("RGB")
        return None

    def page_images(self) -> list[Image.Image]:
        return [img for img in (self.page_image(p.page_number) for p in self.pages) if img is not None]

    def summary(self) -> dict:
        return {
            "engine": self.engine,
            "dpi": self.dpi,
            "page_count": len(self.pages),
            "line_count": len(self.all_lines),
            "mean_confidence": self.mean_confidence,
            "pages": [
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "lines": len(page.lines),
                    "mean_confidence": page.mean_confidence,
                    "preprocessing": page.preprocessing,
                    "script": page.script,
                    "geometry": page.geometry,
                    "engines": sorted({r.engine for r in page.evidence}),
                    "results": [
                        {"engine": r.engine, "variant": r.variant, "lines": len(r.lines),
                         "mean_confidence": r.mean_confidence, "duration_ms": r.duration_ms}
                        for r in page.evidence
                    ],
                    "routing": page.routing,
                }
                for page in self.pages
            ],
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict:
        return {
            "manifest": self.manifest,
            "engine": self.engine,
            "dpi": self.dpi,
            "warnings": list(self.warnings),
            "pages": [page.to_dict() for page in self.pages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRepresentation":
        return cls(
            manifest=data.get("manifest", {}),
            pages=[OcrPage.from_dict(page) for page in data.get("pages", [])],
            engine=data.get("engine", ""),
            dpi=int(data.get("dpi", 0)),
            warnings=list(data.get("warnings", [])),
        )


def group_rows(lines: list, overlap: float = 0.5) -> list[list]:
    """Lines whose vertical extents overlap by `overlap` of the shorter one share a visual row."""
    rows: list[list] = []
    for line in sorted(lines, key=lambda item: (item.page, item.y_center)):
        placed = False
        for row in reversed(rows[-3:]):
            anchor = row[0]
            if anchor.page != line.page:
                continue
            top = max(anchor.bbox[1], line.bbox[1])
            bottom = min(anchor.bbox[3], line.bbox[3])
            if bottom - top >= overlap * min(anchor.height, line.height):
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])
    return [sorted(row, key=lambda item: item.x0) for row in rows]


def _jpeg(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _cap_size(image: Image.Image) -> Image.Image:
    longest = max(image.size)
    if longest <= MAX_CANONICAL_SIDE:
        return image
    scale = MAX_CANONICAL_SIDE / longest
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)


def pdf_text_segments(pdf_bytes: bytes, page_number: int, scale: float) -> tuple[list[tuple[str, tuple[int, int, int, int]]], int, int] | None:
    """Real text segments of a digital PDF page in canonical pixels.

    pdfplumber's line extraction merges a whole table row into one line, which
    destroys column positions. Segments are built from word boxes instead:
    words on one baseline are split wherever the horizontal gap is wider than a
    typical inter-word space, which reproduces the per-cell boxes OCR produces.
    Returns None when the page cannot be mapped reliably (rotation).
    """
    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page = pdf.pages[page_number - 1]
        if int(page.rotation or 0) % 360 != 0:
            return None
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False, extra_attrs=["size"])
        width_px = int(round(float(page.width) * scale))
        height_px = int(round(float(page.height) * scale))

    if not words:
        return [], width_px, height_px

    words.sort(key=lambda w: (round(float(w["top"]) / 2), float(w["x0"])))
    lines: list[list[dict]] = []
    for word in words:
        mid = (float(word["top"]) + float(word["bottom"])) / 2
        if lines:
            last = lines[-1]
            ref_top = min(float(w["top"]) for w in last)
            ref_bottom = max(float(w["bottom"]) for w in last)
            if ref_top - 1 <= mid <= ref_bottom + 1:
                last.append(word)
                continue
        lines.append([word])

    segments = []
    for line in lines:
        line.sort(key=lambda w: float(w["x0"]))
        size = sorted(float(w.get("size") or (float(w["bottom"]) - float(w["top"]))) for w in line)[len(line) // 2]
        gap_limit = max(2.5, size * 0.9)
        current = [line[0]]
        for word in line[1:]:
            if float(word["x0"]) - float(current[-1]["x1"]) > gap_limit:
                segments.append(current)
                current = [word]
            else:
                current.append(word)
        segments.append(current)

    out = []
    for seg in segments:
        text = " ".join(w["text"] for w in seg).strip()
        if not text:
            continue
        x0 = min(float(w["x0"]) for w in seg) * scale
        x1 = max(float(w["x1"]) for w in seg) * scale
        y0 = min(float(w["top"]) for w in seg) * scale
        y1 = max(float(w["bottom"]) for w in seg) * scale
        out.append((text, (int(x0), int(y0), int(round(x1)), int(round(y1)))))
    return out, width_px, height_px


class DocumentOcrService:
    """Builds `DocumentRepresentation`s, caching by document hash.

    OCR engines are reached only through the injected `OcrOrchestrator`.
    """

    def __init__(
        self,
        poppler_path: str | None,
        orchestrator: OcrOrchestrator | None = None,
        dpi: int = CANONICAL_DPI,
        cache_size: int = 4,
        max_pages: int = MAX_PDF_PAGES,
        **_legacy,
    ):
        self.poppler_path = poppler_path
        self._orchestrator = orchestrator
        self.dpi = dpi
        self.cache_size = cache_size
        self.max_pages = max_pages
        self._cache: OrderedDict[str, DocumentRepresentation] = OrderedDict()
        self._lock = threading.Lock()
        self._build_locks: dict[str, threading.Lock] = {}
        self.hits = 0
        self.misses = 0

    @property
    def orchestrator(self) -> OcrOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = build_default_orchestrator()
        return self._orchestrator

    def status(self) -> dict:
        return {**self.orchestrator.status(), "dpi": self.dpi, "cache_hits": self.hits, "cache_misses": self.misses}

    def cached(self, sha256: str) -> DocumentRepresentation | None:
        with self._lock:
            item = self._cache.get(sha256)
            if item:
                self._cache.move_to_end(sha256)
                self.hits += 1
                return item
            self.misses += 1
            return None

    def remember(self, representation: DocumentRepresentation):
        sha = representation.manifest.get("sha256")
        if not sha:
            return
        with self._lock:
            self._cache[sha] = representation
            self._cache.move_to_end(sha)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def represent(self, data: bytes, manifest: DocumentManifest | None = None) -> DocumentRepresentation:
        sha = manifest.sha256 if manifest else hashlib.sha256(data).hexdigest()
        cached = self.cached(sha)
        if cached:
            return cached
        # One build per document even when extraction and verification ask at once.
        with self._lock:
            build_lock = self._build_locks.setdefault(sha, threading.Lock())
        with build_lock:
            cached = self.cached(sha)
            if cached:
                return cached
            rep = self._build(data, manifest, sha)
            self.remember(rep)
        with self._lock:
            self._build_locks.pop(sha, None)
        return rep

    # -- building -------------------------------------------------------------
    def _build(self, data: bytes, manifest: DocumentManifest | None, sha: str) -> DocumentRepresentation:
        pages: list[OcrPage] = []
        jpegs: list[bytes] = []
        warnings: list[str] = []

        if is_pdf(data):
            count = pdf_page_count(data, self.poppler_path)
            if count > self.max_pages:
                raise DocumentTooLargeError(
                    f"The PDF has {count} pages; at most {self.max_pages} pages can be processed per document."
                )
            for number in range(1, count + 1):
                page, jpeg = self._pdf_page(data, number, warnings)
                pages.append(page)
                jpegs.append(jpeg)
                gc.collect()
        else:
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
            except Image.DecompressionBombError as exc:
                raise DocumentTooLargeError("The image is too large to process safely.") from exc
            except Exception as exc:
                raise InvalidImageError() from exc
            image = ImageOps.exif_transpose(image).convert("RGB")
            canonical = AdaptivePreprocessor.variant_a_normalized(_cap_size(image))
            pages.append(self._ocr_page(1, canonical))
            jpegs.append(_jpeg(canonical))
            image.close()

        for page in pages:
            if not page.lines:
                warnings.append(f"Page {page.page_number}: no readable text.")
        manifest_dict = manifest.to_dict() if manifest else {"sha256": sha, "size_bytes": len(data)}
        rep = DocumentRepresentation(
            manifest=manifest_dict,
            pages=pages,
            engine="+".join(sorted({r.engine for p in pages for r in p.evidence})) or "none",
            dpi=self.dpi,
            warnings=warnings,
            page_jpegs=jpegs,
        )
        # No OCR text is a warning, not a failure: a page OCR cannot read may still
        # be legible to the visual model, and the ledger records that nothing was read.
        return rep

    def _render_pdf_page(self, data: bytes, number: int) -> Image.Image:
        from pdf2image import convert_from_bytes

        if not poppler_available(self.poppler_path):
            from app.documents.pdf_images import PdfDependencyMissingError
            raise PdfDependencyMissingError()
        images = convert_from_bytes(data, first_page=number, last_page=number, dpi=self.dpi,
                                    fmt="jpeg", poppler_path=self.poppler_path or None)
        if not images:
            raise InvalidPdfError()
        img = images[0].convert("RGB")
        images[0].close()
        return img

    def _pdf_page(self, data: bytes, number: int, warnings: list[str]) -> tuple[OcrPage, bytes]:
        from app.documents.pdf_inspector import assess_text_reliability

        rendered = self._render_pdf_page(data, number)
        segments = None
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            embedded = reader.pages[number - 1].extract_text() or ""
            reliable, _ = assess_text_reliability(embedded)
            if reliable:
                segments = pdf_text_segments(data, number, self.dpi / 72.0)
                if segments is None:
                    warnings.append(f"Page {number}: rotated digital page; its text is read by OCR so geometry stays real.")
        except Exception as exc:
            logger.warning("Digital text read failed for page %s (%s); using OCR.", number, type(exc).__name__)
            segments = None

        if segments is not None:
            items, width, height = segments
            if (width, height) != rendered.size:
                # Poppler rounding can differ by a pixel; scale boxes onto the render.
                sx, sy = rendered.width / max(1, width), rendered.height / max(1, height)
                items = [(t, (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))) for t, b in items]
            evidence_lines = [
                OcrLineEvidence(line_id=make_line_id("pdf_text", "embedded", number, i), engine="pdf_text",
                                variant="embedded", page=number, text=text, confidence=1.0, bbox=bbox)
                for i, (text, bbox) in enumerate(items)
            ]
            result = OcrResult(engine="pdf_text", page=number, variant="embedded", width=rendered.width,
                               height=rendered.height, lines=evidence_lines, geometry=GEOMETRY_REAL)
            page = OcrPage(
                page_number=number, width=rendered.width, height=rendered.height,
                lines=[OcrLine.from_evidence(ev, i) for i, ev in enumerate(evidence_lines)],
                engine="pdf_text", preprocessing="digital_text", script="PRINTED", geometry=GEOMETRY_REAL,
                evidence=[result],
                routing={"page": number, "difficulty": "DIGITAL", "reasons": ["reliable embedded PDF text"],
                         "selected": "pdf_text:embedded"},
            )
            jpeg = _jpeg(rendered)
            rendered.close()
            return page, jpeg

        canonical = AdaptivePreprocessor.variant_a_normalized(rendered)
        page = self._ocr_page(number, canonical)
        jpeg = _jpeg(canonical)
        rendered.close()
        return page, jpeg

    def _ocr_page(self, number: int, canonical: Image.Image) -> OcrPage:
        evidence = self.orchestrator.read_page(number, canonical)
        primary = evidence.primary
        lines = [OcrLine.from_evidence(ev, i) for i, ev in enumerate(primary.lines)]
        script = "HANDWRITTEN_LIKELY" if lines and primary.mean_confidence < HANDWRITING_CONFIDENCE else "PRINTED"
        return OcrPage(
            page_number=number,
            width=evidence.width,
            height=evidence.height,
            lines=lines,
            engine=primary.engine,
            preprocessing=primary.variant,
            script=script,
            geometry=GEOMETRY_REAL,
            evidence=evidence.results,
            routing=evidence.decision.to_dict(),
        )
