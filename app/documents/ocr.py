"""Local OCR for Accounting AI.

RapidOCR (ONNX PP-OCR models) is the primary reader: it returns text lines with
boxes and confidence. Tesseract is a second, independent engine used by the
verifier to re-read cropped regions. Neither needs a network connection.
"""

import gc
import logging
import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from app.core.errors import ExternalServiceUnavailableError
from app.documents.ingestion import DocumentManifest, load_page_images


logger = logging.getLogger(__name__)

OCR_UNAVAILABLE_CODE = "OCR_ENGINE_UNAVAILABLE"
OCR_UNAVAILABLE_MESSAGE = (
    "Document reading is unavailable because the local OCR engine is not installed "
    "or failed to start. Please contact the administrator."
)
DOCUMENT_UNREADABLE_CODE = "DOCUMENT_UNREADABLE"
DOCUMENT_UNREADABLE_MESSAGE = (
    "No readable text was found in the document. Upload a clearer scan or photo."
)

DEFAULT_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

# Pages whose mean line confidence falls below this are re-read after contrast enhancement.
LOW_QUALITY_CONFIDENCE = 0.75
# Printed text reads at ~0.97+; handwriting rarely does. A heuristic, reported as such.
HANDWRITING_CONFIDENCE = 0.9


class OcrUnavailableError(ExternalServiceUnavailableError):
    code = OCR_UNAVAILABLE_CODE

    def __init__(self, detail: str = ""):
        super().__init__(OCR_UNAVAILABLE_MESSAGE)
        self.detail = detail


class DocumentUnreadableError(ValueError):
    code = DOCUMENT_UNREADABLE_CODE

    def __init__(self):
        super().__init__(DOCUMENT_UNREADABLE_MESSAGE)


@dataclass
class OcrLine:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in page pixels
    page: int
    engine: str
    index: int = 0

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
        )


@dataclass
class OcrPage:
    page_number: int
    width: int
    height: int
    lines: list[OcrLine]
    engine: str
    preprocessing: str = "none"
    script: str = "PRINTED"

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0
        return round(sum(line.confidence for line in self.lines) / len(self.lines), 3)

    @property
    def text(self) -> str:
        return "\n".join(" ".join(line.text for line in row) for row in group_rows(self.lines))

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "engine": self.engine,
            "preprocessing": self.preprocessing,
            "script": self.script,
            "mean_confidence": self.mean_confidence,
            "lines": [line.to_dict() for line in self.lines],
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
        )


@dataclass
class DocumentRepresentation:
    manifest: dict
    pages: list[OcrPage]
    engine: str
    dpi: int
    warnings: list[str] = field(default_factory=list)

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
                    "lines": len(page.lines),
                    "mean_confidence": page.mean_confidence,
                    "preprocessing": page.preprocessing,
                    "script": page.script,
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


def group_rows(lines: list[OcrLine], overlap: float = 0.5) -> list[list[OcrLine]]:
    """Lines whose vertical extents overlap by `overlap` of the shorter one share a visual row."""
    rows: list[list[OcrLine]] = []
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


class OcrProvider(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def read(self, image: Image.Image) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        ...


class RapidOcrProvider:
    name = "rapidocr"

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        try:
            import rapidocr  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_engine(self):
        if self._engine is None:
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OcrUnavailableError("rapidocr is not installed") from exc
            try:
                self._engine = RapidOCR()
            except Exception as exc:
                raise OcrUnavailableError(type(exc).__name__) from exc
            for name in list(logging.root.manager.loggerDict):
                if "rapidocr" in name.lower():
                    logging.getLogger(name).setLevel(logging.WARNING)
        return self._engine

    def read(self, image: Image.Image):
        with self._lock:
            result = self._get_engine()(np.array(image.convert("RGB")))
        if result is None or result.txts is None:
            return []
        lines = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            points = np.asarray(box)
            bbox = (
                int(points[:, 0].min()),
                int(points[:, 1].min()),
                int(points[:, 0].max()),
                int(points[:, 1].max()),
            )
            if str(text).strip():
                lines.append((str(text).strip(), float(score), bbox))
        return lines


def find_tesseract(configured: str | None = None) -> str | None:
    candidates = [configured] if configured else []
    candidates += [shutil.which("tesseract"), *DEFAULT_TESSERACT_PATHS]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


class TesseractOcrProvider:
    name = "tesseract"

    def __init__(self, tesseract_cmd: str | None = None):
        self.tesseract_cmd = find_tesseract(tesseract_cmd)

    def available(self) -> bool:
        if not self.tesseract_cmd:
            return False
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False
        return True

    def _pytesseract(self):
        if not self.available():
            raise OcrUnavailableError("tesseract is not installed")
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        return pytesseract

    def read(self, image: Image.Image):
        pytesseract = self._pytesseract()
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        grouped: dict[tuple, list[int]] = {}
        for i, word in enumerate(data["text"]):
            if str(word).strip() and float(data["conf"][i]) >= 0:
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                grouped.setdefault(key, []).append(i)
        lines = []
        for indexes in grouped.values():
            text = " ".join(str(data["text"][i]).strip() for i in indexes)
            conf = sum(float(data["conf"][i]) for i in indexes) / len(indexes) / 100
            x0 = min(data["left"][i] for i in indexes)
            y0 = min(data["top"][i] for i in indexes)
            x1 = max(data["left"][i] + data["width"][i] for i in indexes)
            y1 = max(data["top"][i] + data["height"][i] for i in indexes)
            lines.append((text, round(conf, 3), (x0, y0, x1, y1)))
        return lines

    def read_region(self, image: Image.Image, bbox: tuple[int, int, int, int], numeric: bool = False) -> tuple[str, float]:
        """Independent re-read of one cropped region (used by the verifier)."""
        pytesseract = self._pytesseract()
        pad = 6
        x0, y0, x1, y1 = bbox
        crop = image.crop((max(0, x0 - pad), max(0, y0 - pad), min(image.width, x1 + pad), min(image.height, y1 + pad)))
        crop = ImageOps.autocontrast(crop.convert("L"))
        if crop.height < 40:
            scale = 40 / max(1, crop.height)
            crop = crop.resize((max(1, int(crop.width * scale)), 40))
        config = "--psm 7"
        if numeric:
            config += " -c tessedit_char_whitelist=0123456789,./-"
        data = pytesseract.image_to_data(crop, config=config, output_type=pytesseract.Output.DICT)
        words = [(str(t).strip(), float(c)) for t, c in zip(data["text"], data["conf"]) if str(t).strip() and float(c) >= 0]
        if not words:
            return "", 0.0
        return " ".join(word for word, _ in words), round(sum(c for _, c in words) / len(words) / 100, 3)


def _enhance(image: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(image.convert("L"), cutoff=2)
    return gray.filter(ImageFilter.SHARPEN).convert("RGB")


class DocumentOcrService:
    """Renders every page and reads it with the primary engine, caching by document hash."""

    def __init__(
        self,
        poppler_path: str | None,
        primary: OcrProvider | None = None,
        secondary: TesseractOcrProvider | None = None,
        dpi: int = 200,
        cache_size: int = 4,
    ):
        self.poppler_path = poppler_path
        self.primary = primary or RapidOcrProvider()
        self.secondary = secondary
        self.dpi = dpi
        self.cache_size = cache_size
        self._cache: OrderedDict[str, DocumentRepresentation] = OrderedDict()
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "primary": {"engine": self.primary.name, "available": self.primary.available()},
            "secondary": {
                "engine": self.secondary.name if self.secondary else None,
                "available": bool(self.secondary and self.secondary.available()),
            },
        }

    def page_images(self, data: bytes) -> list[Image.Image]:
        return load_page_images(data, self.poppler_path, self.dpi)

    def cached(self, sha256: str) -> DocumentRepresentation | None:
        with self._lock:
            return self._cache.get(sha256)

    def remember(self, representation: DocumentRepresentation):
        sha = representation.manifest.get("sha256")
        if not sha:
            return
        with self._lock:
            self._cache[sha] = representation
            self._cache.move_to_end(sha)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def represent(self, data: bytes, manifest: DocumentManifest) -> DocumentRepresentation:
        cached = self.cached(manifest.sha256)
        if cached:
            return cached
        if not self.primary.available():
            raise OcrUnavailableError(f"{self.primary.name} is not installed")

        pages: list[OcrPage] = []
        warnings: list[str] = []
        images = self.page_images(data)
        try:
            for number, image in enumerate(images, start=1):
                pages.append(self._read_page(number, image))
                image.close()
                gc.collect()
        finally:
            for image in images:
                image.close()

        representation = DocumentRepresentation(
            manifest=manifest.to_dict(), pages=pages, engine=self.primary.name, dpi=self.dpi, warnings=warnings
        )
        for page in pages:
            if not page.lines:
                warnings.append(f"Page {page.page_number}: no readable text.")
        if not representation.all_lines:
            raise DocumentUnreadableError()
        self.remember(representation)
        return representation

    def _read_page(self, number: int, image: Image.Image) -> OcrPage:
        raw = self.primary.read(image)
        preprocessing = "none"
        mean = sum(conf for _, conf, _ in raw) / len(raw) if raw else 0.0
        if mean < LOW_QUALITY_CONFIDENCE:
            enhanced = self.primary.read(_enhance(image))
            enhanced_mean = sum(conf for _, conf, _ in enhanced) / len(enhanced) if enhanced else 0.0
            # Keep whichever reading recovered more confident text.
            if len(enhanced) * enhanced_mean > len(raw) * mean:
                raw, mean, preprocessing = enhanced, enhanced_mean, "autocontrast+sharpen"
        lines = [
            OcrLine(text=text, confidence=round(conf, 3), bbox=bbox, page=number, engine=self.primary.name, index=i)
            for i, (text, conf, bbox) in enumerate(raw)
        ]
        script = "HANDWRITTEN_LIKELY" if lines and mean < HANDWRITING_CONFIDENCE else "PRINTED"
        return OcrPage(
            page_number=number,
            width=image.width,
            height=image.height,
            lines=lines,
            engine=self.primary.name,
            preprocessing=preprocessing,
            script=script,
        )
