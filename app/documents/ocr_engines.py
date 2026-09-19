"""OCR engine adapters. Each implements `OcrProvider` from ocr_contract.

Availability is always proven by a real inference call, never by an import:
paddlepaddle 3.3.1 imports cleanly and then fails inside its C++ executor, so an
import check reports an engine READY that cannot read a single page.

Consumers must not instantiate these classes; `OcrOrchestrator` owns them.
"""

import logging
import threading
import time

import numpy as np
from PIL import Image

from app.core.errors import ExternalServiceUnavailableError
from app.documents.ocr_contract import OcrLineEvidence, OcrResult, make_line_id

logger = logging.getLogger(__name__)

OCR_UNAVAILABLE_CODE = "OCR_ENGINE_UNAVAILABLE"
OCR_UNAVAILABLE_MESSAGE = (
    "Document reading is unavailable because the local OCR engine is not installed "
    "or failed to start. Please contact the administrator."
)


class OcrUnavailableError(ExternalServiceUnavailableError):
    code = OCR_UNAVAILABLE_CODE

    def __init__(self, detail: str = ""):
        super().__init__(OCR_UNAVAILABLE_MESSAGE)
        self.detail = detail


def _bbox_from_points(points) -> tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=float)
    return (
        int(pts[:, 0].min()), int(pts[:, 1].min()),
        int(pts[:, 0].max()), int(pts[:, 1].max()),
    )


class _ProvenEngine:
    """Shared availability probing: one real inference, cached with its reason."""

    name = "engine"

    def __init__(self):
        self._engine = None
        self._available: bool | None = None
        self._unavailable_reason = ""
        self._lock = threading.Lock()

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def _import_check(self) -> None:
        raise NotImplementedError

    def _infer(self, array: np.ndarray):
        raise NotImplementedError

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            self._import_check()
        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"{self.name} not importable: {type(exc).__name__}"
            return False
        try:
            probe = np.full((48, 160, 3), 255, dtype=np.uint8)
            with self._lock:
                self._infer(probe)
            self._available = True
            self._unavailable_reason = ""
        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("%s is installed but inference failed: %s", self.name, self._unavailable_reason)
        return self._available

    def _parse(self, raw) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        raise NotImplementedError

    def read(self, image: Image.Image, page_idx: int | None = None, variant: str | None = None) -> OcrResult:
        page = int(page_idx or 1)
        variant = variant or "original"
        rgb = image.convert("RGB")
        array = np.array(rgb)
        started = time.perf_counter()
        warnings: list[str] = []
        try:
            with self._lock:
                raw = self._infer(array)
            parsed = self._parse(raw)
        except OcrUnavailableError:
            raise
        except Exception as exc:
            raise OcrUnavailableError(f"{self.name}: {type(exc).__name__}: {str(exc)[:160]}") from exc
        duration_ms = int((time.perf_counter() - started) * 1000)
        lines = []
        for index, (text, confidence, bbox) in enumerate(parsed):
            clean = str(text).strip()
            if not clean:
                continue
            x0, y0, x1, y1 = bbox
            if x1 <= x0 or y1 <= y0:
                warnings.append(f"line {index}: degenerate box {bbox} for '{clean[:20]}'")
            lines.append(OcrLineEvidence(
                line_id=make_line_id(self.name, variant, page, len(lines)),
                engine=self.name,
                variant=variant,
                page=page,
                text=clean,
                confidence=round(max(0.0, min(1.0, float(confidence))), 4),
                bbox=(int(x0), int(y0), int(x1), int(y1)),
            ))
        return OcrResult(
            engine=self.name, page=page, variant=variant,
            width=rgb.width, height=rgb.height, lines=lines,
            duration_ms=duration_ms, warnings=warnings,
        )


class RapidOcrProvider(_ProvenEngine):
    name = "rapidocr"

    def _import_check(self) -> None:
        import onnxruntime  # noqa: F401
        import rapidocr  # noqa: F401

    def _get_engine(self):
        if self._engine is None:
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OcrUnavailableError("rapidocr is not installed") from exc
            self._engine = RapidOCR()
            for name in list(logging.root.manager.loggerDict):
                if "rapidocr" in name.lower():
                    logging.getLogger(name).setLevel(logging.WARNING)
        return self._engine

    def _infer(self, array: np.ndarray):
        return self._get_engine()(array)

    def _parse(self, raw):
        if raw is None or getattr(raw, "txts", None) is None:
            return []
        out = []
        for box, text, score in zip(raw.boxes, raw.txts, raw.scores):
            out.append((str(text), float(score), _bbox_from_points(box)))
        return out


class PaddleOcrProvider(_ProvenEngine):
    """PaddleOCR 3.x (predict API), CPU. Mobile models keep inference tractable."""

    name = "paddleocr"
    DEFAULT_DET_MODEL = "PP-OCRv5_mobile_det"
    DEFAULT_REC_MODEL = "PP-OCRv5_mobile_rec"

    def __init__(self, lang: str = "en", det_model: str | None = None, rec_model: str | None = None):
        super().__init__()
        self.lang = lang
        self.det_model = det_model or self.DEFAULT_DET_MODEL
        self.rec_model = rec_model or self.DEFAULT_REC_MODEL

    def _import_check(self) -> None:
        import paddleocr  # noqa: F401

    def _get_engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                text_detection_model_name=self.det_model,
                text_recognition_model_name=self.rec_model,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._engine

    def _infer(self, array: np.ndarray):
        return self._get_engine().predict(array)

    def _parse(self, raw):
        if not raw:
            return []
        block = raw[0]
        if not hasattr(block, "get"):
            return []
        texts = block.get("rec_texts") or []
        scores = block.get("rec_scores") or []
        polys = block.get("rec_polys")
        if polys is None:
            polys = block.get("dt_polys") or []
        out = []
        for idx, text in enumerate(texts):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            bbox = _bbox_from_points(polys[idx]) if idx < len(polys) else (0, 0, 0, 0)
            out.append((str(text), score, bbox))
        return out
