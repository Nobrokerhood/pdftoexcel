"""Compatibility shim. Engine adapters live in `ocr_engines`; selection lives in
`ocr_orchestrator`. The former `OcrRouter` was never wired into the pipeline and
has been removed in favour of `OcrOrchestrator`."""

from app.documents.ocr_engines import OcrUnavailableError, PaddleOcrProvider, RapidOcrProvider  # noqa: F401
