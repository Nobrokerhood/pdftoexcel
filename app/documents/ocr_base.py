"""Compatibility shim: the OCR contract lives in `ocr_contract`."""

from app.documents.ocr_contract import OcrLineEvidence as OcrEvidence  # noqa: F401
from app.documents.ocr_contract import OcrLineEvidence, OcrProvider as OcrEngine, OcrResult  # noqa: F401
