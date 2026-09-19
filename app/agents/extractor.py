"""Extraction agent.

`GeminiExtractionProvider` runs the full evidence pipeline
(`DocumentExtractionPipeline`): canonical OCR evidence from every engine,
source candidates, a purpose-specific Gemini contract, fusion and blind visual
arbitration. It never instantiates an OCR engine; it receives the shared
`DocumentOcrService` (which owns the orchestrator).
"""

import logging
from typing import Protocol

from app.accounting.document_result import finalize_extraction
from app.accounting.templates import TemplateDefinition
from app.agents.extraction_pipeline import DocumentExtractionPipeline
from app.documents.ingestion import load_page_images
from app.documents.ocr import DocumentOcrService

logger = logging.getLogger(__name__)


class ExtractionProvider(Protocol):
    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        ...


class GeminiExtractionProvider:
    def __init__(self, gemini_client, ocr_service: DocumentOcrService | None = None):
        self.gemini_client = gemini_client
        self._ocr_service = ocr_service

    def _pipeline(self) -> DocumentExtractionPipeline:
        if self._ocr_service is None:
            self._ocr_service = DocumentOcrService(self.gemini_client.settings.poppler_path)
        return DocumentExtractionPipeline(self.gemini_client, self._ocr_service)

    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        return self._pipeline().extract(source_bytes, purpose, template)


class ExtractionAgent:
    def __init__(self, provider: ExtractionProvider):
        self.provider = provider

    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        return finalize_extraction(purpose, self.provider.extract(source_bytes, purpose, template))


def validate_extraction(purpose: str, result: dict) -> dict:
    """Compatibility name for the strict result finaliser."""
    return finalize_extraction(purpose, result)


def source_parts(source_bytes: bytes, poppler_path: str | None = None):
    return load_page_images(source_bytes, poppler_path, dpi=150)
