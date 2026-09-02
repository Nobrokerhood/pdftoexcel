from typing import Protocol

from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition
from app.agents.extractor import source_parts
from app.core.errors import ServiceNotConfiguredError
from app.services.gemini_client import GeminiDocumentClient


class VerificationProvider(Protocol):
    def verify(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
    ) -> dict:
        ...


class GeminiVerificationProvider:
    def __init__(self, gemini_client: GeminiDocumentClient):
        self.gemini_client = gemini_client

    def verify(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
    ) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")
        prompt = (
            "You are an independent accounting verification agent. Check whether "
            "each extracted value is supported by the original source. Do not "
            "assume extraction is correct. Return JSON only with overall_status "
            "PASSED, FAILED, or NEEDS_REVIEW and fields[]. Each field item must "
            "include field, extracted_value, verified_value, status, confidence, "
            "evidence, and page_number when available.\n"
            f"Purpose: {purpose}\n"
            f"Template code: {template.template_code}\n"
            f"Extracted data: {extracted_data}"
        )
        return self.gemini_client.generate_json([prompt, *source_parts(source_bytes)])


class VerificationAgent:
    def __init__(self, provider: VerificationProvider):
        self.provider = provider

    def verify(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
    ) -> VerificationResult:
        return VerificationResult(
            **self.provider.verify(source_bytes, purpose, template, extracted_data)
        )
