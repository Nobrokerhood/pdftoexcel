import json
from typing import Protocol

from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition
from app.agents.extractor import GeminiExtractionProvider, source_parts, validate_extraction
from app.core.errors import ServiceNotConfiguredError
from app.services.gemini_client import GeminiDocumentClient


class RepairProvider(Protocol):
    def repair(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
        verification_result: VerificationResult,
    ) -> dict:
        ...


class GeminiRepairProvider:
    def __init__(self, gemini_client: GeminiDocumentClient):
        self.gemini_client = gemini_client

    def repair(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
        verification_result: VerificationResult,
    ) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")
        prompt = (
            "You are an extraction repair agent. Correct only fields called out "
            "by verification mismatches, not the whole record. Use null for "
            "unsupported values. Return the complete corrected extraction as one JSON "
            "object only, with exactly these keys: "
            f"{GeminiExtractionProvider._schema_keys(purpose)}\n"
            f"Purpose: {purpose}\n"
            f"Template code: {template.template_code}\n"
            f"Previous extraction: {json.dumps(extracted_data, default=str)}\n"
            f"Verification result: {json.dumps(verification_result.model_dump(mode='json'), default=str)}"
        )
        return self.gemini_client.generate_json(
            [prompt, *source_parts(source_bytes, self.gemini_client.settings.poppler_path)]
        )


class RepairAgent:
    def __init__(self, provider: RepairProvider):
        self.provider = provider

    def repair(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
        verification_result: VerificationResult,
    ) -> dict:
        return validate_extraction(
            purpose,
            self.provider.repair(
                source_bytes, purpose, template, extracted_data, verification_result
            ),
        )
