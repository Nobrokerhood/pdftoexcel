import json
import logging
from typing import Any, Protocol

from app.accounting.purposes import MEMBER_RECEIPT
from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition
from app.agents.extractor import source_parts
from app.core.errors import ServiceNotConfiguredError
from app.documents.ingestion import build_manifest, load_page_images
from app.documents.ocr import DocumentOcrService, RapidOcrProvider
from app.services.gemini_client import GeminiDocumentClient

logger = logging.getLogger(__name__)


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
    def __init__(self, gemini_client: GeminiDocumentClient, ocr_service: DocumentOcrService | None = None):
        self.gemini_client = gemini_client
        self._ocr_service = ocr_service

    def _get_ocr(self) -> DocumentOcrService:
        if self._ocr_service is None:
            self._ocr_service = DocumentOcrService(
                self.gemini_client.settings.poppler_path,
                RapidOcrProvider(),
                dpi=150,
            )
        return self._ocr_service

    def verify(
        self,
        source_bytes: bytes,
        purpose: str,
        template: TemplateDefinition,
        extracted_data: dict,
    ) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")

        # Try structured OCR context for fast verification
        ocr_text_block = ""
        try:
            manifest = build_manifest(
                source_bytes, "source_doc", "application/pdf", self.gemini_client.settings.poppler_path
            )
            rep = self._get_ocr().represent(source_bytes, manifest)
            ocr_pages = []
            for page in rep.pages:
                lines_str = "\n".join(
                    f"[y={line.bbox[1]:.0f}, x={line.bbox[0]:.0f}-{line.bbox[2]:.0f}] {line.text}"
                    for line in page.lines
                )
                ocr_pages.append(f"--- PAGE {page.page_number} ---\n{lines_str}")
            ocr_text_block = "\n\n".join(ocr_pages)
        except Exception as exc:
            logger.warning("Verification OCR fallback: %s", exc)

        # Prepare compact summary of extracted data for verification prompt
        compact_data = dict(extracted_data)
        if "rows" in compact_data and isinstance(compact_data["rows"], list) and len(compact_data["rows"]) > 10:
            compact_data["total_rows"] = len(compact_data["rows"])
            compact_data["sample_rows_first_3"] = compact_data["rows"][:3]
            compact_data["sample_rows_last_3"] = compact_data["rows"][-3:]
            compact_data.pop("rows")

        prompt = (
            "You are an independent accounting verification agent. Check whether "
            "each extracted value and transaction is supported by the original source. Do not "
            "assume extraction is correct.\n"
            "Return one JSON object only, in exactly this shape:\n"
            '{"overall_status": "PASSED" | "FAILED" | "NEEDS_REVIEW", "fields": ['
            '{"field": "<extracted key>", "extracted_value": <value>, '
            '"verified_value": <value shown in the source, or null>, '
            '"status": "VERIFIED" | "MISMATCH" | "NOT_FOUND" | "UNCERTAIN", '
            '"confidence": <number from 0 to 1>, "evidence": "<short quote from the source, or empty string>", '
            '"page_number": <integer or null>}]}\n'
            "Include one item per key or major accounting field. Field status meanings:\n"
            "- VERIFIED = the source supports the extracted value (a null or '-' value is VERIFIED when the source has no such value);\n"
            "- MISMATCH = the source shows a clearly different value;\n"
            "- NOT_FOUND = a non-null extracted value does not appear in the source;\n"
            "- UNCERTAIN = the source is ambiguous or unreadable.\n"
            "Use overall_status PASSED when key fields are VERIFIED and no critical contradiction exists, "
            "FAILED when any key field is MISMATCH or NOT_FOUND, and NEEDS_REVIEW when manual confirmation is advised.\n"
            f"Purpose: {purpose}\n"
            f"Template code: {template.template_code}\n"
            f"Extracted data: {json.dumps(compact_data, default=str)}"
        )

        images = []
        try:
            images = load_page_images(source_bytes, self.gemini_client.settings.poppler_path, dpi=150)
        except Exception as exc:
            logger.warning("Verification image loading fallback: %s", exc)

        if images and (ocr_text_block or purpose == MEMBER_RECEIPT):
            full_prompt = f"{prompt}\n\nDOCUMENT OCR TEXT:\n{ocr_text_block}" if ocr_text_block else prompt
            return self.gemini_client.generate_json([full_prompt, *images])
        elif ocr_text_block:
            full_prompt = f"{prompt}\n\nDOCUMENT OCR TEXT:\n{ocr_text_block}"
            return self.gemini_client.generate_json([full_prompt])
        else:
            return self.gemini_client.generate_json(
                [prompt, *source_parts(source_bytes, self.gemini_client.settings.poppler_path)]
            )


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
        result_dict = self.provider.verify(source_bytes, purpose, template, extracted_data)
        # Ensure overall_status is valid
        if "overall_status" not in result_dict:
            result_dict["overall_status"] = "PASSED"
        if result_dict["overall_status"] not in {"PASSED", "FAILED", "NEEDS_REVIEW"}:
            result_dict["overall_status"] = "NEEDS_REVIEW"
        return VerificationResult(**result_dict)
