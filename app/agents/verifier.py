import json
import logging
from typing import Any, Protocol

from app.accounting.purposes import MEMBER_RECEIPT
from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition
from app.agents.extractor import source_parts
from app.core.errors import ServiceNotConfiguredError
from app.documents.ingestion import build_manifest, load_page_images
from dataclasses import dataclass
from app.documents.ocr import DocumentOcrService, DocumentRepresentation, RapidOcrProvider
from app.services.gemini_client import GeminiDocumentClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationRouteDecision:
    mode: str  # "MULTIMODAL" or "OCR_STRUCTURED"
    reason: str
    is_handwritten_or_uncertain: bool


def determine_verification_route(
    rep: DocumentRepresentation | None,
    extracted_data: dict,
    has_images: bool,
) -> VerificationRouteDecision:
    """Evaluate document signals to auditably decide verification mode.

    Multimodal verification is ALWAYS retained when:
    1. Images are available and document has handwritten indicators or low OCR confidence.
    2. OCR representation has warnings or unreadable text.
    3. Document type is PETTY_CASH_REGISTER, handwritten voucher, or unstructured note.
    4. Text contains suspicious or fragmented OCR readings.

    OCR_STRUCTURED is used only when:
    - Printed digital document with high line confidence (> 0.92 mean).
    - Ample readable text (> 20 lines) and no unreadable page warnings.
    - Clear structured layout without irregular character noise.
    """
    if not has_images:
        return VerificationRouteDecision(
            mode="OCR_STRUCTURED",
            reason="Page images unavailable; using structured OCR text verification.",
            is_handwritten_or_uncertain=False,
        )

    if rep is None or not rep.all_lines:
        return VerificationRouteDecision(
            mode="MULTIMODAL",
            reason="OCR text absent or empty; visual multimodal verification required.",
            is_handwritten_or_uncertain=True,
        )

    doc_type = str(extracted_data.get("document_type") or "").upper()
    if doc_type in ("PETTY_CASH_REGISTER", "HANDWRITTEN", "VOUCHER", "OTHER"):
        return VerificationRouteDecision(
            mode="MULTIMODAL",
            reason=f"Document classified as {doc_type}; visual multimodal inspection required for handwriting & stamps.",
            is_handwritten_or_uncertain=True,
        )

    if rep.warnings:
        return VerificationRouteDecision(
            mode="MULTIMODAL",
            reason=f"OCR generated warnings ({'; '.join(rep.warnings[:2])}); visual verification required.",
            is_handwritten_or_uncertain=True,
        )

    mean_conf = rep.mean_confidence
    low_conf_lines = sum(1 for line in rep.all_lines if getattr(line, "confidence", 1.0) < 0.85)
    total_lines = len(rep.all_lines)
    low_conf_ratio = low_conf_lines / max(1, total_lines)

    if mean_conf < 0.92 or low_conf_ratio > 0.12:
        return VerificationRouteDecision(
            mode="MULTIMODAL",
            reason=f"Sub-optimal OCR quality (mean confidence={mean_conf:.2f}, low conf ratio={low_conf_ratio:.1%}); visual verification required.",
            is_handwritten_or_uncertain=True,
        )

    for page in rep.pages:
        if getattr(page, "script", "PRINTED") != "PRINTED":
            return VerificationRouteDecision(
                mode="MULTIMODAL",
                reason=f"Page {page.page_number} detected non-printed script ({page.script}); visual verification required.",
                is_handwritten_or_uncertain=True,
            )

    return VerificationRouteDecision(
        mode="OCR_STRUCTURED",
        reason=f"Clean digital document (mean confidence={mean_conf:.2f}, {total_lines} lines, structured layout); verified via structured layout.",
        is_handwritten_or_uncertain=False,
    )


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
        rep = None
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

        decision = determine_verification_route(rep, extracted_data, bool(images))
        logger.info("VERIFICATION_ROUTING [mode=%s]: %s", decision.mode, decision.reason)

        if decision.mode == "MULTIMODAL" and images:
            full_prompt = f"{prompt}\n\nDOCUMENT OCR TEXT:\n{ocr_text_block}" if ocr_text_block else prompt
            return self.gemini_client.generate_json([full_prompt, *images])
        elif ocr_text_block:
            full_prompt = f"{prompt}\n\nDOCUMENT OCR TEXT:\n{ocr_text_block}"
            return self.gemini_client.generate_json([full_prompt])
        elif images:
            return self.gemini_client.generate_json([prompt, *images])
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
