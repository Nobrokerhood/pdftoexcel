import gc
import io
from typing import Protocol

from pdf2image import convert_from_bytes
from PIL import Image

from app.accounting.purposes import MEMBER_RECEIPT, VENDOR_INVOICE
from app.accounting.schemas import MemberReceiptExtraction, VendorInvoiceExtraction
from app.accounting.templates import TemplateDefinition
from app.core.errors import ServiceNotConfiguredError
from app.services.gemini_client import GeminiDocumentClient


class ExtractionProvider(Protocol):
    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        ...


class GeminiExtractionProvider:
    def __init__(self, gemini_client: GeminiDocumentClient):
        self.gemini_client = gemini_client

    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")
        prompt = (
            "You are an accounting document extraction agent. Extract only values "
            "directly supported by the source. Use null for missing values. "
            "Do not invent values. Return one JSON object only.\n"
            f"Purpose: {purpose}\n"
            f"Template code: {template.template_code}\n"
            f"Canonical fields: {', '.join(template.fields)}\n"
            f"JSON schema keys: {self._schema_keys(purpose)}"
        )
        parts = [prompt, *source_parts(source_bytes)]
        return self.gemini_client.generate_json(parts)

    @staticmethod
    def _schema_keys(purpose: str) -> str:
        if purpose == MEMBER_RECEIPT:
            return (
                "payment_type, bank_name_or_code, reference_number, tower, flat, "
                "bill_head, amount, transaction_date, comments, meter_number, "
                "cheque_issuer_bank, cheque_date"
            )
        return (
            "bill_number, bill_date, vendor_code, vendor_name, due_date, narration, "
            "cgst_amount, sgst_amount, igst_amount, tds_amount, expenses[] with "
            "expense_code, expense_description, expense_amount"
        )


class ExtractionAgent:
    def __init__(self, provider: ExtractionProvider):
        self.provider = provider

    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        result = self.provider.extract(source_bytes, purpose, template)
        if purpose == MEMBER_RECEIPT:
            return MemberReceiptExtraction(**result).model_dump(mode="json")
        if purpose == VENDOR_INVOICE:
            return VendorInvoiceExtraction(**result).model_dump(mode="json")
        raise ValueError("Unsupported purpose.")


def source_parts(source_bytes: bytes):
    if source_bytes.startswith(b"%PDF"):
        images = []
        try:
            for page in convert_from_bytes(source_bytes, dpi=120, fmt="jpeg"):
                images.append(page.convert("RGB"))
                page.close()
            return images
        finally:
            gc.collect()
    return [Image.open(io.BytesIO(source_bytes)).convert("RGB")]
