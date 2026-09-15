import json
import logging
from typing import Protocol

from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.schemas import CashRegisterExtraction, MemberReceiptExtraction, VendorInvoiceExtraction
from app.accounting.templates import NBH_IMPORT_COLUMNS, TemplateDefinition
from app.core.errors import ServiceNotConfiguredError
from app.documents.ingestion import build_manifest, load_page_images
from app.documents.ocr import DocumentOcrService, RapidOcrProvider
from app.services.gemini_client import GeminiDocumentClient

logger = logging.getLogger(__name__)


class ExtractionProvider(Protocol):
    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        ...


class GeminiExtractionProvider:
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

    def extract(self, source_bytes: bytes, purpose: str, template: TemplateDefinition) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")

        # Step 1: RapidOCR for text & bounding boxes
        ocr_text_block = ""
        is_difficult_or_handwritten = False
        try:
            manifest = build_manifest(
                source_bytes, "source_doc", "application/pdf", self.gemini_client.settings.poppler_path
            )
            rep = self._get_ocr().represent(source_bytes, manifest)
            ocr_pages = []
            all_confs = []
            for page in rep.pages:
                lines_str = "\n".join(
                    f"[y={line.bbox[1]:.0f}, x={line.bbox[0]:.0f}-{line.bbox[2]:.0f}, conf={line.confidence:.2f}] {line.text}"
                    for line in page.lines
                )
                ocr_pages.append(f"--- PAGE {page.page_number} ---\n{lines_str}")
                all_confs.extend(line.confidence for line in page.lines if hasattr(line, "confidence"))
            ocr_text_block = "\n\n".join(ocr_pages)

            # Assess document difficulty / handwriting
            if all_confs:
                mean_conf = sum(all_confs) / len(all_confs)
                low_conf_ratio = sum(1 for c in all_confs if c < 0.80) / len(all_confs)
                if mean_conf < 0.88 or low_conf_ratio > 0.15:
                    is_difficult_or_handwritten = True
        except Exception as exc:
            logger.warning("OCR extraction fallback to images: %s", exc)
            is_difficult_or_handwritten = True

        if purpose == MEMBER_RECEIPT:
            prompt = (
                "You are NoBrokerHood Accounting AI Document Understanding Engine.\n"
                "Carefully inspect the complete document page image and the supporting OCR text.\n\n"
                "DOCUMENT UNDERSTANDING INSTRUCTIONS:\n"
                "1. Inspect the visual page image for all handwritten and printed accounting entries.\n"
                "2. Use OCR text and bounding boxes as supporting evidence, but the visual image is the ultimate truth.\n"
                "3. Distinguish between:\n"
                "   - TRANSACTION: Genuine individual accounting payments/receipts entering the 12 NBH columns.\n"
                "   - INFLOW / RECEIPT: Cash withdrawn or deposits (classify into balance_summary.inflows; do NOT turn into payment transactions).\n"
                "   - GROUPED / SALARY: Rows grouped together (e.g. housekeeping salaries grouped with a bracket). Extract individual staff payments.\n"
                "   - BALANCE / SUMMARY: Opening balance/deficit, carry forward, closing balance, total expenditure, total receipts. Capture in balance_summary; do NOT convert into transactions.\n"
                "   - ANNOTATION / NOTE: Non-transaction notes.\n"
                "4. For each individual transaction, extract the exact 12 NoBrokerHood (NBH) columns:\n"
                "   - 'Payment Type*': 'Cash', 'Cheque', 'Bank Transfer', or '-'\n"
                "   - 'Society Bank Name/Bank code(Given to you by nobrokerhood)*': Bank name/code or '-'\n"
                "   - 'Cheque/Ref No*': Exact voucher number or cheque number or ref number. Read digits carefully from image. Use '-' if missing.\n"
                "   - 'Tower No*': Tower/Wing or '-'\n"
                "   - 'Flat No*': Flat/Unit or '-'\n"
                "   - 'Bill Head*': Primary expense category / bill head written in left column or description (e.g. Office, Clubhouse, Shed (Office), Garden, Salary, Society Electrical, Water, Miscellaneous, Salary Advance, Society). Use '-' if missing.\n"
                "   - 'Amount*': Clean numeric amount (e.g. '900', '140', '10000'). Do NOT include '/-' or currency symbols. Use '-' if missing.\n"
                "   - 'Transaction Date*': Date in DD-MM-YYYY format (e.g. '04-07-2025' or '04-07-25'). Use '-' if missing.\n"
                "   - 'Comments': Complete narration/description from the image.\n"
                "   - 'Meter No': Meter number or '-'\n"
                "   - 'Cheque Issuer Bank': Issuer bank or '-'\n"
                "   - 'Cheque Date': Cheque date or '-'\n"
                "5. HANDWRITING DIGIT PRECISION:\n"
                "   - Carefully inspect handwritten numerals directly from the image.\n"
                "   - Distinguish crossed '7' from '1' (e.g. 750/- vs 150/-).\n"
                "   - Distinguish loop '9' from '7' in voucher numbers (e.g. 290 vs 270).\n"
                "   - Ensure all digits are read up to column borders (e.g. 1789 vs 178).\n"
                "   - Do not merge parenthetical remarks or flat notes (e.g. '(105)') into the amount column (e.g. 2500 vs 1052.5).\n"
                "6. STRICT NO-FABRICATION RULE: Never invent flat numbers, tower numbers, amounts, dates, or reference numbers. Use '-' for any missing field.\n"
                "7. Also extract document-level metadata, balance/summary information, and non-transaction notes so they are preserved for review without polluting the transaction table.\n\n"
                "JSON FORMAT:\n"
                "{\n"
                '  "document_type": "PETTY_CASH_REGISTER" | "MEMBER_RECEIPT" | "BANK_STATEMENT" | "SOCIETY_MEMBER_LEDGER" | "OTHER",\n'
                '  "summary": "Short description of document",\n'
                '  "period": "Document period or -",\n'
                '  "balance_summary": {\n'
                '    "opening_balance": "amount/text or -",\n'
                '    "closing_balance": "amount/text or -",\n'
                '    "total_expenditure": "amount/text or -",\n'
                '    "total_receipts": "amount/text or -",\n'
                '    "inflows": [\n'
                '      {"date": "...", "description": "...", "amount": "...", "ref_no": "..."}\n'
                '    ],\n'
                '    "notes": ["list of balance, bracket or summary notes"]\n'
                '  },\n'
                '  "payment_type": "Payment type or -",\n'
                '  "bank_name_or_code": "Bank name/code or -",\n'
                '  "reference_number": "Ref / UTR / Cheque no or -",\n'
                '  "tower": "Wing / Tower or -",\n'
                '  "flat": "Unit / Flat no or -",\n'
                '  "bill_head": "Bill head / charge category or -",\n'
                '  "amount": "Numeric amount or -",\n'
                '  "transaction_date": "DD-MM-YYYY or -",\n'
                '  "comments": "Narration / description or -",\n'
                '  "meter_number": "Meter no or -",\n'
                '  "cheque_issuer_bank": "Issuer bank or -",\n'
                '  "cheque_date": "DD-MM-YYYY or -",\n'
                '  "rows": [\n'
                "    {\n"
                '      "Payment Type*": "...",\n'
                '      "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "...",\n'
                '      "Cheque/Ref No*": "...",\n'
                '      "Tower No*": "...",\n'
                '      "Flat No*": "...",\n'
                '      "Bill Head*": "...",\n'
                '      "Amount*": "...",\n'
                '      "Transaction Date*": "...",\n'
                '      "Comments": "...",\n'
                '      "Meter No": "...",\n'
                '      "Cheque Issuer Bank": "...",\n'
                '      "Cheque Date": "..."\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
            )
        else:
            prompt = (
                "You are an accounting document extraction agent. Extract only values "
                "directly supported by the source. Use null for missing values. "
                "Do not invent values. Return one JSON object only.\n"
                f"Purpose: {purpose}\n"
                f"Template code: {template.template_code}\n"
                f"Canonical fields: {', '.join(template.fields)}\n"
                f"JSON schema keys: {self._schema_keys(purpose)}"
            )

        # Intelligent Multimodal Routing
        images = []
        try:
            images = load_page_images(source_bytes, self.gemini_client.settings.poppler_path, dpi=180)
        except Exception as exc:
            logger.warning("Could not load page images for multimodal extraction: %s", exc)

        if images and (is_difficult_or_handwritten or purpose == MEMBER_RECEIPT or not ocr_text_block):
            full_prompt = f"{prompt}\nSUPPORTING OCR TEXT & BOUNDING BOXES:\n{ocr_text_block}" if ocr_text_block else prompt
            return self.gemini_client.generate_json([full_prompt, *images])
        elif ocr_text_block:
            full_prompt = f"{prompt}\nDOCUMENT OCR TEXT:\n{ocr_text_block}"
            return self.gemini_client.generate_json([full_prompt])
        else:
            parts = [prompt, *source_parts(source_bytes, self.gemini_client.settings.poppler_path)]
            return self.gemini_client.generate_json(parts)

    @staticmethod
    def _schema_keys(purpose: str) -> str:
        if purpose == MEMBER_RECEIPT:
            return (
                "payment_type, bank_name_or_code, reference_number, tower, flat, "
                "bill_head, amount, transaction_date, comments, meter_number, "
                "cheque_issuer_bank, cheque_date, rows[]"
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
        return validate_extraction(purpose, self.provider.extract(source_bytes, purpose, template))


def _clean_dash(val):
    if val is None or str(val).strip() in {"", "-", "null", "None", "N/A", "UNKNOWN"}:
        return "-"
    return str(val).strip()


def validate_extraction(purpose: str, result: dict) -> dict:
    if purpose == MEMBER_RECEIPT:
        rows = result.get("rows")
        if isinstance(rows, list) and len(rows) > 0:
            # Ensure each row has the 12 columns with '-' for missing
            normalized_rows = []
            for r in rows:
                if isinstance(r, dict):
                    row_dict = {}
                    for col in NBH_IMPORT_COLUMNS:
                        val = r.get(col)
                        if val is None:
                            # Try camel_case fallback
                            col_clean = col.rstrip("*").replace(" ", "_").lower()
                            val = r.get(col_clean, "-")
                        row_dict[col] = _clean_dash(val)
                    normalized_rows.append(row_dict)
            result["rows"] = normalized_rows
            # If top-level fields are missing or dash, populate from first row
            first = normalized_rows[0]
            if not result.get("amount") or result.get("amount") == "-":
                result["amount"] = first.get("Amount*", "-")
            if not result.get("tower") or result.get("tower") == "-":
                result["tower"] = first.get("Tower No*", "-")
            if not result.get("flat") or result.get("flat") == "-":
                result["flat"] = first.get("Flat No*", "-")
            if not result.get("bill_head") or result.get("bill_head") == "-":
                result["bill_head"] = first.get("Bill Head*", "-")
            if not result.get("payment_type") or result.get("payment_type") == "-":
                result["payment_type"] = first.get("Payment Type*", "-")
            if not result.get("transaction_date") or result.get("transaction_date") == "-":
                result["transaction_date"] = first.get("Transaction Date*", "-")
            if not result.get("reference_number") or result.get("reference_number") == "-":
                result["reference_number"] = first.get("Cheque/Ref No*", "-")
            if not result.get("bank_name_or_code") or result.get("bank_name_or_code") == "-":
                result["bank_name_or_code"] = first.get("Society Bank Name/Bank code(Given to you by nobrokerhood)*", "-")
            if not result.get("comments") or result.get("comments") == "-":
                result["comments"] = first.get("Comments", "-")
        else:
            # Single-record: build 1-item rows
            single_row = {
                "Payment Type*": _clean_dash(result.get("payment_type")),
                "Society Bank Name/Bank code(Given to you by nobrokerhood)*": _clean_dash(result.get("bank_name_or_code")),
                "Cheque/Ref No*": _clean_dash(result.get("reference_number")),
                "Tower No*": _clean_dash(result.get("tower")),
                "Flat No*": _clean_dash(result.get("flat")),
                "Bill Head*": _clean_dash(result.get("bill_head")),
                "Amount*": _clean_dash(result.get("amount")),
                "Transaction Date*": _clean_dash(result.get("transaction_date")),
                "Comments": _clean_dash(result.get("comments")),
                "Meter No": _clean_dash(result.get("meter_number")),
                "Cheque Issuer Bank": _clean_dash(result.get("cheque_issuer_bank")),
                "Cheque Date": _clean_dash(result.get("cheque_date")),
            }
            result["rows"] = [single_row]

        # Attach generalized accounting reconciliation
        try:
            from app.accounting.reconciliation import AccountingReconciliationService
            recon_service = AccountingReconciliationService()
            result["reconciliation"] = recon_service.reconcile(result)
        except Exception as exc:
            logger.warning("Reconciliation computation failed: %s", exc)

        return MemberReceiptExtraction(**result).model_dump(mode="json")
    if purpose == VENDOR_INVOICE:
        return VendorInvoiceExtraction(**result).model_dump(mode="json")
    if purpose == PETTY_CASH_REGISTER:
        return CashRegisterExtraction(**result).model_dump(mode="json")
    raise ValueError("Unsupported purpose.")


def source_parts(source_bytes: bytes, poppler_path: str | None = None):
    return load_page_images(source_bytes, poppler_path, dpi=120)
