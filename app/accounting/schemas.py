import re
from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field


Purpose = Literal["MEMBER_RECEIPT", "VENDOR_INVOICE"]

_CURRENCY = re.compile(r"₹|(?<![A-Za-z])(?:INR|Rs\.?)(?![A-Za-z])", re.IGNORECASE)
# Digits with optional thousands (12,450) or Indian lakh (1,00,000) grouping and decimals.
_GROUPED_NUMBER = re.compile(r"-?(?:\d+|\d{1,3}(?:,\d{2,3})+)(?:\.\d+)?")


def _normalize_amount(value: Any) -> Any:
    # Documents print amounts as "12,450.75" or "Rs. 1,00,000"; strip currency and
    # grouping so they parse. Anything else (e.g. "12.450,75") is passed through
    # unchanged so Decimal rejects it rather than guessing.
    if not isinstance(value, str):
        return value
    cleaned = _CURRENCY.sub("", value).replace(" ", "").strip()
    if cleaned in {"", "-", "--", "N/A", "NA", "null", "None"}:
        return None
    return cleaned.replace(",", "") if _GROUPED_NUMBER.fullmatch(cleaned) else value


Amount = Annotated[Decimal | None, BeforeValidator(_normalize_amount)]


class MemberReceiptExtraction(BaseModel):
    payment_type: str | None = None
    bank_name_or_code: str | None = None
    reference_number: str | None = None
    tower: str | None = None
    flat: str | None = None
    bill_head: str | None = None
    amount: Amount = None
    transaction_date: str | None = None
    comments: str | None = None
    meter_number: str | None = None
    cheque_issuer_bank: str | None = None
    cheque_date: str | None = None
    document_type: str | None = None
    summary: str | None = None
    balance_summary: dict[str, Any] | None = None
    reconciliation: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)


class VendorExpense(BaseModel):
    expense_code: str | None = None
    expense_description: str | None = None
    expense_amount: Amount = None


class VendorInvoiceExtraction(BaseModel):
    bill_number: str | None = None
    bill_date: str | None = None
    vendor_code: str | None = None
    vendor_name: str | None = None
    due_date: str | None = None
    narration: str | None = None
    cgst_amount: Amount = Decimal("0")
    sgst_amount: Amount = Decimal("0")
    igst_amount: Amount = Decimal("0")
    tds_amount: Amount = Decimal("0")
    expenses: list[VendorExpense] = Field(default_factory=list)


class CashRegisterRow(BaseModel):
    source_page: int | None = None
    source_row: int | None = None
    serial_no: str | None = None
    voucher_no: str | None = None
    date: str | None = None
    particulars: str | None = None
    category: str | None = None
    # PAYMENT = money out of the cash register, RECEIPT = cash received into it.
    debit_credit: Literal["PAYMENT", "RECEIPT"] | None = None
    amount: Amount = None
    amount_status: Literal["FOUND", "MISSING", "AMBIGUOUS"] = "FOUND"
    amount_candidates: list[str] = Field(default_factory=list)
    running_balance: Amount = None
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = "LOW"
    expense_code: str | None = None


class CashRegisterExtraction(BaseModel):
    register_period: str | None = None
    opening_balance: Amount = None
    closing_balance: Amount = None
    written_payment_total: Amount = None
    written_receipt_total: Amount = None
    rows: list[CashRegisterRow] = Field(default_factory=list)


class VerificationFieldResult(BaseModel):
    field: str
    extracted_value: Any = None
    verified_value: Any = None
    status: Literal["VERIFIED", "MISMATCH", "NOT_FOUND", "UNCERTAIN"]
    confidence: float = 0
    # Absent fields have nothing to quote; models return null for them.
    evidence: Annotated[str, BeforeValidator(lambda value: "" if value is None else value)] = ""
    page_number: int | None = None


class VerificationResult(BaseModel):
    overall_status: Literal["PASSED", "FAILED", "NEEDS_REVIEW"]
    fields: list[VerificationFieldResult] = Field(default_factory=list)


class MappingMissingItem(BaseModel):
    type: str
    source_value: str
    field: str = ""
    suggested_category: str | None = None
    reason: str = ""
    rows: list[int] = Field(default_factory=list)


class MappingResult(BaseModel):
    status: Literal["MAPPED", "NEEDS_MAPPING"]
    mapped_data: dict[str, Any] = Field(default_factory=dict)
    missing: list[MappingMissingItem] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    field: str
    severity: Literal["CRITICAL", "WARNING"]
    message: str
    code: str = ""
    rule_id: str = ""
    row: int | None = None
    evidence: str = ""
    current_value: Any = None
    suggested_value: Any = None
    action: str = ""


class ValidationResult(BaseModel):
    status: Literal["PASSED", "BLOCKED"]
    issues: list[ValidationIssue] = Field(default_factory=list)


class HumanCorrection(BaseModel):
    field: str
    old_value: Any = None
    new_value: Any = None
    user_email: str
    timestamp: str


class JobSummary(BaseModel):
    job_id: str
    purpose: str
    template_code: str
    source_filename: str
    overall_status: str
    current_step: str
    human_status: str
