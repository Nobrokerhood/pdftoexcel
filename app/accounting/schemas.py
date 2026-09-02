from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field


Purpose = Literal["MEMBER_RECEIPT", "VENDOR_INVOICE"]


class MemberReceiptExtraction(BaseModel):
    payment_type: str | None = None
    bank_name_or_code: str | None = None
    reference_number: str | None = None
    tower: str | None = None
    flat: str | None = None
    bill_head: str | None = None
    amount: Decimal | None = None
    transaction_date: str | None = None
    comments: str | None = None
    meter_number: str | None = None
    cheque_issuer_bank: str | None = None
    cheque_date: str | None = None


class VendorExpense(BaseModel):
    expense_code: str | None = None
    expense_description: str | None = None
    expense_amount: Decimal | None = None


class VendorInvoiceExtraction(BaseModel):
    bill_number: str | None = None
    bill_date: str | None = None
    vendor_code: str | None = None
    vendor_name: str | None = None
    due_date: str | None = None
    narration: str | None = None
    cgst_amount: Decimal | None = Decimal("0")
    sgst_amount: Decimal | None = Decimal("0")
    igst_amount: Decimal | None = Decimal("0")
    tds_amount: Decimal | None = Decimal("0")
    expenses: list[VendorExpense] = Field(default_factory=list)


class VerificationFieldResult(BaseModel):
    field: str
    extracted_value: Any = None
    verified_value: Any = None
    status: Literal["VERIFIED", "MISMATCH", "NOT_FOUND", "UNCERTAIN"]
    confidence: float = 0
    evidence: str = ""
    page_number: int | None = None


class VerificationResult(BaseModel):
    overall_status: Literal["PASSED", "FAILED", "NEEDS_REVIEW"]
    fields: list[VerificationFieldResult] = Field(default_factory=list)


class MappingMissingItem(BaseModel):
    type: str
    source_value: str


class MappingResult(BaseModel):
    status: Literal["MAPPED", "NEEDS_MAPPING"]
    mapped_data: dict[str, Any] = Field(default_factory=dict)
    missing: list[MappingMissingItem] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    field: str
    severity: Literal["CRITICAL", "WARNING"]
    message: str


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
