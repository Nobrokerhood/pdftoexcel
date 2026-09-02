from decimal import Decimal, InvalidOperation

from app.accounting.purposes import MEMBER_RECEIPT, VENDOR_INVOICE
from app.accounting.schemas import ValidationIssue, ValidationResult


def _amount(value) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


class AccountingValidationService:
    def validate(self, purpose: str, data: dict) -> ValidationResult:
        purpose = purpose.upper()
        if purpose == MEMBER_RECEIPT:
            return self._member_receipt(data)
        if purpose == VENDOR_INVOICE:
            return self._vendor_invoice(data)
        return ValidationResult(
            status="BLOCKED",
            issues=[
                ValidationIssue(
                    field="purpose",
                    severity="CRITICAL",
                    message="Unsupported purpose.",
                )
            ],
        )

    def _member_receipt(self, data: dict) -> ValidationResult:
        issues = []
        amount = _amount(data.get("amount"))
        if amount is None:
            issues.append(ValidationIssue(field="amount", severity="CRITICAL", message="Payment amount is required."))
        elif amount <= 0:
            issues.append(ValidationIssue(field="amount", severity="CRITICAL", message="Payment amount must be greater than zero."))
        if not data.get("transaction_date"):
            issues.append(ValidationIssue(field="transaction_date", severity="CRITICAL", message="Transaction date is required."))
        if not data.get("tower"):
            issues.append(ValidationIssue(field="tower", severity="CRITICAL", message="Tower is required."))
        if not data.get("flat"):
            issues.append(ValidationIssue(field="flat", severity="CRITICAL", message="Flat is required."))
        if not data.get("reference_number"):
            issues.append(ValidationIssue(field="reference_number", severity="CRITICAL", message="Reference number is required."))
        if not data.get("bank_name_or_code"):
            issues.append(ValidationIssue(field="bank_name_or_code", severity="CRITICAL", message="Bank code or mapping is required."))
        if not data.get("bill_head"):
            issues.append(ValidationIssue(field="bill_head", severity="CRITICAL", message="Bill head is required."))
        return ValidationResult(status="BLOCKED" if issues else "PASSED", issues=issues)

    def _vendor_invoice(self, data: dict) -> ValidationResult:
        issues = []
        if not data.get("bill_number"):
            issues.append(ValidationIssue(field="bill_number", severity="CRITICAL", message="Bill number is required."))
        if not data.get("bill_date"):
            issues.append(ValidationIssue(field="bill_date", severity="CRITICAL", message="Bill date is required."))
        if not data.get("vendor_code"):
            issues.append(ValidationIssue(field="vendor_code", severity="CRITICAL", message="Vendor code or mapping is required."))

        expenses = data.get("expenses") or []
        if not expenses:
            issues.append(ValidationIssue(field="expenses", severity="CRITICAL", message="At least one expense entry is required."))
        for index, expense in enumerate(expenses, start=1):
            if not expense.get("expense_code"):
                issues.append(ValidationIssue(field=f"expenses[{index}].expense_code", severity="CRITICAL", message="Expense code or mapping is required."))
            amount = _amount(expense.get("expense_amount"))
            if amount is None or amount <= 0:
                issues.append(ValidationIssue(field=f"expenses[{index}].expense_amount", severity="CRITICAL", message="Expense amount must be greater than zero."))

        for field in ["cgst_amount", "sgst_amount", "igst_amount", "tds_amount"]:
            amount = _amount(data.get(field) or 0)
            if amount is not None and amount < 0:
                issues.append(ValidationIssue(field=field, severity="CRITICAL", message="Tax amounts must be non-negative."))

        return ValidationResult(status="BLOCKED" if issues else "PASSED", issues=issues)
