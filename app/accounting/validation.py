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
        rows = data.get("rows")

        if isinstance(rows, list) and len(rows) > 1:
            seen_refs = {}
            doc_type = data.get("document_type")
            if doc_type and doc_type in {"VENDOR_INVOICE", "PETTY_CASH_REGISTER"}:
                issues.append(
                    ValidationIssue(
                        field="document_type",
                        severity="WARNING",
                        code="PURPOSE_MISMATCH",
                        message=f"Document appears to be {doc_type}, but is being processed as MEMBER_RECEIPT.",
                    )
                )

            for idx, row in enumerate(rows, start=1):
                if not isinstance(row, dict):
                    issues.append(
                        ValidationIssue(
                            field=f"rows[{idx}]",
                            severity="CRITICAL",
                            code="INVALID_ROW_FORMAT",
                            message=f"Row {idx} is not a valid transaction object.",
                        )
                    )
                    continue

                # Check Amount
                amt_str = row.get("Amount*", "-")
                if amt_str not in {None, "", "-"}:
                    amt = _amount(amt_str)
                    if amt is None:
                        issues.append(
                            ValidationIssue(
                                field=f"rows[{idx}].Amount*",
                                severity="CRITICAL",
                                code="INVALID_AMOUNT",
                                row=idx,
                                current_value=amt_str,
                                message=f"Row {idx}: Amount '{amt_str}' is not a valid number.",
                            )
                        )
                    elif amt < 0:
                        issues.append(
                            ValidationIssue(
                                field=f"rows[{idx}].Amount*",
                                severity="WARNING",
                                code="NEGATIVE_AMOUNT",
                                row=idx,
                                current_value=str(amt),
                                message=f"Row {idx}: Amount {amt} is negative.",
                            )
                        )

                # Check invalid placeholder strings
                for col_name, val in row.items():
                    if isinstance(val, str) and val.strip() in {"null", "None", "NaN", "undefined", "UNKNOWN"}:
                        issues.append(
                            ValidationIssue(
                                field=f"rows[{idx}].{col_name}",
                                severity="WARNING",
                                code="INVALID_PLACEHOLDER",
                                row=idx,
                                current_value=val,
                                message=f"Row {idx}: Field '{col_name}' contains placeholder '{val}'; should be '-'.",
                            )
                        )

                # Check duplicate references
                ref = row.get("Cheque/Ref No*", "-")
                if ref not in {None, "", "-"}:
                    if ref in seen_refs:
                        seen_refs[ref].append(idx)
                    else:
                        seen_refs[ref] = [idx]

            # Report duplicates
            for ref_val, row_indices in seen_refs.items():
                if len(row_indices) > 1:
                    issues.append(
                        ValidationIssue(
                            field="Cheque/Ref No*",
                            severity="WARNING",
                            code="DUPLICATE_REFERENCE",
                            message=f"Reference '{ref_val}' appears in multiple rows: {row_indices}.",
                        )
                    )

            has_critical = any(issue.severity == "CRITICAL" for issue in issues)
            return ValidationResult(status="BLOCKED" if has_critical else "PASSED", issues=issues)

        # Single-record fallback
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
