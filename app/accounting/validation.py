"""Deterministic validation and review-item derivation.

Validation reads the CURRENT document state (rows, ledger, reconciliation) and
derives the review items from it, so a reviewer's resolution (confirming a row,
fixing an amount, dismissing a candidate) is reflected by re-running
validation. Nothing here changes a value.

CRITICAL issues block approval; WARNING issues are shown but never block.
An unmapped Bill Head / bank / vendor is a WARNING: the source value is exported.
"""

from collections import defaultdict

from app.accounting.dates import parse_date
from app.accounting.money import parse_amount
from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.schemas import ValidationIssue, ValidationResult

CONFIRMED_ROW_STATES = ("ACCEPTED", "USER_CONFIRMED")
SUPPORTED = (MEMBER_RECEIPT, VENDOR_INVOICE, PETTY_CASH_REGISTER)


class AccountingValidationService:
    def validate(self, purpose: str, data: dict) -> ValidationResult:
        purpose = (purpose or "").upper()
        if purpose not in SUPPORTED:
            return ValidationResult(status="BLOCKED", issues=[ValidationIssue(
                field="purpose", severity="CRITICAL", code="UNSUPPORTED_PURPOSE", message="Unsupported purpose.")])
        issues: list[ValidationIssue] = []
        rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
        evidence = data.get("row_evidence") or {}

        if not rows:
            issues.append(ValidationIssue(
                field="rows", severity="CRITICAL", code="NO_TRANSACTIONS",
                message=data.get("extraction_notice") or
                "No transaction rows were extracted. A workbook cannot be approved without transactions; "
                "review the source document and the candidate ledger.",
                action="Add the transactions manually or reject the document."))

        doc_type = str(data.get("document_type") or "").upper()
        mismatched = {
            MEMBER_RECEIPT: {"VENDOR_INVOICE", "TAX_INVOICE", "CASH_MEMO", "LABOUR_BILL", "PETTY_CASH_REGISTER"},
            VENDOR_INVOICE: {"PETTY_CASH_REGISTER", "BANK_STATEMENT", "MEMBER_RECEIPT"},
            PETTY_CASH_REGISTER: {"VENDOR_INVOICE", "TAX_INVOICE", "BANK_STATEMENT"},
        }.get(purpose, set())
        if doc_type in mismatched:
            issues.append(ValidationIssue(
                field="document_type", severity="WARNING", code="PURPOSE_MISMATCH",
                message=f"Document appears to be {doc_type}, but is being processed as {purpose}."))

        refs = defaultdict(list)
        for idx, row in enumerate(rows, start=1):
            row_id = row.get("_row_id") or f"row{idx}"
            amount = parse_amount(row.get("Amount*"))
            if not amount.found:
                issues.append(ValidationIssue(
                    field=f"{row_id}.Amount*", row=idx, row_id=row_id, severity="CRITICAL",
                    code="AMOUNT_MISSING" if amount.status == "MISSING" else "AMOUNT_INVALID",
                    current_value=row.get("Amount*"),
                    message=f"Row {row_id}: amount {'is missing' if amount.status == 'MISSING' else 'is not a valid amount'}"
                            f" ({amount.reason}).",
                    suggested_value=", ".join(amount.candidates) or None,
                    action="Enter the amount shown on the source document."))
            elif amount.value <= 0:
                issues.append(ValidationIssue(
                    field=f"{row_id}.Amount*", row=idx, row_id=row_id, severity="WARNING",
                    code="NON_POSITIVE_AMOUNT", current_value=row.get("Amount*"),
                    message=f"Row {row_id}: amount {amount.value} is not positive."))
            date = parse_date(row.get("Transaction Date*"))
            if not date.found:
                issues.append(ValidationIssue(
                    field=f"{row_id}.Transaction Date*", row=idx, row_id=row_id, severity="CRITICAL",
                    code="DATE_MISSING" if date.status == "MISSING" else "DATE_INVALID",
                    current_value=row.get("Transaction Date*"),
                    message=f"Row {row_id}: transaction date {'is missing' if date.status == 'MISSING' else 'is invalid'}"
                            f" ({date.reason}).",
                    action="Enter the date as DD-MM-YYYY."))
            cheque_date = row.get("Cheque Date", "-")
            if cheque_date not in (None, "", "-") and not parse_date(cheque_date).found:
                issues.append(ValidationIssue(
                    field=f"{row_id}.Cheque Date", row=idx, row_id=row_id, severity="CRITICAL",
                    code="CHEQUE_DATE_INVALID", current_value=cheque_date,
                    message=f"Row {row_id}: cheque date '{cheque_date}' is not a valid date; use DD-MM-YYYY or '-'."))
            status = row.get("_status") or "UNVERIFIED"
            if status not in CONFIRMED_ROW_STATES:
                reasons = (evidence.get(row_id) or {}).get("reasons") or []
                issues.append(ValidationIssue(
                    field=row_id, row=idx, row_id=row_id, severity="CRITICAL", code="ROW_NEEDS_REVIEW",
                    message=f"Row {row_id} needs review: " + ("; ".join(reasons) if reasons else
                                                              "not independently verified against the source."),
                    evidence="; ".join(reasons),
                    action="Check the row against the source, correct it if needed, then confirm it."))
            ref = row.get("Cheque/Ref No*", "-")
            if ref not in (None, "", "-"):
                refs[ref].append(row_id)

        for ref, ids in refs.items():
            if len(ids) > 1:
                issues.append(ValidationIssue(
                    field="Cheque/Ref No*", severity="WARNING", code="DUPLICATE_REFERENCE",
                    message=f"Reference '{ref}' appears on rows {', '.join(ids)}."))

        ledger = data.get("candidate_ledger") or {}
        if ledger and not ledger.get("balanced", True):
            issues.append(ValidationIssue(
                field="candidate_ledger", severity="CRITICAL", code="LEDGER_UNBALANCED",
                message=f"Source candidate ledger does not balance: {ledger.get('equation')}; "
                        f"unaccounted: {', '.join(ledger.get('unaccounted') or [])}."))
        for cand in ledger.get("candidates") or []:
            if cand.get("status") == "UNRESOLVED":
                issues.append(ValidationIssue(
                    field=cand.get("candidate_id", ""), severity="CRITICAL", code="UNRESOLVED_SOURCE_ROW",
                    row_id=cand.get("candidate_id"),
                    message=f"Source region {cand.get('candidate_id')} (page {cand.get('page')}) looks like a "
                            f"transaction row that extraction did not report: {cand.get('status_reason')}",
                    action="Add it as a row or dismiss it with a reason."))

        for check in (data.get("reconciliation") or {}).get("checks") or []:
            if check.get("status") in ("DISCREPANCY", "INCOMPLETE"):
                issues.append(ValidationIssue(
                    field=f"reconciliation.{check.get('check_id')}", severity="WARNING",
                    code="RECONCILIATION_DISCREPANCY",
                    current_value=check.get("source_value"), suggested_value=check.get("calculated_value"),
                    message=f"{check.get('label')}: source {check.get('source_value')}, calculated "
                            f"{check.get('calculated_value')}, difference {check.get('difference')}. "
                            f"Kept as written in the source."))

        for item in data.get("mapping_missing") or []:
            issues.append(ValidationIssue(
                field=str(item.get("type")), severity="WARNING", code="MAPPING_MISSING",
                current_value=item.get("source_value"),
                message=f"No Mapping_Master entry for {item.get('type')} '{item.get('source_value')}'; "
                        f"the source value will be exported."))

        if data.get("extraction_outcome") == "EXTRACTION_CONTRACT_VIOLATION" or (
                data.get("_extraction_provider") == "LOCAL_OCR" and data.get("provider_note")):
            issues.append(ValidationIssue(
                field="extraction", severity="WARNING", code="DEGRADED_EXTRACTION",
                message=f"AI extraction was not used: {data.get('provider_note') or 'contract violation'}. "
                        f"Rows come from OCR evidence only."))

        resolved = data.get("review_resolutions") or {}
        blocking = [i for i in issues if i.severity == "CRITICAL" and not self._resolved(i, resolved)]
        for issue in issues:
            key = issue_key(issue)
            if key in resolved:
                issue.resolution = resolved[key]
        return ValidationResult(status="BLOCKED" if blocking else "PASSED", issues=issues)

    @staticmethod
    def _resolved(issue: ValidationIssue, resolved: dict) -> bool:
        # Field errors (invalid amount/date) are resolved only by fixing the value.
        if issue.code in ("ROW_NEEDS_REVIEW", "UNRESOLVED_SOURCE_ROW"):
            return issue_key(issue) in resolved
        return False


def issue_key(issue: ValidationIssue) -> str:
    return f"{issue.code}:{issue.row_id or issue.field}"
