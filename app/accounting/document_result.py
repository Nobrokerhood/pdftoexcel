"""Final shape of an extraction result, whatever produced it.

The production pipeline (`DocumentExtractionPipeline`) already emits rows with
immutable `_row_id`s, field evidence and a candidate ledger. Other providers
(tests, legacy adapters) may return the older shapes; they are converted here
with explicit rules, never by rebuilding a transaction out of document-level
fields when no rows exist.

Row metadata keys (never exported to the NBH sheet):
  _row_id        immutable identity; positions are never used as identity
  _status        ACCEPTED | NEEDS_REVIEW | UNVERIFIED | USER_CONFIRMED
  _edited_fields columns a reviewer changed (immutable to every automated stage)
  USER_EDITED    True when any field was edited (kept for compatibility)
"""

import logging
from typing import Any

from app.accounting.money import format_amount, parse_amount
from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.reconciliation import AccountingReconciliationService
from app.accounting.templates import NBH_IMPORT_COLUMNS

logger = logging.getLogger(__name__)

META_KEYS = ("_row_id", "_status", "_edited_fields", "USER_EDITED")
PLACEHOLDERS = {"", "null", "none", "n/a", "unknown", "undefined", "nan"}

_SYNONYMS = {
    "Payment Type*": ("payment_type", "Payment Type", "payment_mode"),
    "Society Bank Name/Bank code(Given to you by nobrokerhood)*": ("bank_name_or_code", "Society Bank Name/Bank code", "bank_code", "bank_name"),
    "Cheque/Ref No*": ("reference_number", "Cheque/Ref No", "ref_no", "cheque_no", "voucher_no"),
    "Tower No*": ("tower", "Tower No"),
    "Flat No*": ("flat", "Flat No", "flat_no"),
    "Bill Head*": ("bill_head", "Bill Head", "category"),
    "Amount*": ("amount", "Amount", "expense_amount"),
    "Transaction Date*": ("transaction_date", "Transaction Date", "date"),
    "Comments": ("comments", "narration", "particulars", "description", "expense_description"),
    "Meter No": ("meter_number", "meter_no"),
    "Cheque Issuer Bank": ("cheque_issuer_bank", "issuer_bank"),
    "Cheque Date": ("cheque_date",),
}


def clean_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return "-" if text.lower() in PLACEHOLDERS else text


def normalize_row(raw: dict, fallback_id: str) -> dict:
    row = {}
    for col in NBH_IMPORT_COLUMNS:
        value = raw.get(col)
        if value is None:
            for alt in _SYNONYMS.get(col, ()):
                if raw.get(alt) is not None:
                    value = raw.get(alt)
                    break
        row[col] = clean_cell(value)
    row["_row_id"] = str(raw.get("_row_id") or fallback_id)
    row["_status"] = str(raw.get("_status") or "UNVERIFIED")
    edited = raw.get("_edited_fields") or []
    if raw.get("USER_EDITED") and not edited:
        edited = [c for c in NBH_IMPORT_COLUMNS]
    row["_edited_fields"] = [c for c in edited if c in NBH_IMPORT_COLUMNS]
    if row["_edited_fields"]:
        row["USER_EDITED"] = True
    return row


def _legacy_rows(purpose: str, data: dict) -> tuple[list[dict], str | None]:
    """Rows from older provider shapes. Returns (rows, notice)."""
    rows = data.get("rows")
    if isinstance(rows, list) and rows:
        return [r for r in rows if isinstance(r, dict)], None
    if purpose == VENDOR_INVOICE:
        out = []
        vendor = clean_cell(data.get("vendor_name"))
        for e in data.get("expenses") or []:
            if not isinstance(e, dict):
                continue
            desc = clean_cell(e.get("expense_description"))
            out.append({
                "Cheque/Ref No*": data.get("bill_number"),
                "Transaction Date*": data.get("bill_date"),
                "Amount*": e.get("expense_amount"),
                "Bill Head*": e.get("expense_code"),
                "Comments": f"{vendor} - {desc}" if vendor != "-" and desc != "-" else desc,
            })
        return out, None
    if purpose == MEMBER_RECEIPT:
        # A single-receipt payload is one row ONLY if it carries a real accounting
        # value; an all-placeholder object is never turned into a transaction.
        single = {col: None for col in NBH_IMPORT_COLUMNS}
        for col, alts in _SYNONYMS.items():
            for alt in alts:
                if data.get(alt) not in (None, ""):
                    single[col] = data.get(alt)
                    break
        meaningful = parse_amount(single["Amount*"]).found or clean_cell(single["Cheque/Ref No*"]) != "-"
        if meaningful:
            return [single], None
        return [], ("No accounting transactions could be reliably extracted from this document. "
                    "Nothing has been invented; please review the original document.")
    return [], None


_NON_TX_TYPES = ("TOTAL", "SUBTOTAL", "OPENING_BALANCE", "CLOSING_BALANCE", "CARRY_FORWARD")


def _separate_non_transactions(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Provider rows that are totals/balances are not transactions: they are
    recorded (with a reason) instead of being exported."""
    from app.accounting.semantic_rows import classify_semantic_row

    kept, excluded = [], []
    for index, row in enumerate(rows, start=1):
        text = str(row.get("Comments") or row.get("particulars") or row.get("description") or row.get("narration") or "")
        amount = parse_amount(row.get("Amount*") or row.get("amount"))
        try:
            kind = classify_semantic_row(text, amount=amount.value if amount.found else None,
                                         document_purpose="PETTY_CASH_REGISTER")
        except Exception:
            kind = None
        if kind in _NON_TX_TYPES:
            excluded.append({
                "candidate_id": f"p{index:03d}", "page": row.get("page"), "classification": kind,
                "status": "NON_TRANSACTION",
                "status_reason": f"{kind.replace('_', ' ').lower()} line; kept for reconciliation, not exported",
                "row_id": None, "fields": {"TEXT": [{"text": text}]},
            })
        else:
            kept.append(row)
    return kept, excluded


def finalize_extraction(purpose: str, data: Any) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"Extraction result for {purpose} must be an object, got {type(data).__name__}")
    purpose = purpose.upper()
    result = dict(data)
    result["purpose"] = purpose
    if "row_evidence" in data:
        raw_rows = [r for r in data.get("rows") or [] if isinstance(r, dict)]
        notice = data.get("extraction_notice")
    else:
        raw_rows, notice = _legacy_rows(purpose, data)
        raw_rows, candidates = _separate_non_transactions(raw_rows)
        result.setdefault("row_evidence", {})
        result["candidate_ledger"] = {
            "schema_version": 2, "source_candidates": len(candidates), "accepted": 0,
            "needs_review": 0, "rejected_with_reason": 0,
            "non_transaction": sum(1 for c in candidates if c["status"] == "NON_TRANSACTION"),
            "unresolved": 0, "model_only_rows": len(raw_rows),
            "balanced": True,
            "equation": (f"{len(candidates)} provider row(s) excluded as non-transactions; "
                         f"{len(raw_rows)} provider row(s) without OCR candidate evidence"),
            "unaccounted": [], "candidates": candidates,
        }
        if notice:
            result["extraction_notice"] = notice
    rows = [normalize_row(r, f"r0_{i + 1:03d}") for i, r in enumerate(raw_rows)]
    # Row ids must be unique; a duplicate id would make edits ambiguous.
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if row["_row_id"] in seen:
            row["_row_id"] = f"{row['_row_id']}_{i + 1}"
        seen.add(row["_row_id"])
    result["rows"] = rows
    result["inflow_rows"] = [normalize_row(r, f"ri_{i + 1:03d}") for i, r in enumerate(data.get("inflow_rows") or [])
                             if isinstance(r, dict)]
    if not rows:
        result["extraction_outcome"] = result.get("extraction_outcome") or "NO_RELIABLE_TRANSACTIONS"
    else:
        result.setdefault("extraction_outcome", "EXTRACTED")
    result.setdefault("_extraction_provider", data.get("_extraction_provider") or "UNKNOWN")
    try:
        result["reconciliation"] = AccountingReconciliationService().reconcile(result, purpose)
    except Exception as exc:  # reconciliation must never block extraction, but is never silent
        logger.warning("Reconciliation failed: %s", exc)
        result["reconciliation"] = {"overall_status": "ERROR", "error": type(exc).__name__, "checks": []}
    return result


def exported_amount(row: dict):
    reading = parse_amount(row.get("Amount*"))
    return reading.value if reading.found else None


__all__ = ["finalize_extraction", "normalize_row", "clean_cell", "META_KEYS", "format_amount", "PETTY_CASH_REGISTER"]
