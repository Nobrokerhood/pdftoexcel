"""Assemble decided NBH rows for one document.

Inputs: the OCR representation, its source candidates, and (optionally) a
validated Gemini extraction. Output: NBH rows with immutable row ids, per-field
evidence, the candidate ledger, and balance information.

When Gemini is unavailable or its response broke the contract, the same
assembly runs on OCR candidates alone: rows are still produced from the source
(never zero rows while the source shows transactions), and every one is marked
NEEDS_REVIEW unless independent OCR engines agree.
"""

import logging
from collections import Counter
from typing import Callable

from app.accounting.dates import canonical_date_text
from app.accounting.fusion import (
    ACCEPTED, ARBITRATED, AMOUNT_COL, CONFLICT, DATE_COL, GEMINI_CROP, GEMINI_PAGE, LOCAL_EXTRACTOR, KEY_COLUMNS, MANDATORY_COLUMNS, MISSING,
    NEEDS_REVIEW, REF_COL, SINGLE_SOURCE, VERIFIED, DecidedRow, FieldDecision, SourceVote, _canon, align_rows,
    build_ledger, decide_field, finalize_candidates, ocr_votes,
)
from app.accounting.money import format_amount, parse_amount
from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.source_candidates import (
    AMOUNT as AMOUNT_ROLE, CATEGORY, INFLOW, OTHER_AMOUNT, PARTICULARS, RECEIPT_AMOUNT, TRANSACTION, SourceCandidate, amount_readings, anchors_readable,
    band_digit_text, date_readings, ref_readings,
)
from app.accounting.templates import NBH_IMPORT_COLUMNS
from app.agents.gemini_contract import member_row_values

logger = logging.getLogger(__name__)

# Arbitration callback: rows needing a focused look -> {row_id: {column: (decision, value)}}
Arbiter = Callable[[list[DecidedRow], dict], dict]


def _blank_values() -> dict[str, str]:
    return {col: "-" for col in NBH_IMPORT_COLUMNS}


def _clean(text) -> str:
    if text is None:
        return "-"
    t = str(text).strip()
    return t if t and t.lower() not in {"null", "none", "n/a", "unknown", "undefined", "nan"} else "-"


def _row_id(cand: SourceCandidate | None, index: int) -> str:
    if cand is not None:
        return "r" + cand.candidate_id[1:]
    return f"rm_{index + 1:03d}"


def _model_rows(purpose: str, model) -> list[dict]:
    """Normalise each purpose's rows to {values, kind, source_line_ids, _sig}."""
    rows = []
    if model is None:
        return rows
    if purpose == PETTY_CASH_REGISTER:
        for r in model.rows:
            values = _blank_values()
            values["Payment Type*"] = "Cash"
            values[REF_COL] = _clean(r.voucher_no)
            values["Bill Head*"] = _clean(r.category)
            values[AMOUNT_COL] = _clean(r.amount)
            values[DATE_COL] = _clean(r.date)
            values["Comments"] = _clean(r.particulars)
            rows.append({"values": values, "kind": "INFLOW" if r.row_kind == "INFLOW" else "EXPENSE",
                         "source_line_ids": r.source_line_ids, "serial_no": r.serial_no})
    elif purpose == MEMBER_RECEIPT:
        for r in model.rows:
            rows.append({"values": member_row_values(r), "kind": "TRANSACTION",
                         "source_line_ids": r.get("source_line_ids") or []})
    elif purpose == VENDOR_INVOICE:
        vendor = _clean(model.vendor_name)
        for item in model.line_items:
            values = _blank_values()
            values[REF_COL] = _clean(model.bill_number)
            values[DATE_COL] = _clean(model.bill_date)
            values[AMOUNT_COL] = _clean(item.amount)
            desc = _clean(item.description)
            values["Comments"] = f"{vendor} - {desc}" if vendor != "-" and desc != "-" else (desc if desc != "-" else vendor)
            rows.append({"values": values, "kind": "LINE_ITEM", "source_line_ids": item.source_line_ids,
                         "quantity": item.quantity, "rate": item.rate})
    for r in rows:
        v = r["values"]
        r["_sig"] = {"ref": _canon(REF_COL, v[REF_COL]), "date": _canon(DATE_COL, v[DATE_COL]),
                     "amount": _canon(AMOUNT_COL, v[AMOUNT_COL])}
    return rows


def _most_common(values: list[str]) -> str | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _ocr_row_values(cand: SourceCandidate, purpose: str) -> dict[str, str]:
    """Row values read from OCR evidence alone (used when no model row exists)."""
    values = _blank_values()
    refs = [d for d, _ in ref_readings(cand) if len(d) >= 2]
    dates = [v.isoformat() for v, _ in date_readings(cand)]
    # Primary amount column first; otherwise the left-most other amount column
    # (debit/credit precede a running balance on statements).
    amounts = [format_amount(v) for v, _ in amount_readings(cand)]
    if not amounts:
        others = sorted(amount_readings(cand, roles=(RECEIPT_AMOUNT, OTHER_AMOUNT)), key=lambda t: t[1].bbox[0])
        amounts = [format_amount(others[0][0])] if others else []
    if refs:
        values[REF_COL] = _most_common(refs)
    if dates:
        values[DATE_COL] = canonical_date_text(_most_common(dates))
    if amounts:
        values[AMOUNT_COL] = _most_common(amounts)
    text = cand.fields.get(PARTICULARS) or []
    if text:
        values["Comments"] = text[0].text
    cat = cand.fields.get(CATEGORY) or []
    if cat:
        values["Bill Head*"] = cat[0].text
    if purpose == PETTY_CASH_REGISTER:
        values["Payment Type*"] = "Cash"
    return values


def _decide_row(row: DecidedRow, cand: SourceCandidate | None, digital: bool, extra_votes: dict | None = None,
                doc_digits: str = "", document_level: tuple = (), anchored_to: str | None = None):
    """(Re)compute decisions for the key columns of a row."""
    original = row.extra.setdefault("proposed", {c: row.values.get(c, "-") for c in KEY_COLUMNS})
    for column in KEY_COLUMNS:
        proposed = original.get(column, "-")
        votes = ocr_votes(cand, column)
        if column in document_level:
            # Vendor bill number/date are document-level: confirm against all document text.
            canon = _canon(column, proposed)
            digits = "".join(ch for ch in (canon or "") if ch.isdigit())
            if canon and len(digits) >= 2 and digits in doc_digits:
                votes = [SourceVote("document_text", canon, proposed, "found in document OCR text")]
        if row.origin != "OCR_ONLY":
            source = row.extra.get("proposal_source", GEMINI_PAGE)
            canon = _canon(column, proposed)
            if canon is not None:
                votes.append(SourceVote(source, canon, str(proposed), "proposed value"))
            elif proposed not in ("-", None, ""):
                votes.append(SourceVote(source, str(proposed), str(proposed), "proposed value (unparsed)"))
        for vote in (extra_votes or {}).get(column, []):
            votes.append(vote)
        decision = decide_field(column, proposed, votes,
                                band_digit_text(cand) if (digital and cand is not None) else "",
                                anchored_to=anchored_to)
        if column in document_level and votes and votes[0].source == "document_text" and decision.status in (
                SINGLE_SOURCE, CONFLICT):
            if decision.value == votes[0].value:
                decision = FieldDecision(column, decision.value, VERIFIED,
                                         "extracted value is present in the document text", votes)
        row.decisions[column] = decision
        if decision.value not in (None, ""):
            if column == AMOUNT_COL and decision.value != "-":
                row.values[column] = decision.value
            elif column == DATE_COL and decision.value != "-":
                row.values[column] = canonical_date_text(decision.value)
            elif column == REF_COL and decision.value != "-":
                # Keep the written reference text (e.g. 'IMPS/5116...') when it is the
                # decided value; replace it only when the evidence chose another value.
                if _canon(REF_COL, proposed) != decision.value:
                    row.values[column] = decision.value
            elif decision.value == "-":
                row.values[column] = "-" if _canon(column, proposed) is None and proposed in ("-", None, "") else row.values[column]


def _row_status(row: DecidedRow, mandatory=MANDATORY_COLUMNS):
    reasons = []
    for column, decision in row.decisions.items():
        if decision.status not in (VERIFIED, ARBITRATED, MISSING):
            reasons.append(f"{column.rstrip('*')}: {decision.reason}")
        if decision.status == MISSING and column in mandatory:
            reasons.append(f"{column.rstrip('*')}: not legible or not written in the source")
    if row.origin == "MODEL_ONLY":
        reasons.append("this row was reported by extraction but no matching row was found in the OCR evidence")
    if row.origin == "OCR_ONLY":
        reasons.append("OCR found this table row but extraction did not report it")
    row.reasons = reasons
    row.status = ACCEPTED if not reasons else NEEDS_REVIEW


def assemble_document(
    purpose: str,
    rep,
    candidates: list[SourceCandidate],
    layouts: list,
    model=None,
    shape_notes: list[str] | None = None,
    contract_error: str | None = None,
    alias_to_line: dict[str, str] | None = None,
    arbiter: Arbiter | None = None,
    local_rows: list[dict] | None = None,
) -> dict:
    alias_to_line = alias_to_line or {}
    digital_pages = {p.page_number for p in rep.pages if p.engine == "pdf_text"}
    handwritten_pages = {p.page_number for p in rep.pages if p.script != "PRINTED" or
                         (p.routing or {}).get("difficulty") == "DIFFICULT"}
    tabular_pages = {l.page for l in layouts if l.tabular}
    doc_digits = "".join(ch for p in rep.pages for r in p.evidence for l in r.lines for ch in l.text if ch.isdigit())
    document_level = (REF_COL, DATE_COL) if purpose == VENDOR_INVOICE else ()
    # Engine whose text the page extraction was shown, per page.
    # Gemini saw the primary engine's text; the local extractor READ it: neither is
    # independent of that engine.
    prompt_engine = {p.page_number: p.engine for p in rep.pages}

    def anchor(page):
        return prompt_engine.get(page) if page else None

    model_rows = _model_rows(purpose, model) if model is not None else list(local_rows or [])
    for r in model_rows:
        if "_sig" not in r:
            v = r["values"]
            r["_sig"] = {"ref": _canon(REF_COL, v[REF_COL]), "date": _canon(DATE_COL, v[DATE_COL]),
                         "amount": _canon(AMOUNT_COL, v[AMOUNT_COL])}
    alignment = align_rows(model_rows, candidates, alias_to_line)

    rows: list[DecidedRow] = []
    used_ids: set[str] = set()
    for idx, mrow in enumerate(model_rows):
        cand = alignment.get(idx)
        row_id = _row_id(cand, idx)
        if row_id in used_ids:
            row_id = f"{row_id}_{idx + 1}"
        used_ids.add(row_id)
        row = DecidedRow(
            row_id=row_id, page=cand.page if cand else None, bbox=cand.bbox if cand else None,
            kind=mrow["kind"], candidate_id=cand.candidate_id if cand else None,
            origin="MATCHED" if cand else "MODEL_ONLY", values=dict(mrow["values"]), gemini_index=idx,
            extra={**{k: v for k, v in mrow.items() if k in ("serial_no", "quantity", "rate") and v},
                   "proposal_source": GEMINI_PAGE if model is not None else LOCAL_EXTRACTOR},
        )
        _decide_row(row, cand, cand is not None and cand.page in digital_pages, doc_digits=doc_digits,
                    document_level=document_level, anchored_to=anchor(row.page))
        rows.append(row)

    # Source rows extraction did not report: never dropped. In a transaction
    # table they become review rows built from OCR; elsewhere the ledger records them.
    matched = {c.candidate_id for c in alignment.values()}
    if purpose in (PETTY_CASH_REGISTER, MEMBER_RECEIPT):
        for cand in candidates:
            if cand.candidate_id in matched or cand.page not in tabular_pages:
                continue
            is_inflow = cand.classification == INFLOW and purpose == PETTY_CASH_REGISTER
            if cand.classification != TRANSACTION and not is_inflow:
                continue
            if is_inflow:
                if not amount_readings(cand, roles=(RECEIPT_AMOUNT,)):
                    continue
            elif not (amount_readings(cand, roles=(AMOUNT_ROLE, RECEIPT_AMOUNT, OTHER_AMOUNT))
                      and anchors_readable(cand)):
                continue
            values = _ocr_row_values(cand, purpose)
            if is_inflow:
                amts = [format_amount(v) for v, _ in amount_readings(cand, roles=(RECEIPT_AMOUNT,))]
                values[AMOUNT_COL] = _most_common(amts) or "-"
            kind = "INFLOW" if is_inflow else ("EXPENSE" if purpose == PETTY_CASH_REGISTER else "TRANSACTION")
            row = DecidedRow(row_id=_row_id(cand, 0), page=cand.page, bbox=cand.bbox, kind=kind,
                             candidate_id=cand.candidate_id, origin="OCR_ONLY", values=values)
            _decide_row(row, cand, cand.page in digital_pages, doc_digits=doc_digits)
            rows.append(row)

    by_cand = {c.candidate_id: c for c in candidates}
    for row in rows:
        _row_status(row)

    # Focused visual arbitration for uncertain fields (and every handwritten amount).
    arbitration_trace: dict = {"requested": 0, "rows": [], "status": "NOT_RUN"}
    if arbiter is not None:
        needing = [r for r in rows if r.status != ACCEPTED or (r.page in handwritten_pages)]
        if needing:
            arbitration_trace["requested"] = len(needing)
            try:
                verdicts = arbiter(needing, {"purpose": purpose}) or {}
                failures = verdicts.pop("__failures__", [])
                arbitration_trace["status"] = "PARTIAL" if failures else "COMPLETED"
                arbitration_trace["failures"] = failures
            except Exception as exc:  # arbitration failure never changes values
                logger.warning("Visual arbitration failed: %s", type(exc).__name__)
                verdicts = {}
                arbitration_trace["status"] = f"FAILED: {type(exc).__name__}"
            for row in needing:
                verdict = verdicts.get(row.row_id)
                if not verdict:
                    continue
                extra = {}
                for column, (decision, value) in verdict.items():
                    canon = _canon(column, value) if value not in (None, "", "-") else None
                    if decision in ("ACCEPT", "CORRECT") and canon:
                        extra.setdefault(column, []).append(SourceVote(GEMINI_CROP, canon, str(value), decision))
                arbitration_trace["rows"].append({"row_id": row.row_id, "verdict": {
                    c: {"decision": d, "value": v} for c, (d, v) in verdict.items()}})
                cand = by_cand.get(row.candidate_id or "")
                _decide_row(row, cand, cand is not None and cand.page in digital_pages, extra, doc_digits,
                            document_level, anchored_to=anchor(row.page))
                _row_status(row)

    finalize_candidates(candidates, rows, tabular_pages)
    model_only = sum(1 for r in rows if r.origin == "MODEL_ONLY")
    ledger = build_ledger(candidates, model_only)

    exported = [r for r in rows if r.kind != "INFLOW"]
    inflows = [r for r in rows if r.kind == "INFLOW"]
    outcome = "EXTRACTED"
    notice = None
    if not exported:
        outcome = "NO_RELIABLE_TRANSACTIONS"
        notice = ("No accounting transactions could be reliably extracted from this document. Every source "
                  "region is listed in the candidate ledger with the reason it was not exported. Nothing has "
                  "been invented; please review the original document.")
    elif any(r.status != ACCEPTED for r in exported):
        outcome = "EXTRACTED_WITH_REVIEW_ITEMS"
    if contract_error:
        outcome = "EXTRACTION_CONTRACT_VIOLATION"
        notice = (f"The AI extraction response did not match the {purpose} contract ({contract_error}). "
                  f"Rows below were built from OCR evidence only and all require review.")

    return {
        "rows": [r.nbh_row() for r in exported],
        "row_evidence": {r.row_id: r.evidence() for r in rows},
        "inflow_rows": [{**r.nbh_row(), "_kind": "INFLOW"} for r in inflows],
        "candidate_ledger": ledger,
        "layouts": [l.to_dict() for l in layouts],
        "arbitration": arbitration_trace,
        "extraction_outcome": outcome,
        "extraction_notice": notice,
        "shape_notes": list(shape_notes or []),
    }
