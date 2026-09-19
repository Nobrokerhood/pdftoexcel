"""Evidence fusion: extracted rows + OCR source candidates -> decided NBH rows.

Evidence hierarchy (highest first), applied per field:

    SOURCE DOCUMENT TEXT (digital PDF text)
  > MULTI-OCR AGREEMENT (two independent OCR engines)
  > GEOMETRY / ROW STRUCTURE (the value sits in this row's column)
  > GEMINI VISUAL (page read, then focused crop arbitration)
  > SINGLE OCR READING

Rules that are never relaxed:
* A field is VERIFIED only when two independent sources agree and no
  independent source exclusively reads something else. Otherwise the row is
  NEEDS_REVIEW with every candidate value and the reason shown.
* Confidence scores never pick a value.
* Every source candidate ends in exactly one ledger state.
* Rows get an immutable `row_id` here; later stages never use position.
"""

import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from app.accounting.dates import parse_date
from app.accounting.money import format_amount, parse_amount
from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.source_candidates import (
    ACCEPTED, AMOUNT, ANNOTATION, BALANCE, CONTINUATION, DATE, DOCUMENT_TEXT, HEADER, INFLOW, NEEDS_REVIEW,
    NON_TRANSACTION, OTHER_AMOUNT, RECEIPT_AMOUNT, REF, REJECTED_WITH_REASON, TERMINAL_STATES, TOTAL, TRANSACTION, UNRESOLVED,
    SourceCandidate, amount_readings, anchors_readable, band_digit_text, date_readings, ref_readings,
)
from app.accounting.templates import NBH_IMPORT_COLUMNS

VERIFIED = "VERIFIED"
ARBITRATED = "ARBITRATED"   # OCR disagreement resolved by blind visual arbitration
ACCEPTED_FIELD_STATES = ("VERIFIED", "ARBITRATED")
CONFLICT = "CONFLICT"
SINGLE_SOURCE = "SINGLE_SOURCE"
MISSING = "MISSING"
USER_EDITED = "USER_EDITED"

REF_COL = "Cheque/Ref No*"
DATE_COL = "Transaction Date*"
AMOUNT_COL = "Amount*"
KEY_COLUMNS = (REF_COL, DATE_COL, AMOUNT_COL)
MANDATORY_COLUMNS = (AMOUNT_COL, DATE_COL)

GEMINI_PAGE = "gemini_page"
GEMINI_CROP = "gemini_crop"
HIERARCHY = ("pdf_text", "multi_ocr", GEMINI_CROP, GEMINI_PAGE, "rapidocr", "paddleocr")

ALL_AMOUNT_ROLES = (AMOUNT, RECEIPT_AMOUNT, OTHER_AMOUNT)


@dataclass
class SourceVote:
    source: str       # pdf_text | rapidocr | paddleocr | gemini_page | gemini_crop | user
    value: str        # canonical comparable value
    raw: str
    detail: str = ""  # variant / line id / decision

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FieldDecision:
    column: str
    value: str
    status: str
    reason: str
    votes: list[SourceVote] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"column": self.column, "value": self.value, "status": self.status, "reason": self.reason,
                "votes": [v.to_dict() for v in self.votes], "alternatives": list(self.alternatives)}


@dataclass
class DecidedRow:
    row_id: str
    page: int | None
    bbox: tuple | None
    kind: str                       # EXPENSE | INFLOW | TRANSACTION | LINE_ITEM
    candidate_id: str | None
    origin: str                     # MATCHED | MODEL_ONLY | OCR_ONLY
    values: dict[str, str]
    decisions: dict[str, FieldDecision] = field(default_factory=dict)
    status: str = NEEDS_REVIEW
    reasons: list[str] = field(default_factory=list)
    gemini_index: int | None = None
    extra: dict = field(default_factory=dict)

    def nbh_row(self) -> dict:
        row = {col: self.values.get(col, "-") or "-" for col in NBH_IMPORT_COLUMNS}
        row["_row_id"] = self.row_id
        row["_status"] = self.status
        return row

    def evidence(self) -> dict:
        return {
            "row_id": self.row_id, "page": self.page, "bbox": list(self.bbox) if self.bbox else None,
            "kind": self.kind, "candidate_id": self.candidate_id, "origin": self.origin,
            "status": self.status, "reasons": list(self.reasons),
            "fields": {col: d.to_dict() for col, d in self.decisions.items()},
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# comparable values
# ---------------------------------------------------------------------------

def canon_amount(text: Any) -> str | None:
    reading = parse_amount(text)
    return format_amount(reading.value) if reading.found else None


def canon_date(text: Any) -> str | None:
    reading = parse_date(text)
    return reading.value.isoformat() if reading.found else None


def canon_ref(text: Any) -> str | None:
    if text is None:
        return None
    t = str(text).strip()
    if t in ("", "-"):
        return None
    digits = re.sub(r"\D", "", t)
    return digits or t.upper()


def _engine_source(engine: str) -> str:
    return engine if engine in ("pdf_text", "rapidocr", "paddleocr") else engine


def ocr_votes(cand: SourceCandidate | None, column: str) -> list[SourceVote]:
    if cand is None:
        return []
    votes: list[SourceVote] = []
    if column == AMOUNT_COL:
        for value, ev in amount_readings(cand, roles=ALL_AMOUNT_ROLES):
            votes.append(SourceVote(_engine_source(ev.engine), format_amount(value), ev.text, f"{ev.variant} {ev.line_id}"))
    elif column == DATE_COL:
        for value, ev in date_readings(cand):
            votes.append(SourceVote(_engine_source(ev.engine), value.isoformat(), ev.text, f"{ev.variant} {ev.line_id}"))
    elif column == REF_COL:
        for digits, ev in ref_readings(cand):
            votes.append(SourceVote(_engine_source(ev.engine), digits, ev.text, f"{ev.variant} {ev.line_id}"))
    return votes


def _canon(column: str, text: Any) -> str | None:
    if column == AMOUNT_COL:
        return canon_amount(text)
    if column == DATE_COL:
        return canon_date(text)
    if column == REF_COL:
        return canon_ref(text)
    return None


def decide_field(column: str, proposed_raw: Any, votes: list[SourceVote], digital_text: str = "",
                 anchored_to: str | None = None) -> FieldDecision:
    """Decide one key field from independent sources. Never picks by confidence."""
    by_source: dict[str, set[str]] = {}
    for v in votes:
        by_source.setdefault(v.source, set()).add(v.value)

    proposed = _canon(column, proposed_raw)
    values = {v.value for v in votes}
    if proposed is None and not values:
        if proposed_raw not in (None, "", "-"):
            return FieldDecision(column, str(proposed_raw), NEEDS_REVIEW,
                                 f"'{proposed_raw}' is not a valid {column.rstrip('*').lower()}", votes)
        return FieldDecision(column, "-", MISSING, "no source shows a value", votes)

    def support(value: str) -> set[str]:
        return {s for s, vals in by_source.items() if value in vals}

    # Digital PDF text is the document itself. Amounts and refs are also
    # confirmed when their digits appear in the row's digital text.
    if "pdf_text" in by_source or digital_text:
        target = proposed or next(iter(by_source.get("pdf_text", [])), None)
        in_text = target is not None and (
            target in by_source.get("pdf_text", set())
            or (column in (AMOUNT_COL, REF_COL) and len(re.sub(r"\D", "", target)) >= 2
                and re.sub(r"\D", "", target) in digital_text)
        )
        if in_text:
            return FieldDecision(column, target, VERIFIED, "value is present in the document's own digital text", votes)
        if target is not None and digital_text:
            return FieldDecision(column, target, CONFLICT,
                                 "value not found in this row's digital text; confirm against the source", votes,
                                 sorted(values - {target}))

    # Independence: the page extraction was shown the primary OCR engine's text,
    # so the two are one evidence group, never two agreeing sources.
    def group(source: str) -> str:
        return anchored_to if (source == GEMINI_PAGE and anchored_to) else source

    groups: dict[str, set[str]] = {}
    for src, vals in by_source.items():
        groups.setdefault(group(src), set()).update(vals)

    def g_support(value: str) -> set[str]:
        return {g for g, vals in groups.items() if value in vals}

    candidates = set(values) | ({proposed} if proposed else set())
    ranked = []
    for value in candidates:
        sup = g_support(value)
        ocr_groups = {g for g in sup if g in ("rapidocr", "paddleocr", "pdf_text")}
        rank = (len(ocr_groups) >= 2, len(sup), GEMINI_CROP in sup, GEMINI_PAGE in support(value))
        ranked.append((rank, value))
    ranked.sort(reverse=True)
    best = ranked[0][1]
    sup = g_support(best)
    dissent = {g for g, vals in groups.items() if best not in vals}
    alternatives = sorted(v for v in candidates if v != best)
    names = ", ".join(sorted(sup)) or "none"

    if len(sup) >= 2 and not dissent:
        return FieldDecision(column, best, VERIFIED, f"independent sources agree ({names})", votes, alternatives)
    if len(sup) >= 2 and GEMINI_CROP in sup and len(dissent) == 1 and len(sup) > len(dissent):
        other = next(iter(dissent))
        return FieldDecision(
            column, best, ARBITRATED,
            f"OCR disagreement resolved by blind visual arbitration: {names} read {best}; {other} read "
            f"{', '.join(sorted(groups[other]))}", votes, alternatives)
    if len(sup) >= 2:
        return FieldDecision(
            column, best, CONFLICT,
            f"{names} read {best}; {', '.join(sorted(dissent))} read {', '.join(alternatives)}",
            votes, alternatives)
    if not alternatives:
        return FieldDecision(column, best, SINGLE_SOURCE, f"only one independent source ({names}) shows this value",
                             votes, alternatives)
    return FieldDecision(column, best, CONFLICT,
                         f"sources disagree: {best} ({names}) vs {', '.join(alternatives)}", votes, alternatives)


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------

def _match_score(cand: SourceCandidate, ref: str | None, date: str | None, amount: str | None) -> int:
    score = 0
    if ref and any(d == ref for d, _ in ref_readings(cand)):
        score += 2
    elif ref and len(ref) >= 5 and ref in band_digit_text(cand):
        score += 2
    if amount and any(format_amount(v) == amount for v, _ in amount_readings(cand, roles=ALL_AMOUNT_ROLES)):
        score += 2
    if date and any(v.isoformat() == date for v, _ in date_readings(cand)):
        score += 1
    return score


def align_rows(model_rows: list[dict], candidates: list[SourceCandidate], alias_to_line: dict[str, str]) -> dict[int, SourceCandidate]:
    """Map extracted row index -> source candidate (one-to-one)."""
    by_line: dict[str, SourceCandidate] = {}
    for cand in candidates:
        for lid in cand.line_ids:
            by_line[lid] = cand
        for items in cand.fields.values():
            for ev in items:
                by_line.setdefault(ev.line_id, cand)
    eligible = {TRANSACTION, INFLOW, CONTINUATION, ANNOTATION, TOTAL, BALANCE}
    taken: set[str] = set()
    result: dict[int, SourceCandidate] = {}

    def parent(c: SourceCandidate) -> SourceCandidate:
        if c.classification == CONTINUATION and c.parent_id:
            return next((x for x in candidates if x.candidate_id == c.parent_id), c)
        return c

    # Pass 1: cited OCR line ids.
    for idx, row in enumerate(model_rows):
        counts: dict[str, int] = {}
        for alias in row.get("source_line_ids") or []:
            lid = alias_to_line.get(str(alias).strip())
            cand = by_line.get(lid) if lid else None
            if cand is not None and cand.classification in eligible:
                cand = parent(cand)
                counts[cand.candidate_id] = counts.get(cand.candidate_id, 0) + 1
        if not counts:
            continue
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        for cid, _ in ordered:
            cand = next(c for c in candidates if c.candidate_id == cid)
            if cid in taken:
                continue
            # A cited region must not contradict the row outright.
            sig = row.get("_sig", {})
            if cand.classification in (TRANSACTION, INFLOW) or _match_score(cand, sig.get("ref"), sig.get("date"), sig.get("amount")) >= 2:
                result[idx] = cand
                taken.add(cid)
                break

    # Pass 2: value matching for rows without usable citations.
    for idx, row in enumerate(model_rows):
        if idx in result:
            continue
        sig = row.get("_sig", {})
        best, best_score = None, 0
        for cand in candidates:
            if cand.candidate_id in taken or cand.classification not in (TRANSACTION, INFLOW):
                continue
            score = _match_score(cand, sig.get("ref"), sig.get("date"), sig.get("amount"))
            if score > best_score:
                best, best_score = cand, score
        if best is not None and best_score >= 2:
            result[idx] = best
            taken.add(best.candidate_id)
    return result


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------

_NON_TX_REASON = {
    HEADER: "column header",
    BALANCE: "balance / carry-forward line; kept in the balance summary",
    TOTAL: "written total; used for reconciliation, not a transaction",
    DOCUMENT_TEXT: "document header text (ids, period, address)",
    ANNOTATION: "narrative text with no date, reference or amount",
}


def build_ledger(candidates: list[SourceCandidate], model_only_rows: int) -> dict:
    counts = {state: 0 for state in TERMINAL_STATES}
    missing_state = []
    for cand in candidates:
        if cand.status in counts:
            counts[cand.status] += 1
        else:
            missing_state.append(cand.candidate_id)
    total = len(candidates)
    balanced = not missing_state and sum(counts.values()) == total
    return {
        "schema_version": 2,
        "source_candidates": total,
        "accepted": counts[ACCEPTED],
        "needs_review": counts[NEEDS_REVIEW],
        "rejected_with_reason": counts[REJECTED_WITH_REASON],
        "non_transaction": counts[NON_TRANSACTION],
        "unresolved": counts[UNRESOLVED],
        "model_only_rows": model_only_rows,
        "balanced": balanced,
        "equation": (f"{total} source candidates = {counts[ACCEPTED]} accepted + {counts[NEEDS_REVIEW]} needs review"
                     f" + {counts[REJECTED_WITH_REASON]} rejected + {counts[NON_TRANSACTION]} non-transaction"
                     f" + {counts[UNRESOLVED]} unresolved"),
        "unaccounted": missing_state,
        "candidates": [c.to_dict() for c in candidates],
    }


def finalize_candidates(candidates: list[SourceCandidate], rows: list[DecidedRow], tabular_pages: set[int]):
    """Give every candidate exactly one terminal state with a reason."""
    by_id = {c.candidate_id: c for c in candidates}
    for row in rows:
        if row.candidate_id and row.candidate_id in by_id:
            cand = by_id[row.candidate_id]
            cand.row_id = row.row_id
            if row.kind == "INFLOW":
                cand.status = NON_TRANSACTION
                cand.status_reason = f"cash inflow recorded in the balance summary as {row.row_id}"
            else:
                cand.status = ACCEPTED if row.status == ACCEPTED else NEEDS_REVIEW
                cand.status_reason = f"exported as {row.row_id}"
    for cand in candidates:
        if cand.status:
            continue
        cls = cand.classification
        if cls == CONTINUATION:
            parent = by_id.get(cand.parent_id or "")
            cand.status = NON_TRANSACTION
            cand.status_reason = (f"wrapped narration of {parent.row_id or parent.candidate_id}"
                                  if parent else "wrapped narration")
        elif cls in _NON_TX_REASON:
            cand.status = NON_TRANSACTION
            cand.status_reason = _NON_TX_REASON[cls]
        elif cls in (TRANSACTION, INFLOW):
            readable = anchors_readable(cand) or bool(amount_readings(cand, roles=ALL_AMOUNT_ROLES))
            if not readable:
                cand.status = REJECTED_WITH_REASON
                cand.status_reason = "no readable reference, date or amount in this region"
            elif cand.page in tabular_pages:
                cand.status = UNRESOLVED
                cand.status_reason = ("a table row with readable accounting values that extraction did not report; "
                                      "a reviewer must confirm it is not a missing transaction")
            else:
                cand.status = NON_TRANSACTION
                cand.status_reason = "amount-bearing text outside a transaction table (not a reported line item)"
        else:
            cand.status = NON_TRANSACTION
            cand.status_reason = f"classified {cls or 'unknown'}"
