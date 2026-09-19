"""Human review actions. Rows are addressed by immutable row_id, never position.

Every action returns audit entries (issue, resolution, user, timestamp,
before, after, reason). A reviewer-edited field is recorded in the row's
`_edited_fields` and is immutable to every automated stage afterwards (repair,
mapping, re-validation, re-extraction).
"""

from datetime import datetime, timezone

from app.accounting.dates import canonical_date_text, parse_date
from app.accounting.money import format_amount, parse_amount
from app.accounting.schemas import HumanCorrection
from app.accounting.templates import NBH_IMPORT_COLUMNS


class ReviewError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _audit(user: str, action: str, row_id: str | None, field: str, before, after, reason: str | None,
           issue: str | None = None) -> HumanCorrection:
    return HumanCorrection(field=field, old_value=before, new_value=after, user_email=user, timestamp=_now(),
                           row_id=row_id, action=action, issue=issue, reason=reason)


def _rows(data: dict) -> list[dict]:
    return data.setdefault("rows", [])


def _find_row(data: dict, row_id: str) -> dict:
    for row in _rows(data):
        if row.get("_row_id") == row_id:
            return row
    raise ReviewError(f"ROW_NOT_FOUND: {row_id}")


def _candidates(data: dict) -> list[dict]:
    return (data.get("candidate_ledger") or {}).get("candidates") or []


def recount_ledger(data: dict) -> None:
    ledger = data.get("candidate_ledger")
    if not ledger or not ledger.get("candidates"):
        return
    states = ("ACCEPTED", "NEEDS_REVIEW", "REJECTED_WITH_REASON", "NON_TRANSACTION", "UNRESOLVED")
    counts = {s: 0 for s in states}
    unaccounted = []
    for cand in ledger["candidates"]:
        if cand.get("status") in counts:
            counts[cand["status"]] += 1
        else:
            unaccounted.append(cand.get("candidate_id"))
    total = len(ledger["candidates"])
    ledger.update({
        "accepted": counts["ACCEPTED"], "needs_review": counts["NEEDS_REVIEW"],
        "rejected_with_reason": counts["REJECTED_WITH_REASON"], "non_transaction": counts["NON_TRANSACTION"],
        "unresolved": counts["UNRESOLVED"], "unaccounted": unaccounted,
        "balanced": not unaccounted and sum(counts.values()) == total,
        "equation": (f"{total} source candidates = {counts['ACCEPTED']} accepted + {counts['NEEDS_REVIEW']} needs review"
                     f" + {counts['REJECTED_WITH_REASON']} rejected + {counts['NON_TRANSACTION']} non-transaction"
                     f" + {counts['UNRESOLVED']} unresolved"),
    })


def _sync_candidate(data: dict, row: dict):
    for cand in _candidates(data):
        if cand.get("row_id") == row.get("_row_id"):
            cand["status"] = "ACCEPTED" if row.get("_status") in ("ACCEPTED", "USER_CONFIRMED") else "NEEDS_REVIEW"
            cand["status_reason"] = f"exported as {row.get('_row_id')} ({row.get('_status')})"
    recount_ledger(data)


def _normalise_value(column: str, value) -> str:
    text = "-" if value is None else str(value).strip() or "-"
    if column == "Amount*" and text != "-":
        reading = parse_amount(text)
        return format_amount(reading.value) if reading.found else text
    if column in ("Transaction Date*", "Cheque Date") and text != "-":
        reading = parse_date(text)
        return canonical_date_text(reading.value) if reading.found else text
    return text


def confirm_row(data: dict, row_id: str, user: str, reason: str | None = None) -> list[HumanCorrection]:
    row = _find_row(data, row_id)
    before = row.get("_status")
    row["_status"] = "USER_CONFIRMED"
    _sync_candidate(data, row)
    return [_audit(user, "CONFIRM", row_id, "_status", before, "USER_CONFIRMED", reason, f"ROW_NEEDS_REVIEW:{row_id}")]


def edit_row(data: dict, row_id: str, changes: dict, user: str, reason: str | None = None,
             confirm: bool = True) -> list[HumanCorrection]:
    row = _find_row(data, row_id)
    unknown = [c for c in changes if c not in NBH_IMPORT_COLUMNS]
    if unknown:
        raise ReviewError(f"UNKNOWN_COLUMNS: {', '.join(unknown)}")
    entries = []
    edited = list(row.get("_edited_fields") or [])
    for column, value in changes.items():
        new = _normalise_value(column, value)
        before = row.get(column)
        if new == before:
            continue
        row[column] = new
        if column not in edited:
            edited.append(column)
        entries.append(_audit(user, "EDIT", row_id, column, before, new, reason, f"ROW_NEEDS_REVIEW:{row_id}"))
    row["_edited_fields"] = edited
    if edited:
        row["USER_EDITED"] = True
    if confirm:
        before = row.get("_status")
        row["_status"] = "USER_CONFIRMED"
        if before != "USER_CONFIRMED":
            entries.append(_audit(user, "CONFIRM", row_id, "_status", before, "USER_CONFIRMED", reason,
                                  f"ROW_NEEDS_REVIEW:{row_id}"))
    _sync_candidate(data, row)
    return entries


def add_row(data: dict, values: dict, user: str, reason: str | None = None, row_id: str | None = None) -> list[HumanCorrection]:
    if not reason:
        raise ReviewError("REASON_REQUIRED: adding a row needs a reason")
    existing = {r.get("_row_id") for r in _rows(data)}
    n = 1
    while row_id is None or row_id in existing:
        row_id = f"rh_{n:03d}"
        n += 1
    row = {col: _normalise_value(col, values.get(col)) for col in NBH_IMPORT_COLUMNS}
    row.update({"_row_id": row_id, "_status": "USER_CONFIRMED", "_edited_fields": list(NBH_IMPORT_COLUMNS),
                "USER_EDITED": True})
    _rows(data).append(row)
    return [_audit(user, "ADD_ROW", row_id, "*", None, {c: row[c] for c in NBH_IMPORT_COLUMNS}, reason)]


def delete_row(data: dict, row_id: str, user: str, reason: str | None) -> list[HumanCorrection]:
    if not reason:
        raise ReviewError("REASON_REQUIRED: deleting a row needs a reason")
    row = _find_row(data, row_id)
    data["rows"] = [r for r in _rows(data) if r.get("_row_id") != row_id]
    for cand in _candidates(data):
        if cand.get("row_id") == row_id:
            cand["status"] = "REJECTED_WITH_REASON"
            cand["status_reason"] = f"row {row_id} removed by reviewer: {reason}"
    recount_ledger(data)
    return [_audit(user, "DELETE_ROW", row_id, "*", {c: row.get(c) for c in NBH_IMPORT_COLUMNS}, None, reason)]


def _find_candidate(data: dict, candidate_id: str) -> dict:
    for cand in _candidates(data):
        if cand.get("candidate_id") == candidate_id:
            return cand
    raise ReviewError(f"CANDIDATE_NOT_FOUND: {candidate_id}")


def dismiss_candidate(data: dict, candidate_id: str, user: str, reason: str | None) -> list[HumanCorrection]:
    if not reason:
        raise ReviewError("REASON_REQUIRED: dismissing a source row needs a reason")
    cand = _find_candidate(data, candidate_id)
    before = cand.get("status")
    cand["status"] = "REJECTED_WITH_REASON"
    cand["status_reason"] = f"dismissed by reviewer: {reason}"
    recount_ledger(data)
    return [_audit(user, "DISMISS", candidate_id, "candidate", before, "REJECTED_WITH_REASON", reason,
                   f"UNRESOLVED_SOURCE_ROW:{candidate_id}")]


def promote_candidate(data: dict, candidate_id: str, values: dict, user: str, reason: str | None) -> list[HumanCorrection]:
    cand = _find_candidate(data, candidate_id)
    row_id = "r" + candidate_id[1:] if candidate_id.startswith("c") else None
    entries = add_row(data, values, user, reason or "added from an unresolved source region", row_id=row_id)
    new_id = entries[0].row_id
    cand["status"] = "ACCEPTED"
    cand["row_id"] = new_id
    cand["status_reason"] = f"added by reviewer as {new_id}"
    recount_ledger(data)
    entries.append(_audit(user, "PROMOTE", new_id, "candidate", "UNRESOLVED", "ACCEPTED", reason,
                          f"UNRESOLVED_SOURCE_ROW:{candidate_id}"))
    return entries


def apply_row_list(data: dict, submitted: list[dict], user: str, reason: str | None = None) -> list[HumanCorrection]:
    """Apply a full edited row list (the review grid's Save): diff by row_id."""
    entries: list[HumanCorrection] = []
    current = {r.get("_row_id"): r for r in _rows(data)}
    seen = set()
    for item in submitted:
        if not isinstance(item, dict):
            continue
        rid = item.get("_row_id")
        if rid and rid in current:
            seen.add(rid)
            changes = {c: item.get(c) for c in NBH_IMPORT_COLUMNS
                       if c in item and _normalise_value(c, item.get(c)) != current[rid].get(c)}
            confirm = bool(item.get("_confirm")) or bool(changes)
            if changes or confirm:
                entries += edit_row(data, rid, changes, user, reason, confirm=confirm)
        else:
            entries += add_row(data, item, user, reason or "row added in review grid")
    for rid in list(current):
        if rid not in seen:
            entries += delete_row(data, rid, user, reason or "row removed in review grid")
    return entries
