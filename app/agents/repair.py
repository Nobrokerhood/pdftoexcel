"""Field-level, bounded repair.

The previous repair regenerated the whole record, so values in rows nobody had
flagged could change; user edits were restored by list position. Now:

* a repair request lists explicit field ids (`<row_id>.<column>`) with the
  current value, the reason, the evidence and the allowed candidate values;
* the response may only set those field ids; anything else is ignored;
* reviewer-edited fields are never requested and never written;
* when candidates exist, only a candidate value is accepted;
* a repaired value is a PROPOSAL: the row stays NEEDS_REVIEW for a human;
* rows are addressed by row_id, so reordering cannot misplace a value.
"""

import copy
import json
import logging
from typing import Protocol

from app.accounting.dates import canonical_date_text, parse_date
from app.accounting.money import format_amount, parse_amount
from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition

logger = logging.getLogger(__name__)

REPAIRABLE_STATES = ("CONFLICT", "SINGLE_SOURCE", "NEEDS_REVIEW", "MISSING")
REPAIRABLE_COLUMNS = ("Cheque/Ref No*", "Transaction Date*", "Amount*")
MAX_REPAIR_FIELDS = 40


class RepairProvider(Protocol):
    def repair(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, requests: list[dict]) -> dict:
        ...


def repair_requests(extracted_data: dict) -> list[dict]:
    evidence = extracted_data.get("row_evidence") or {}
    requests = []
    for row in extracted_data.get("rows") or []:
        rid = row.get("_row_id")
        if not rid or row.get("_status") in ("ACCEPTED", "USER_CONFIRMED"):
            continue
        edited = set(row.get("_edited_fields") or [])
        decisions = (evidence.get(rid) or {}).get("fields") or {}
        for column in REPAIRABLE_COLUMNS:
            if column in edited or row.get("USER_EDITED") and not row.get("_edited_fields"):
                continue
            d = decisions.get(column)
            current = row.get(column, "-")
            missing_mandatory = column in ("Amount*", "Transaction Date*") and current in ("-", "", None)
            if d is None:
                # No evidence at all (row supplied directly): only a missing mandatory value is repaired.
                if not missing_mandatory:
                    continue
                reason, candidates = "value missing", []
            else:
                # Evidence rows already had blind visual arbitration of their conflicts;
                # asking the model again adds no independent evidence. Only a mandatory
                # value that no source could read, on a row with a source region, is retried.
                if not (missing_mandatory and d.get("status") == "MISSING"
                        and (evidence.get(rid) or {}).get("bbox")):
                    continue
                reason = d.get("reason", "") or "no source could read this value"
                candidates = []
            requests.append({"field_id": f"{rid}.{column}", "row_id": rid, "column": column,
                             "current_value": current, "reason": reason, "allowed_candidates": candidates,
                             "page": (evidence.get(rid) or {}).get("page")})
    return requests[:MAX_REPAIR_FIELDS]


class GeminiRepairProvider:
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client

    def repair(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, requests: list[dict]) -> dict:
        if not self.gemini_client.settings.gemini_api_key:
            from app.core.errors import ServiceNotConfiguredError
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")
        from app.agents.extractor import source_parts
        prompt = (
            "You are a field-level repair agent for accounting extraction. For each requested field id, read the "
            "value from the page image. If allowed_candidates is non-empty, answer only with one of them or null. "
            "Never use arithmetic or totals to choose a digit. Do not return any field that was not requested.\n"
            'Return JSON only: {"corrections": [{"field_id": "<as given>", "value": "<value or null>", '
            '"evidence": "<what you see>"}]}\n'
            f"Purpose: {purpose}\nREQUESTS: {json.dumps(requests, ensure_ascii=False)}"
        )
        return self.gemini_client.generate_json(
            [prompt, *source_parts(source_bytes, self.gemini_client.settings.poppler_path)], purpose="repair")


def _canonical(column: str, value) -> str | None:
    if column == "Amount*":
        reading = parse_amount(value)
        return format_amount(reading.value) if reading.found else None
    if column in ("Transaction Date*", "Cheque Date"):
        reading = parse_date(value)
        return canonical_date_text(reading.value) if reading.found else None
    text = str(value).strip() if value is not None else ""
    return text or None


class RepairAgent:
    def __init__(self, provider: RepairProvider):
        self.provider = provider

    def repair(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, extracted_data: dict,
               verification_result: VerificationResult | None = None) -> dict:
        data = copy.deepcopy(extracted_data)
        requests = repair_requests(data)
        log = {"requested": [r["field_id"] for r in requests], "applied": [], "rejected": []}
        if not requests:
            data["repair_status"] = "NOTHING_TO_REPAIR"
            data["repair_log"] = log
            return data

        raw = self.provider.repair(source_bytes, purpose, template, requests)
        by_id = {r["field_id"]: r for r in requests}
        rows = {r.get("_row_id"): r for r in data.get("rows") or []}
        corrections = raw.get("corrections") if isinstance(raw, dict) else None
        for item in corrections or []:
            if not isinstance(item, dict):
                continue
            fid = str(item.get("field_id", ""))
            req = by_id.get(fid)
            if req is None:
                log["rejected"].append({"field_id": fid, "reason": "not a requested field"})
                continue
            row = rows.get(req["row_id"])
            if row is None or req["column"] in (row.get("_edited_fields") or []):
                log["rejected"].append({"field_id": fid, "reason": "row missing or field edited by a reviewer"})
                continue
            value = item.get("value")
            canon = _canonical(req["column"], value) if value not in (None, "", "null") else None
            if canon is None:
                log["rejected"].append({"field_id": fid, "reason": "no readable value proposed"})
                continue
            comparable = canon if req["column"] != "Transaction Date*" else parse_date(canon).value.isoformat()
            if req["allowed_candidates"] and comparable not in req["allowed_candidates"] and canon not in req["allowed_candidates"]:
                log["rejected"].append({"field_id": fid, "reason": f"'{value}' is not one of the evidenced candidates"})
                continue
            before = row.get(req["column"])
            row[req["column"]] = canon
            row["_status"] = "NEEDS_REVIEW"
            ev = (data.setdefault("row_evidence", {}).setdefault(req["row_id"], {"fields": {}, "reasons": []}))
            ev.setdefault("fields", {}).setdefault(req["column"], {})["repair"] = {
                "before": before, "after": canon, "evidence": str(item.get("evidence") or "")[:300],
                "note": "repair proposal; confirm against the source"}
            log["applied"].append({"field_id": fid, "before": before, "after": canon})
        data["repair_status"] = "APPLIED" if log["applied"] else "NO_CHANGE"
        data["repair_log"] = log
        return data
