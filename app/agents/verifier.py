"""Row-identity-based verification.

Rules:
* Every result is attributed by immutable `row_id`. Position, batch index and
  free-form field names are never used to decide which row a result is about.
* A row with fusion evidence (multi-OCR + blind visual arbitration) takes its
  verification from that evidence. A model result may DOWNGRADE such a row
  (a MISMATCH), never upgrade a row the evidence left in conflict.
* Rows without evidence (rows supplied directly by a provider) are verified by
  a Gemini prompt keyed by row_id. Rows the response does not cover, or covers
  ambiguously, are UNVERIFIED.
* A missing or unknown overall status is NEEDS_REVIEW. There is no default PASSED.
* Verification unavailable (no key, quota, timeout) -> NEEDS_REVIEW with the
  reason; never PASSED.
"""

import json
import logging
from dataclasses import dataclass
from typing import Protocol

from app.accounting.candidates import is_placeholder, row_is_meaningful
from app.accounting.schemas import VerificationResult
from app.accounting.templates import TemplateDefinition
from app.documents.ocr import DocumentOcrService, DocumentRepresentation

logger = logging.getLogger(__name__)

ROW_VERIFIED = "VERIFIED"
ROW_NEEDS_REVIEW = "NEEDS_REVIEW"
ROW_UNVERIFIED = "UNVERIFIED"
MODEL_BATCH = 15


@dataclass(frozen=True)
class VerificationRouteDecision:
    mode: str  # "MULTIMODAL" or "OCR_STRUCTURED"
    reason: str
    is_handwritten_or_uncertain: bool


def determine_verification_route(rep: DocumentRepresentation | None, extracted_data: dict, has_images: bool) -> VerificationRouteDecision:
    """Whether the model must see page images (it must for anything not clean print)."""
    if not has_images:
        return VerificationRouteDecision("OCR_STRUCTURED", "Page images unavailable; using structured OCR text verification.", False)
    if rep is None or not rep.all_lines:
        return VerificationRouteDecision("MULTIMODAL", "OCR text absent or empty; visual multimodal verification required.", True)
    doc_type = str(extracted_data.get("document_type") or "").upper()
    if doc_type in ("PETTY_CASH_REGISTER", "HANDWRITTEN", "VOUCHER", "OTHER"):
        return VerificationRouteDecision("MULTIMODAL", f"Document classified as {doc_type}; visual inspection required.", True)
    if rep.warnings:
        return VerificationRouteDecision("MULTIMODAL", f"OCR generated warnings ({'; '.join(rep.warnings[:2])}); visual verification required.", True)
    mean_conf = rep.mean_confidence
    low = sum(1 for line in rep.all_lines if getattr(line, "confidence", 1.0) < 0.85) / max(1, len(rep.all_lines))
    if mean_conf < 0.92 or low > 0.12:
        return VerificationRouteDecision("MULTIMODAL", f"Sub-optimal OCR quality (mean confidence={mean_conf:.2f}, low conf ratio={low:.1%}); visual verification required.", True)
    for page in rep.pages:
        if getattr(page, "script", "PRINTED") != "PRINTED":
            return VerificationRouteDecision("MULTIMODAL", f"Page {page.page_number} detected non-printed script ({page.script}); visual verification required.", True)
    return VerificationRouteDecision("OCR_STRUCTURED", f"Clean digital document (mean confidence={mean_conf:.2f}, {len(rep.all_lines)} lines).", False)


class VerificationProvider(Protocol):
    def verify(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, extracted_data: dict) -> dict:
        ...


def _row_ids(extracted_data: dict) -> list[str]:
    return [str(r.get("_row_id")) for r in extracted_data.get("rows") or [] if isinstance(r, dict) and r.get("_row_id")]


class GeminiVerificationProvider:
    """Verifies rows that carry no fusion evidence, keyed by row_id."""

    def __init__(self, gemini_client, ocr_service: DocumentOcrService | None = None):
        self.gemini_client = gemini_client
        self._ocr_service = ocr_service

    def verify(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, extracted_data: dict) -> dict:
        evidence = extracted_data.get("row_evidence") or {}
        rows = [r for r in extracted_data.get("rows") or [] if isinstance(r, dict)]
        pending = [r for r in rows if r.get("_row_id") not in evidence and r.get("_status") != "USER_CONFIRMED"]
        result = {"overall_status": "NEEDS_REVIEW", "fields": [], "rows": {}, "provider": "EVIDENCE",
                  "method": "Fusion evidence (multi-OCR + blind visual arbitration) per row_id", "notes": []}
        if not pending:
            return result
        if not self.gemini_client.settings.gemini_api_key:
            result["notes"].append("Model verification unavailable (no API key); rows without evidence are UNVERIFIED.")
            return result
        try:
            if self._ocr_service is not None:
                from app.documents.ingestion import build_manifest
                rep = self._ocr_service.represent(
                    source_bytes, build_manifest(source_bytes, "source", "application/octet-stream",
                                                 self.gemini_client.settings.poppler_path))
                images = rep.page_images()
            else:
                from app.agents.extractor import source_parts
                images = source_parts(source_bytes, self.gemini_client.settings.poppler_path)
        except Exception as exc:
            result["notes"].append(f"Source could not be rendered for verification ({type(exc).__name__}).")
            return result

        result["provider"] = "GEMINI"
        result["method"] = "Gemini row_id-keyed verification against the page images"
        statuses = []
        for start in range(0, len(pending), MODEL_BATCH):
            batch = pending[start:start + MODEL_BATCH]
            payload = [{"row_id": r["_row_id"], **{k: v for k, v in r.items() if not k.startswith("_") and k != "USER_EDITED"}}
                       for r in batch]
            prompt = (
                "You are an independent accounting verification agent. For EACH row below, check every value "
                "against the page image. Never use arithmetic to decide a handwritten digit.\n"
                "Return JSON only: {\"overall_status\": \"PASSED\"|\"NEEDS_REVIEW\", \"rows\": [{\"row_id\": \"<exactly as "
                "given>\", \"status\": \"VERIFIED\"|\"MISMATCH\"|\"NOT_FOUND\"|\"UNCERTAIN\", \"fields\": [{\"column\": "
                "\"<column>\", \"verified_value\": \"<value in source or null>\", \"status\": \"VERIFIED\"|\"MISMATCH\"|"
                "\"NOT_FOUND\"|\"UNCERTAIN\", \"evidence\": \"<short quote>\"}]}]}\n"
                f"Purpose: {purpose}\nROWS: {json.dumps(payload, default=str, ensure_ascii=False)}"
            )
            try:
                raw = self.gemini_client.generate_json([prompt, *images], purpose="verification")
            except Exception as exc:
                result["notes"].append(f"Model verification failed for {len(batch)} row(s): {exc}; they stay UNVERIFIED.")
                if getattr(exc, "__class__", None).__name__ in ("GeminiSpendCapError", "GeminiQuotaExhaustedError",
                                                               "GeminiAuthenticationError", "GeminiFallbackUnavailableError"):
                    break  # the next batch would fail identically; do not spend more calls
                continue
            if not isinstance(raw, dict):
                result["notes"].append("Model verification response was not an object; rows stay UNVERIFIED.")
                continue
            statuses.append(raw.get("overall_status"))
            result["fields"] += [dict(item, _batch_ids=[r["_row_id"] for r in batch]) for item in raw.get("rows") or []
                                 if isinstance(item, dict)]
        return result


class VerificationAgent:
    def __init__(self, provider: VerificationProvider):
        self.provider = provider

    def verify(self, source_bytes: bytes, purpose: str, template: TemplateDefinition, extracted_data: dict) -> VerificationResult:
        raw = self.provider.verify(source_bytes, purpose, template, extracted_data)
        return adapt_verification(raw if isinstance(raw, dict) else {}, extracted_data)


_ROW_STATES = {"VERIFIED", "MISMATCH", "NOT_FOUND", "UNCERTAIN"}


def adapt_verification(raw: dict, extracted_data: dict) -> VerificationResult:
    """Strict adapter: provider output -> row_id-attributed VerificationResult."""
    rows = [r for r in extracted_data.get("rows") or [] if isinstance(r, dict)]
    ids = _row_ids(extracted_data)
    known = set(ids)
    evidence = extracted_data.get("row_evidence") or {}
    notes = list(raw.get("notes") or [])
    fields_out: list[dict] = []
    row_state: dict[str, str] = {}

    # 1. Model results, attributed by row_id only.
    model_state: dict[str, str] = {}
    entries = list(raw.get("fields") or [])
    if isinstance(raw.get("rows"), list):
        entries += raw["rows"]
    for item in entries:
        if not isinstance(item, dict):
            continue
        rid = item.get("row_id")
        if rid is None:
            continue  # unattributable: a result without row_id verifies nothing
        rid = str(rid)
        if rid not in known:
            notes.append(f"Ignored a verification result for unknown row_id '{rid}'.")
            continue
        status = str(item.get("status", "")).upper()
        if status not in _ROW_STATES:
            status = "UNCERTAIN"
        if rid in model_state and model_state[rid] != status:
            status = "UNCERTAIN"  # ambiguous: two different verdicts for one row
        model_state[rid] = status
        for f in item.get("fields") or [{}]:
            if not isinstance(f, dict):
                continue
            f_status = str(f.get("status", status)).upper()
            fields_out.append({
                "field": f"{rid}.{f.get('column') or '*'}", "row_id": rid, "column": f.get("column"),
                "extracted_value": f.get("extracted_value"), "verified_value": f.get("verified_value"),
                "status": f_status if f_status in _ROW_STATES else "UNCERTAIN",
                "confidence": float(f.get("confidence") or 0) if isinstance(f.get("confidence"), (int, float)) else 0.0,
                "evidence": str(f.get("evidence") or "")[:300], "page_number": f.get("page_number"),
            })
    if isinstance(raw.get("rows"), dict):  # compact {row_id: status} form
        for rid, status in raw["rows"].items():
            if str(rid) in known:
                model_state[str(rid)] = str(status).upper() if str(status).upper() in _ROW_STATES else "UNCERTAIN"

    # 2. Combine with evidence and reviewer state.
    for row in rows:
        rid = str(row.get("_row_id"))
        ev = evidence.get(rid)
        model = model_state.get(rid)
        if row.get("_status") == "USER_CONFIRMED":
            state = ROW_VERIFIED
        elif not row_is_meaningful(row):
            state = ROW_NEEDS_REVIEW
            notes.append(f"{rid}: no accounting value in any column; it cannot be a verified transaction.")
        elif ev is not None:
            state = ROW_VERIFIED if ev.get("status") == "ACCEPTED" else ROW_NEEDS_REVIEW
            if model in ("MISMATCH", "NOT_FOUND") and state == ROW_VERIFIED:
                state = ROW_NEEDS_REVIEW
                notes.append(f"{rid}: model verification reported {model}; downgraded for review.")
        elif model == "VERIFIED":
            state = ROW_VERIFIED
        elif model in ("MISMATCH", "NOT_FOUND", "UNCERTAIN"):
            state = ROW_NEEDS_REVIEW
        else:
            state = ROW_UNVERIFIED
        row_state[rid] = state
        fields_out.append({
            "field": rid, "row_id": rid, "column": None, "extracted_value": row.get("Amount*"),
            "verified_value": None,
            "status": {"VERIFIED": "VERIFIED", "NEEDS_REVIEW": "UNCERTAIN", "UNVERIFIED": "UNVERIFIED"}[state],
            "confidence": 0.0, "evidence": "; ".join((ev or {}).get("reasons") or []) or
            ("fusion evidence accepted" if ev else (f"model: {model}" if model else "no verification covered this row")),
            "page_number": (ev or {}).get("page"),
        })

    ledger = extracted_data.get("candidate_ledger") or {}
    unresolved = [c for c in ledger.get("candidates") or [] if c.get("status") == "UNRESOLVED"]
    overall = "PASSED"
    if not rows:
        overall = "NEEDS_REVIEW"
        notes.append("No transaction rows: nothing can be verified.")
    elif any(s != ROW_VERIFIED for s in row_state.values()):
        overall = "NEEDS_REVIEW"
    if ledger and not ledger.get("balanced", True):
        overall = "NEEDS_REVIEW"
        notes.append("Candidate ledger is not balanced.")
    if unresolved:
        overall = "NEEDS_REVIEW"
        notes.append(f"{len(unresolved)} unresolved source region(s) may be missing transactions.")
    declared = str(raw.get("overall_status") or "").upper()
    if declared in ("FAILED", "NEEDS_REVIEW") and overall == "PASSED":
        overall = "NEEDS_REVIEW"
    return VerificationResult(
        overall_status=overall, fields=fields_out, provider=str(raw.get("provider") or "EVIDENCE"),
        method=str(raw.get("method") or ""), rows=row_state, notes=notes,
    )


def apply_verification(extracted_data: dict, result: VerificationResult) -> None:
    """Write row verification into row _status. Reviewer confirmations are never overwritten."""
    for row in extracted_data.get("rows") or []:
        rid = str(row.get("_row_id"))
        if row.get("_status") == "USER_CONFIRMED":
            continue
        state = result.rows.get(rid)
        if state == ROW_VERIFIED:
            row["_status"] = "ACCEPTED"
        elif state == ROW_NEEDS_REVIEW:
            row["_status"] = "NEEDS_REVIEW"
        else:
            row["_status"] = "UNVERIFIED"


def enforce_verification_guards(result: dict, extracted_data: dict) -> dict:
    """Compatibility wrapper: the guards are now inside `adapt_verification`."""
    return adapt_verification(result, extracted_data).model_dump(mode="json")


__all__ = ["GeminiVerificationProvider", "VerificationAgent", "VerificationResult", "adapt_verification",
           "apply_verification", "enforce_verification_guards", "determine_verification_route",
           "VerificationRouteDecision", "is_placeholder"]
