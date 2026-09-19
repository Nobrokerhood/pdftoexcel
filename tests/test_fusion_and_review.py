"""Evidence fusion decisions, candidate ledger, review actions, persistence."""

import ast
import pathlib

import pytest

from app.accounting import review
from app.accounting.fusion import (
    ARBITRATED, CONFLICT, GEMINI_CROP, GEMINI_PAGE, SINGLE_SOURCE, VERIFIED, SourceVote, decide_field,
)
from app.processing.jobs import JobRunner, ProcessingJob
from app.processing.stores import MAX_CELL_CHARS, decode_state, encode_state, job_from_state, job_to_state

AMT = "Amount*"


def v(source, value, raw=None):
    return SourceVote(source, value, raw or value)


def test_two_independent_ocr_engines_verify():
    d = decide_field(AMT, None, [v("rapidocr", "900"), v("paddleocr", "900")])
    assert (d.status, d.value) == (VERIFIED, "900")


def test_page_read_anchored_to_ocr_is_not_independent():
    """Gemini was shown RapidOCR's text; agreeing with it is not corroboration."""
    d = decide_field("Cheque/Ref No*", "206", [v("rapidocr", "206"), v(GEMINI_PAGE, "206")], anchored_to="rapidocr")
    assert d.status == SINGLE_SOURCE


def test_blind_crop_resolves_single_engine_disagreement():
    votes = [v("rapidocr", "1787"), v("paddleocr", "189"), v(GEMINI_PAGE, "1789"), v(GEMINI_CROP, "1789")]
    d = decide_field(AMT, "1789", votes, anchored_to="rapidocr")
    # rapidocr group {1787, 1789(page)} and crop read 1789; paddleocr dissents alone
    assert d.value == "1789" and d.status == ARBITRATED


def test_two_ocr_engines_outrank_a_crop_and_stay_in_conflict():
    """307-style: both engines read 150, the blind crop reads 750 -> flagged, never silently decided."""
    votes = [v("rapidocr", "150"), v("paddleocr", "150"), v(GEMINI_PAGE, "150"), v(GEMINI_CROP, "750")]
    d = decide_field(AMT, "150", votes, anchored_to="rapidocr")
    assert d.status == CONFLICT and "750" in d.alternatives


def test_confidence_never_picks_a_value():
    votes = [v("rapidocr", "900"), v("paddleocr", "9001")]
    d = decide_field(AMT, None, votes)
    assert d.status == CONFLICT


def test_digital_text_is_source_truth():
    d = decide_field(AMT, "14759", [v("pdf_text", "14759")], digital_text="41624121414759")
    assert d.status == VERIFIED


# ---- review actions and the ledger ----------------------------------------------

def _doc():
    return {
        "rows": [{"Amount*": "900", "Transaction Date*": "04-07-2025", "_row_id": "r1_005", "_status": "NEEDS_REVIEW"}],
        "candidate_ledger": {"candidates": [
            {"candidate_id": "c1_005", "status": "NEEDS_REVIEW", "row_id": "r1_005"},
            {"candidate_id": "c1_010", "status": "UNRESOLVED", "row_id": None},
            {"candidate_id": "c1_020", "status": "NON_TRANSACTION", "row_id": None},
        ]},
    }


def test_confirm_edit_dismiss_promote_keep_the_ledger_balanced():
    doc = _doc()
    entries = review.edit_row(doc, "r1_005", {"Amount*": "Rs. 1,800/-"}, "a@x", "source shows 1800")
    assert doc["rows"][0]["Amount*"] == "1800" and doc["rows"][0]["_status"] == "USER_CONFIRMED"
    assert doc["rows"][0]["_edited_fields"] == ["Amount*"]
    assert {e.action for e in entries} == {"EDIT", "CONFIRM"} and entries[0].old_value == "900"
    with pytest.raises(review.ReviewError):
        review.dismiss_candidate(doc, "c1_010", "a@x", None)          # reason required
    review.promote_candidate(doc, "c1_010", {"Amount*": "300", "Transaction Date*": "07-07-2025"}, "a@x", "missed row")
    ledger = doc["candidate_ledger"]
    assert ledger["balanced"] and ledger["unresolved"] == 0 and ledger["accepted"] == 2
    assert [r["_row_id"] for r in doc["rows"]] == ["r1_005", "r1_010"]


def test_delete_row_records_rejection_with_reason():
    doc = _doc()
    review.delete_row(doc, "r1_005", "a@x", "duplicate of r1_006")
    cand = doc["candidate_ledger"]["candidates"][0]
    assert cand["status"] == "REJECTED_WITH_REASON" and "duplicate" in cand["status_reason"]
    assert doc["candidate_ledger"]["balanced"]


def test_grid_save_matches_rows_by_id_not_position():
    doc = {"rows": [{"Amount*": "1", "Transaction Date*": "04-07-2025", "_row_id": "a", "_status": "ACCEPTED"},
                    {"Amount*": "2", "Transaction Date*": "04-07-2025", "_row_id": "b", "_status": "ACCEPTED"}]}
    submitted = [dict(doc["rows"][1], **{"Amount*": "20"}), dict(doc["rows"][0])]   # reordered
    review.apply_row_list(doc, submitted, "a@x")
    by_id = {r["_row_id"]: r for r in doc["rows"]}
    assert by_id["a"]["Amount*"] == "1" and by_id["b"]["Amount*"] == "20"


# ---- persistence ------------------------------------------------------------------

def test_large_job_state_round_trips_without_truncation():
    job = ProcessingJob("j1", "s", "u@x", "PETTY_CASH_REGISTER", "T", "f.pdf", "application/pdf", b"")
    job.extracted_data = {"rows": [{"Comments": f"row {i} " + "x" * 400, "_row_id": f"r{i}"} for i in range(400)]}
    state = job_to_state(job)
    raw_len = len(str(state))
    encoded = encode_state(state)
    assert raw_len > MAX_CELL_CHARS          # the old store truncated this into a stub
    restored = job_from_state(decode_state(encoded))
    assert restored.extracted_data == job.extracted_data


def test_background_runner_returns_before_work_finishes():
    import threading
    started, release = threading.Event(), threading.Event()
    runner = JobRunner("background", 1)

    def work():
        started.set()
        release.wait(5)

    runner.submit("j", work)
    assert started.wait(5) and runner.is_active("j")
    release.set()


# ---- architecture ------------------------------------------------------------------

def test_no_module_outside_the_orchestration_layer_instantiates_an_ocr_engine():
    allowed = {"ocr_orchestrator.py", "ocr_engines.py", "capabilities.py"}
    offenders = []
    for path in pathlib.Path("app").rglob("*.py"):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) in (
                    "RapidOcrProvider", "PaddleOcrProvider", "RapidOCR", "PaddleOCR"):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
