"""Golden corpus: real OCR on the benchmark documents (skipped when the
untracked testing_folder is absent). Ground truth lives only here.

These run the deterministic evidence path (no Gemini) so they are
reproducible offline; they prove the zero-row-loss guarantees that do not depend
on the model. Model-dependent accuracy is measured by scripts/golden_benchmark.py.
"""

import os
from dataclasses import replace
from decimal import Decimal

import pytest

from app.accounting.source_candidates import TRANSACTION, amount_readings, build_candidates, ref_readings
from app.core.config import get_settings

CORPUS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testing_folder")
HANDWRITTEN = os.path.join(CORPUS, "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf")

TRUTH = [(266, 900), (267, 140), (268, 700), (269, 130), (270, 1800), (271, 300), (274, 200), (284, 10000),
         (285, 8400), (286, 9000), (287, 10000), (288, 9400), (289, 12000), (290, 10000), (291, 880), (292, 200),
         (296, 1090), (297, 2250), (298, 5000), (299, 2000), (300, 2500), (301, 2400), (302, 1789), (303, 1000),
         (304, 3000), (305, 200), (306, 200), (307, 750), (308, 200)]

pytestmark = pytest.mark.skipif(not os.path.exists(HANDWRITTEN), reason="golden corpus not present")


@pytest.fixture(scope="module")
def ocr_service():
    from app.documents.ocr import DocumentOcrService
    return DocumentOcrService(get_settings().poppler_path)


@pytest.fixture(scope="module")
def handwritten(ocr_service):
    from app.documents.ingestion import build_manifest
    data = open(HANDWRITTEN, "rb").read()
    return data, ocr_service.represent(data, build_manifest(data, "hw.pdf", "application/pdf", get_settings().poppler_path))


def test_ocr_evidence_retains_every_engine_and_variant(handwritten):
    _, rep = handwritten
    page = rep.pages[0]
    engines = {(r.engine, r.variant) for r in page.evidence}
    assert ("rapidocr", "canonical") in engines
    assert len(engines) >= 2, "a difficult page must keep alternative OCR evidence"
    assert page.routing["difficulty"] == "DIFFICULT" and page.routing["selected"]
    assert all(r.width == page.width and r.height == page.height for r in page.evidence)  # one coordinate space
    assert rep.page_image(1).size == (page.width, page.height)                           # Gemini sees the same image


def test_every_genuine_handwritten_row_exists_as_a_source_candidate(handwritten):
    """The old date-anchored builder lost 270, 299, 300, 304, 306, 308 (and refs of 284/287/296)."""
    _, rep = handwritten
    candidates, layouts = build_candidates(rep)
    tx = sorted([c for c in candidates if c.classification == TRANSACTION and ref_readings(c)],
                key=lambda c: c.bbox[1])
    assert len(tx) == 29, [c.candidate_id for c in tx]
    misses = []
    for cand, (ref, amount) in zip(tx, TRUTH):
        refs = {d for d, _ in ref_readings(cand)}
        amounts = {v for v, _ in amount_readings(cand)}
        if str(ref) not in refs and Decimal(amount) not in amounts:
            misses.append((ref, sorted(refs), sorted(map(str, amounts))))
    assert misses == [], misses                 # each row is identified by its ref or its amount
    assert layouts[0].tabular


def test_degraded_mode_exports_all_29_rows_with_a_balanced_ledger(handwritten, ocr_service):
    """No Gemini at all: rows still come from source evidence, all flagged for review."""
    from app.agents.extraction_pipeline import DocumentExtractionPipeline
    from app.services.gemini_client import GeminiDocumentClient

    data, _ = handwritten
    client = GeminiDocumentClient(replace(get_settings(), gemini_api_key=None))
    result = DocumentExtractionPipeline(client, ocr_service).extract(data, "PETTY_CASH_REGISTER")
    ledger = result["candidate_ledger"]
    assert ledger["balanced"] and not ledger["unaccounted"]
    assert sum(ledger[k] for k in ("accepted", "needs_review", "rejected_with_reason", "non_transaction",
                                   "unresolved")) == ledger["source_candidates"]
    rows = sorted(result["rows"], key=lambda r: result["row_evidence"][r["_row_id"]]["bbox"][1])
    expense_rows = [r for r in rows if r["Cheque/Ref No*"] != "-"]
    assert len(expense_rows) == 29
    misses = []
    for row, (ref, amount) in zip(expense_rows, TRUTH):
        fields = result["row_evidence"][row["_row_id"]]["fields"]
        seen_refs = {v["value"] for v in fields["Cheque/Ref No*"]["votes"]}
        seen_amounts = {v["value"] for v in fields["Amount*"]["votes"]}
        if str(ref) not in seen_refs and str(amount) not in seen_amounts:
            misses.append(ref)
    assert misses == []            # every genuine row, incl. the previously lost 270/299/300/304/306/308
    assert all(r["_status"] == "NEEDS_REVIEW" for r in result["rows"])   # OCR-only rows always go to a human
    assert all(r["_status"] != "ACCEPTED" or r["_row_id"] for r in result["rows"])
    assert result["_extraction_provider"] == "LOCAL_OCR"


def test_low_quality_labour_bill_produces_no_fake_rows(ocr_service):
    path = os.path.join(CORPUS, "unnamed.jpg")
    if not os.path.exists(path):
        pytest.skip("unnamed.jpg absent")
    from app.agents.extraction_pipeline import DocumentExtractionPipeline
    from app.services.gemini_client import GeminiDocumentClient

    client = GeminiDocumentClient(replace(get_settings(), gemini_api_key=None))
    result = DocumentExtractionPipeline(client, ocr_service).extract(open(path, "rb").read(), "VENDOR_INVOICE")
    for row in result["rows"]:
        assert row["Amount*"] != "-" or row["_status"] != "ACCEPTED"
    assert all(r["_status"] != "ACCEPTED" for r in result["rows"])
    assert result["candidate_ledger"]["balanced"]


def test_digital_pdf_rows_are_candidates_with_real_geometry(ocr_service):
    path = os.path.join(CORPUS, "sample-radhakrishna.pdf")
    if not os.path.exists(path):
        pytest.skip("radhakrishna absent")
    from app.documents.ingestion import build_manifest
    data = open(path, "rb").read()
    rep = ocr_service.represent(data, build_manifest(data, "rk.pdf", "application/pdf", get_settings().poppler_path))
    candidates, _ = build_candidates(rep)
    tx = [c for c in candidates if c.classification == TRANSACTION]
    names = " ".join(c.all_text() for c in tx)
    assert 20 <= len(tx) <= 22, len(tx)          # A1..A20 (+ at most header/total bands, never splits)
    assert "Taikar" in names
