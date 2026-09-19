"""Failure injection: every external dependency failing must degrade safely
(never PASSED, never fabricated, never an HTTP 500 for the upload)."""

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from app.agents.extraction_pipeline import DocumentExtractionPipeline
from app.documents.ocr import DocumentOcrService
from app.documents.ocr_orchestrator import OcrOrchestrator
from app.google.sheets_service import SheetsWriteQueue
from tests.conftest import png_bytes
from tests.test_accounting_workflow import FakeSheetsService, client_for, headers, records, settings, start_job, token


class DeadEngine:
    name = "rapidocr"
    unavailable_reason = "injected: engine failed to start"

    def available(self):
        return False

    def read(self, *a, **k):
        raise AssertionError("must not be called")


class FakeGemini:
    def __init__(self, outcome):
        self.outcome = outcome
        self.settings = replace(settings(), gemini_api_key="k")
        self.calls = 0

    def generate_json(self, parts, **kwargs):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class WorkingEngine:
    name = "rapidocr"
    unavailable_reason = ""

    def available(self):
        return True

    def read(self, image, page_idx=None, variant=None):
        from app.documents.ocr_contract import OcrLineEvidence, OcrResult
        lines = [OcrLineEvidence(f"p1.rapidocr.{variant}.0", "rapidocr", variant or "canonical", 1, "TOTAL 8700",
                                 0.99, (10, 10, 120, 30))]
        return OcrResult("rapidocr", 1, variant or "canonical", image.width, image.height, lines)


def _doc_png():
    img = Image.new("RGB", (400, 200), "white")
    ImageDraw.Draw(img).text((10, 10), "TOTAL 8700", fill="black")
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_engine_unavailable_fails_the_job_cleanly_not_500():
    from app.agents.extractor import ExtractionAgent, GeminiExtractionProvider
    from app.services.gemini_client import GeminiDocumentClient

    client, app, drive, _ = client_for()
    dead = DocumentOcrService(None, OcrOrchestrator([DeadEngine()]))
    app.state.extraction_agent.provider = GeminiExtractionProvider(GeminiDocumentClient(settings()), dead)
    session = token(client)
    response = start_job(client, session)
    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "FAILED"
    stage = next(p for p in job["progress"] if p["stage"] == "OCR_AND_EXTRACTION")
    assert stage["status"] == "FAILED"
    assert "OCR engine" in job["last_error"] or "Document reading is unavailable" in job["last_error"]
    assert client.post(f"/processing/jobs/{job['job_id']}/approve", headers=headers(session)).status_code == 409


def test_every_sheets_call_failing_never_breaks_upload_or_processing():
    class BrokenSheets(FakeSheetsService):
        def append_row(self, *a, **k):
            raise RuntimeError("APIError: [429]: Quota exceeded")

        def update_row_by_key(self, *a, **k):
            raise RuntimeError("APIError: [429]: Quota exceeded")

    sheets = BrokenSheets(records())
    sheets.write_queue = SheetsWriteQueue(synchronous=True)
    from app.app_factory import create_app
    from fastapi.testclient import TestClient
    from tests.test_accounting_workflow import (FakeDriveService, FakeVerifier, MEMBER_DATA, VENDOR_DATA,
                                                StaticExtractionProvider, StaticRepairProvider,
                                                StaticVerificationProvider)
    app = create_app(settings=settings(), sheets_service=sheets, drive_service=FakeDriveService(),
                     google_token_verifier=FakeVerifier(),
                     extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA,
                                                                   "VENDOR_INVOICE": VENDOR_DATA}),
                     verification_provider=StaticVerificationProvider(), repair_provider=StaticRepairProvider({}))
    client = TestClient(app)
    session = token(client)
    response = start_job(client, session)
    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "NEEDS_REVIEW"          # in-memory state is authoritative
    assert sheets.write_queue.failures > 0                   # failures were recorded, not raised
    approved = client.post(f"/processing/jobs/{job['job_id']}/approve", headers=headers(session))
    assert approved.status_code == 200 and approved.json()["overall_status"] == "COMPLETED"


def test_write_queue_retries_quota_then_succeeds(monkeypatch):
    queue = SheetsWriteQueue(synchronous=True)
    monkeypatch.setattr("app.google.sheets_service.time.sleep", lambda s: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("[429]: Quota exceeded for quota metric")
        return True

    assert queue._execute(flaky, retry=True) is True and len(attempts) == 3


@pytest.mark.parametrize("outcome, expected_provider, expected_outcome", [
    (TimeoutError("read timed out"), "LOCAL_OCR", None),
    ("definitely not json", "LOCAL_OCR", "EXTRACTION_CONTRACT_VIOLATION"),
    ({"Payment Type*": "Cash", "Amount*": "900"}, "LOCAL_OCR", "EXTRACTION_CONTRACT_VIOLATION"),  # flat dict
    ([1, 2, 3], "LOCAL_OCR", "EXTRACTION_CONTRACT_VIOLATION"),
    ({"rows": []}, "GEMINI", "NO_RELIABLE_TRANSACTIONS"),                                       # empty result
])
def test_gemini_failure_modes_degrade_without_fabrication(outcome, expected_provider, expected_outcome):
    gemini = FakeGemini(outcome)
    ocr = DocumentOcrService(None, OcrOrchestrator([WorkingEngine()]))
    result = DocumentExtractionPipeline(gemini, ocr).extract(_doc_png(), "PETTY_CASH_REGISTER")
    assert result["_extraction_provider"] == expected_provider
    if expected_outcome:
        assert result["extraction_outcome"] == expected_outcome
    assert all(r["_status"] != "ACCEPTED" for r in result["rows"])   # nothing unverified is accepted
    assert result["candidate_ledger"]["balanced"]
    assert gemini.calls >= 1


def test_zero_rows_from_gemini_is_never_an_approvable_empty_workbook():
    gemini = FakeGemini({"rows": []})
    ocr = DocumentOcrService(None, OcrOrchestrator([WorkingEngine()]))
    from app.accounting.document_result import finalize_extraction
    from app.accounting.validation import AccountingValidationService
    data = finalize_extraction("PETTY_CASH_REGISTER",
                               DocumentExtractionPipeline(gemini, ocr).extract(_doc_png(), "PETTY_CASH_REGISTER"))
    result = AccountingValidationService().validate("PETTY_CASH_REGISTER", data)
    assert result.status == "BLOCKED" and any(i.code == "NO_TRANSACTIONS" for i in result.issues)


def test_production_without_a_detectable_memory_limit_never_loads_paddleocr(monkeypatch):
    """Loading PaddleOCR crashed the Render instance twice (limit not exposed via cgroups)."""
    from app.core import resources
    monkeypatch.setattr(resources, "container_memory_limit_mb", lambda: None)
    monkeypatch.delenv("OCR_ENSEMBLE", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    profile = resources.ocr_profile()
    assert profile["ensemble"] == "off" and "not detectable" in profile["reasons"][0]
    monkeypatch.setenv("OCR_ENSEMBLE", "auto")          # an operator can opt in explicitly
    assert resources.ocr_profile()["ensemble"] == "auto"


def test_small_container_disables_the_ensemble_and_lowers_dpi(monkeypatch):
    from app.core import resources
    monkeypatch.setattr(resources, "container_memory_limit_mb", lambda: 900)
    monkeypatch.delenv("OCR_ENSEMBLE", raising=False)
    monkeypatch.delenv("OCR_CANONICAL_DPI", raising=False)
    profile = resources.ocr_profile()
    assert (profile["ensemble"], profile["dpi"]) == ("off", 150)


def test_undersized_instance_refuses_uploads_instead_of_crashing(monkeypatch):
    """Production has 512 MB: loading OCR would OOM-kill it mid-job."""
    from app.core import resources
    monkeypatch.setattr(resources, "container_memory_limit_mb", lambda: 512)
    client, app, drive, _ = client_for()
    response = start_job(client, token(client))
    assert response.status_code == 503 and "1024 MB" in response.json()["detail"]
    assert drive.uploads == []                                   # nothing half-done
    from app.documents.capabilities import collect_capabilities
    report = collect_capabilities(settings(), ocr_service=app.state.ocr_service)
    assert report["capabilities"]["rapidocr"]["detail"].startswith("not probed")
    assert report["overall"] == "NOT_READY" and "memory" in report["missing_required"]
