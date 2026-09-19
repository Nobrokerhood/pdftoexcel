import io
import os
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PyPDF2 import PdfWriter

import app.core.config  # noqa: F401  (loads .env so a configured POPPLER_PATH is visible)
from app.agents.extractor import source_parts
from app.app_factory import create_app
from app.documents import pdf_images
from app.documents.pdf_images import (
    INVALID_PDF_MESSAGE,
    PDF_DEPENDENCY_MISSING_MESSAGE,
    InvalidPdfError,
    PdfDependencyMissingError,
    pdf_page_count,
    pdf_to_images,
    poppler_available,
)
from app.google.drive_service import GoogleDriveError
from tests.test_accounting_workflow import (
    MEMBER_DATA,
    FakeDriveService,
    FakeSheetsService,
    FakeVerifier,
    StaticExtractionProvider,
    StaticRepairProvider,
    StaticVerificationProvider,
    headers,
    records,
    settings,
    token,
)


def local_poppler_path():
    configured = os.getenv("POPPLER_PATH") or None
    return configured if poppler_available(configured) else None


requires_poppler = pytest.mark.skipif(
    not poppler_available(os.getenv("POPPLER_PATH") or None),
    reason="Poppler (pdfinfo/pdftoppm) is not installed on this machine.",
)


def sample_pdf(pages=2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=400)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


INVALID_PDF = b"%PDF-1.4\nthis is not a real pdf body\n%%EOF"


@pytest.fixture
def missing_poppler_dir():
    return os.path.join(os.path.dirname(__file__), "__poppler_not_installed__")


def test_poppler_detection_uses_configured_directory(monkeypatch):
    present = {os.path.join("poppler-bin", "pdfinfo.exe"), os.path.join("poppler-bin", "pdftoppm.exe")}
    monkeypatch.setattr(pdf_images.os.path, "isfile", lambda path: path in present)
    assert poppler_available("poppler-bin")

    present.discard(os.path.join("poppler-bin", "pdftoppm.exe"))
    assert not poppler_available("poppler-bin")


def test_poppler_detection_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(pdf_images.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    assert poppler_available(None)

    monkeypatch.setattr(pdf_images.shutil, "which", lambda tool: None)
    assert not poppler_available(None)


def test_missing_poppler_raises_actionable_error(missing_poppler_dir):
    with pytest.raises(PdfDependencyMissingError) as raised:
        pdf_to_images(sample_pdf(), missing_poppler_dir, dpi=72)

    assert raised.value.code == "PDF_PROCESSING_DEPENDENCY_MISSING"
    assert str(raised.value) == PDF_DEPENDENCY_MISSING_MESSAGE


@requires_poppler
def test_real_pdf_converts_to_page_images():
    images = pdf_to_images(sample_pdf(pages=2), local_poppler_path(), dpi=72)

    assert len(images) == 2
    assert all(image.mode == "RGB" and image.size[0] > 0 for image in images)
    assert pdf_page_count(sample_pdf(pages=3), local_poppler_path()) == 3


@requires_poppler
def test_invalid_pdf_is_reported_as_invalid():
    with pytest.raises(InvalidPdfError) as raised:
        pdf_page_count(INVALID_PDF, local_poppler_path())
    assert str(raised.value) == INVALID_PDF_MESSAGE


@requires_poppler
def test_extraction_source_parts_render_pdf_pages():
    assert len(source_parts(sample_pdf(pages=1), local_poppler_path())) == 1


def test_source_parts_still_handle_png_without_poppler(missing_poppler_dir):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (5, 5), "white").save(buf, format="PNG")

    parts = source_parts(buf.getvalue(), missing_poppler_dir)

    assert len(parts) == 1 and parts[0].size == (5, 5)


class FailingDriveService(FakeDriveService):
    def upload_file(self, *args, **kwargs):
        raise GoogleDriveError("403 storageQuotaExceeded: raw Google detail")


def processing_client(poppler_path, drive=None):
    drive = drive or FakeDriveService()
    sheets = FakeSheetsService(records())
    app = create_app(
        settings=replace(settings(), poppler_path=poppler_path),
        sheets_service=sheets,
        drive_service=drive,
        google_token_verifier=FakeVerifier(),
        extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA}),
        verification_provider=StaticVerificationProvider(),
        repair_provider=StaticRepairProvider(MEMBER_DATA),
    )
    return TestClient(app), drive, sheets


def upload(client, content, filename="receipt.pdf", content_type="application/pdf"):
    return client.post(
        "/processing/jobs",
        headers=headers(token(client)),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": (filename, content, content_type)},
    )


def processing_rows(sheets):
    return [values for sheet_id, values in sheets.appended if sheet_id == "processing"]


def test_pdf_upload_without_poppler_fails_before_drive_or_job_creation(missing_poppler_dir):
    client, drive, sheets = processing_client(missing_poppler_dir)

    response = upload(client, sample_pdf())

    assert response.status_code == 503
    assert response.json()["detail"] == PDF_DEPENDENCY_MISSING_MESSAGE
    assert drive.uploads == []
    assert processing_rows(sheets) == []


@requires_poppler
def test_invalid_pdf_upload_is_rejected_before_drive():
    client, drive, sheets = processing_client(local_poppler_path())

    response = upload(client, INVALID_PDF)

    assert response.status_code == 400
    assert response.json()["detail"] == INVALID_PDF_MESSAGE
    assert drive.uploads == []
    assert processing_rows(sheets) == []


@requires_poppler
def test_valid_pdf_upload_runs_workflow_to_review():
    client, drive, _ = processing_client(local_poppler_path())

    response = upload(client, sample_pdf())

    assert response.status_code == 200
    assert response.json()["overall_status"] == "NEEDS_REVIEW"
    assert drive.uploads[0]["folder_id"] == "member-in"


def test_drive_upload_failure_returns_safe_message_and_keeps_detail_in_job(missing_poppler_dir):
    client, _, sheets = processing_client(missing_poppler_dir, drive=FailingDriveService())

    from tests.conftest import png_bytes
    response = upload(client, png_bytes(), "receipt.png", "image/png")

    assert response.status_code == 502
    assert "Google Drive upload failed" in response.json()["detail"]
    assert "private_key" not in response.text
    job_list = client.get("/processing/jobs", headers=headers(token(client))).json()["jobs"]
    assert "storageQuotaExceeded" in job_list[0]["last_error"]


def test_legacy_converter_reports_missing_poppler(missing_poppler_dir):
    client, _, _ = processing_client(missing_poppler_dir)
    client.app.state.settings = replace(client.app.state.settings, gemini_api_key="unused-key")
    client.app.state.gemini_client.settings = client.app.state.settings

    response = client.post(
        "/export-to-excel/",
        headers=headers(token(client)),  # legacy tools require a session
        files={"file": ("table.pdf", sample_pdf(), "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == PDF_DEPENDENCY_MISSING_MESSAGE
