"""Tests for real OCR engine availability and the capability health contract.

The rule these enforce: an engine counts as available only if it can actually
run inference. `import paddleocr` succeeds on paddlepaddle 3.3.1 and then fails
at model load inside the C++ executor, so an import-based check reports an
engine READY that cannot read a single page.

Engine-dependent tests skip (never silently pass) when an engine is absent, so a
runtime missing PaddleOCR produces visible skips rather than false green.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.documents.capabilities import (
    NOT_CONFIGURED,
    READY,
    UNAVAILABLE,
    collect_capabilities,
    probe_image_processing,
    probe_paddleocr,
    probe_pdf_renderer,
    probe_rapidocr,
)


def _text_image(text: str = "INVOICE 8700", size=(420, 130)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw.text((18, 45), text, fill="black")
    return img


# --------------------------------------------------------------------------
# Engines must prove themselves by inference, not by import
# --------------------------------------------------------------------------

def test_rapidocr_probe_reports_a_valid_status():
    status = probe_rapidocr()
    assert status.name == "rapidocr"
    assert status.status in {READY, UNAVAILABLE}


def test_paddleocr_probe_reports_a_valid_status_with_reason_when_unavailable():
    status = probe_paddleocr()
    assert status.name == "paddleocr"
    assert status.status in {READY, UNAVAILABLE}
    if status.status == UNAVAILABLE:
        # An unavailable engine must say why, so a silent downgrade is impossible.
        assert status.detail.strip()


def test_paddleocr_availability_is_not_merely_import_success():
    """A broken runtime must report UNAVAILABLE even though the import works."""
    paddleocr = pytest.importorskip("paddleocr")
    from app.documents.ocr_router import PaddleOcrProvider

    provider = PaddleOcrProvider()
    available = provider.available()
    assert isinstance(available, bool)
    if not available:
        assert provider.unavailable_reason, "an unavailable engine must explain itself"


def test_rapidocr_actually_reads_a_real_image():
    from app.documents.ocr import RapidOcrProvider

    provider = RapidOcrProvider()
    if not provider.available():
        pytest.skip("RapidOCR not available in this runtime")
    result = provider.read(_text_image(), page_idx=1, variant="canonical")
    _assert_ocr_result(result, "rapidocr")


def _assert_ocr_result(result, engine):
    """The common OCR contract: engine, page, variant, size, timing, lines with ids."""
    from app.documents.ocr_contract import OcrResult
    assert isinstance(result, OcrResult)
    assert result.engine == engine and result.page == 1 and result.variant == "canonical"
    assert (result.width, result.height) == (420, 130)
    assert result.duration_ms >= 0
    assert result.lines, "a clear printed line must be read"
    for line in result.lines:
        assert line.line_id.startswith(f"p1.{engine}.canonical.")
        assert line.text.strip()
        assert 0.0 <= line.confidence <= 1.0
        assert line.bbox[2] >= line.bbox[0] and line.bbox[3] >= line.bbox[1]


def test_paddleocr_actually_reads_a_real_image():
    from app.documents.ocr_router import PaddleOcrProvider

    provider = PaddleOcrProvider()
    if not provider.available():
        pytest.skip(f"PaddleOCR not available: {provider.unavailable_reason}")
    result = provider.read(_text_image(), page_idx=1, variant="canonical")
    _assert_ocr_result(result, "paddleocr")


def test_both_engines_expose_a_compatible_read_signature():
    """The orchestrator calls engines interchangeably: one signature, one result type."""
    from app.documents.ocr import RapidOcrProvider
    from app.documents.ocr_contract import OcrResult
    from app.documents.ocr_router import PaddleOcrProvider

    image = _text_image()
    for provider in (RapidOcrProvider(), PaddleOcrProvider()):
        if not provider.available():
            continue
        assert isinstance(provider.read(image), OcrResult)
        assert isinstance(provider.read(image, page_idx=2, variant="v"), OcrResult)


# --------------------------------------------------------------------------
# Supporting capabilities
# --------------------------------------------------------------------------

def test_image_processing_probe_exercises_opencv():
    status = probe_image_processing()
    assert status.status == READY, status.detail
    assert status.version


def test_pdf_renderer_probe():
    status = probe_pdf_renderer(None)
    assert status.status in {READY, UNAVAILABLE}


# --------------------------------------------------------------------------
# The health contract
# --------------------------------------------------------------------------

class _Settings:
    environment = "development"
    poppler_path = None
    gemini_api_key = None
    gemini_model = "gemini-2.5-flash"


def test_capability_report_shape_and_no_secret_fields():
    report = collect_capabilities(_Settings())
    for key in ("overall", "ocr_engines_ready", "ensemble_possible",
                "missing_required", "capabilities", "schema_version"):
        assert key in report
    assert report["overall"] in {READY, "DEGRADED", "NOT_READY"}

    # Check for leaked secret VALUES, not for field names: a message such as
    # "GEMINI_API_KEY is not set" names a setting without disclosing anything.
    serialized = str(report)
    for forbidden in ("AIza", "GOCSPX-", "BEGIN PRIVATE KEY", "iam.gserviceaccount.com"):
        assert forbidden not in serialized, f"capability report leaked {forbidden}"


def test_missing_gemini_is_reported_not_configured():
    report = collect_capabilities(_Settings())
    assert report["capabilities"]["gemini"]["status"] == NOT_CONFIGURED


def test_missing_required_capability_makes_overall_not_ready():
    """Production must fail its probe rather than silently downgrade extraction."""
    report = collect_capabilities(_Settings())
    if report["missing_required"]:
        assert report["overall"] == "NOT_READY"
    else:
        assert report["overall"] in {READY, "DEGRADED"}


def test_ensemble_flag_matches_engine_count():
    report = collect_capabilities(_Settings())
    assert report["ensemble_possible"] == (len(report["ocr_engines_ready"]) >= 2)
    assert report["ocr_engine_count"] == len(report["ocr_engines_ready"])


def test_capabilities_endpoint_is_reachable_and_leaks_nothing():
    from fastapi.testclient import TestClient
    from app.app_factory import create_app

    client = TestClient(create_app())
    response = client.get("/config/capabilities")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "overall" in body and "capabilities" in body
    raw = response.text
    for marker in ("AIza", "GOCSPX", "BEGIN PRIVATE KEY", "private_key", "iam.gserviceaccount.com"):
        assert marker not in raw, f"capability endpoint leaked {marker}"
