import io
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter

from main import app


client = TestClient(app)


def _sample_pdf(page_count: int = 3) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _sample_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_health_route_imports_without_external_secrets():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "NoBrokerHood PDF to Excel & Split API running."


def test_split_pdf_returns_zip_parts_without_external_services():
    response = client.post(
        "/split-pdf/",
        params={"pages_per_file": 2},
        files={"file": ("sample.pdf", _sample_pdf(3), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    with ZipFile(io.BytesIO(response.content)) as zip_file:
        assert zip_file.namelist() == ["part_1.pdf", "part_2.pdf"]
        first_part = PdfReader(io.BytesIO(zip_file.read("part_1.pdf")))
        second_part = PdfReader(io.BytesIO(zip_file.read("part_2.pdf")))

    assert len(first_part.pages) == 2
    assert len(second_part.pages) == 1


def test_split_pdf_rejects_non_pdf_upload():
    response = client.post(
        "/split-pdf/",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_split_pdf_rejects_invalid_page_group_size():
    response = client.post(
        "/split-pdf/",
        params={"pages_per_file": 0},
        files={"file": ("sample.pdf", _sample_pdf(1), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "pages_per_file must be greater than zero."


def test_gemini_endpoints_report_missing_configuration():
    response = client.post(
        "/process-document/",
        files={"file": ("sample.png", _sample_png(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "GEMINI_API_KEY is not configured."
