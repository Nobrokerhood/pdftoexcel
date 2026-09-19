"""
Tests for PageLevelPdfInspector and adaptive page routing.
"""

import os
import pytest
from app.documents.pdf_inspector import PageLevelPdfInspector, assess_text_reliability


from app.core.config import get_settings

POPPLER_PATH = get_settings().poppler_path  # from configuration / PATH, never a developer path
TESTING_FOLDER = r"D:\Gen Ai\pdftoexcel\testing_folder"


def test_assess_text_reliability_edge_cases():
    assert assess_text_reliability("")[0] is False
    assert assess_text_reliability("Short")[0] is False
    assert assess_text_reliability("A" * 100)[0] is False  # no whitespace tokens
    
    clean_text = "Transaction Date Narration Cheque Ref Amount Debit Credit Balance 04-07-2025 Opening Balance 10000.00"
    is_rel, ratio = assess_text_reliability(clean_text)
    assert is_rel is True
    assert ratio > 0.95


def test_inspect_digital_pdf():
    pdf_path = os.path.join(TESTING_FOLDER, "sample-radhakrishna.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("Test file not found")
    
    inspector = PageLevelPdfInspector(poppler_path=POPPLER_PATH)
    pdf_bytes = open(pdf_path, "rb").read()
    results = inspector.inspect_document(pdf_bytes)

    assert len(results) == 1
    page1 = results[0]
    assert page1.page_number == 1
    assert page1.is_digital is True
    assert page1.is_scanned is False
    assert page1.routing_decision == "DIGITAL_TEXT"
    assert page1.text_char_count > 4000
    assert "Member Name" in page1.embedded_text


def test_inspect_scanned_pdf():
    pdf_path = os.path.join(TESTING_FOLDER, "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("Test file not found")
    
    inspector = PageLevelPdfInspector(poppler_path=POPPLER_PATH)
    pdf_bytes = open(pdf_path, "rb").read()
    results = inspector.inspect_document(pdf_bytes)

    assert len(results) == 1
    page1 = results[0]
    assert page1.page_number == 1
    assert page1.is_digital is False
    assert page1.is_scanned is True
    assert page1.routing_decision == "SCANNED_RENDER"
    assert page1.text_char_count == 0


def test_stream_pages_memory_bounded():
    pdf_path = os.path.join(TESTING_FOLDER, "IDFCFIRSTBankstatement_10178263032_134000056 (2)_page-0008.pdf")
    if not os.path.exists(pdf_path):
        pytest.skip("Test file not found")
    
    inspector = PageLevelPdfInspector(poppler_path=POPPLER_PATH)
    pdf_bytes = open(pdf_path, "rb").read()
    
    pages_seen = []
    for insp, img in inspector.stream_pages(pdf_bytes, dpi=100):
        pages_seen.append(insp.page_number)
        assert insp.routing_decision == "SCANNED_RENDER"
        assert img is not None
        assert img.width > 0 and img.height > 0
    
    assert pages_seen == [1, 2]
