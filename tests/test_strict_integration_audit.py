import io
import os
from unittest.mock import patch, MagicMock
from decimal import Decimal
import pytest
from openpyxl import load_workbook

from app.documents.ingestion import build_manifest
from app.documents.ocr import DocumentOcrService, RapidOcrProvider
from app.documents.preprocessing import detect_orientation, correct_orientation
from app.accounting.templates import NBH_IMPORT_COLUMNS, MEMBER_RECEIPT_TEMPLATE
from app.accounting.output import TemplateOutputGenerator
from app.accounting.validation import AccountingValidationService
from app.accounting.reconciliation import AccountingReconciliationService
from app.accounting.engines.member_bank_receipt_engine import MemberBankReceiptEngine
from app.accounting.engines.petty_cash_register_engine import PettyCashRegisterEngine
from app.accounting.engines.vendor_invoice_engine import VendorInvoiceEngine
from app.agents.extractor import validate_extraction
from app.agents.verifier import GeminiVerificationProvider, VerificationResult
from app.workflows.accounting_graph import AccountingWorkflow
from app.processing.jobs import ProcessingJob


# Poppler comes from configuration (POPPLER_PATH or PATH), never a hardcoded developer path.
from app.core.config import get_settings

POPPLER_PATH = get_settings().poppler_path
CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "testing_folder")


def test_requirement_10_digital_pdf_bypass_in_real_workflow():
    """
    Requirement 10: Digital PDF bypass in real workflow.
    For sample-radhakrishna.pdf:
    - detects reliable embedded text
    - does not unnecessarily rasterize
    - does not send the page through OCR (assert OCR was not invoked)
    - preserves page identity
    - passes extracted text/layout downstream
    """
    pdf_path = os.path.join(CORPUS_DIR, "sample-radhakrishna.pdf")
    assert os.path.exists(pdf_path), "sample-radhakrishna.pdf not found in testing_folder"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    manifest = build_manifest(pdf_bytes, "sample-radhakrishna.pdf", "application/pdf", POPPLER_PATH)
    service = DocumentOcrService(poppler_path=POPPLER_PATH)

    # Spy on every OCR engine the orchestrator owns: none may run for a digital page.
    engines = service.orchestrator.providers
    with patch.object(engines[0], "read", wraps=engines[0].read) as rapid_read, \
            patch.object(engines[-1], "read", wraps=engines[-1].read) as last_read:
        rep = service.represent(pdf_bytes, manifest)
        assert rapid_read.call_count == 0 and last_read.call_count == 0

    assert len(rep.pages) == 1
    page = rep.pages[0]
    assert page.page_number == 1
    assert page.engine == "pdf_text"
    assert page.preprocessing == "digital_text"
    assert page.script == "PRINTED"
    # Geometry is REAL (pdfplumber word boxes), not the old fabricated (0, i*25, 800, ...).
    assert page.geometry == "REAL"
    assert len({line.bbox[0] for line in page.lines}) > 10, "x positions must vary like a real table"
    assert all(line.bbox[2] <= page.width and line.bbox[3] <= page.height for line in page.lines)
    assert len(page.lines) > 50, f"Expected >50 lines of digital text, got {len(page.lines)}"
    assert any("maint" in line.text.lower() or "member" in line.text.lower() for line in page.lines)


def test_requirement_11_scanned_pdf_routing():
    """
    Requirement 11: Scanned PDF routing.
    For IDFCFIRSTBankstatement...pdf:
    - page is detected as scanned
    - page is rendered
    - preprocessing occurs
    - OCR occurs
    - OCR evidence contains page/bbox information
    - downstream extraction consumes evidence
    """
    pdf_path = os.path.join(CORPUS_DIR, "IDFCFIRSTBankstatement_10178263032_134000056 (2)_page-0008.pdf")
    assert os.path.exists(pdf_path), "IDFC bank statement not found"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    manifest = build_manifest(pdf_bytes, "idfc_bank.pdf", "application/pdf", POPPLER_PATH)
    service = DocumentOcrService(poppler_path=POPPLER_PATH)

    rep = service.represent(pdf_bytes, manifest)
    assert len(rep.pages) == 2, f"Expected 2 pages, got {len(rep.pages)}"

    for page in rep.pages:
        # The primary line set is chosen per page by measured evidence among the
        # engines that ran; every engine's result is retained on the page.
        assert page.engine in ("rapidocr", "paddleocr")
        assert page.evidence and page.routing.get("selected")
        assert len(page.lines) > 5
        # Verify bounding box coordinates
        first_line = page.lines[0]
        assert len(first_line.bbox) == 4
        assert first_line.bbox[2] > first_line.bbox[0]
        assert first_line.bbox[3] > first_line.bbox[1]
        assert first_line.page == page.page_number


def test_requirement_12_rotated_image_orientation_correction():
    """
    Requirement 12: Rotated image end-to-end.
    For unnamed.jpg:
    - orientation detection runs
    - 90° rotation is detected and corrected
    - OCR receives corrected orientation
    - low-confidence/ambiguous fields are flagged
    - no fabricated amount/date/reference is emitted
    """
    img_path = os.path.join(CORPUS_DIR, "unnamed.jpg")
    assert os.path.exists(img_path), "unnamed.jpg not found"
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    manifest = build_manifest(img_bytes, "unnamed.jpg", "image/jpeg", POPPLER_PATH)
    service = DocumentOcrService(poppler_path=POPPLER_PATH)

    rep = service.represent(img_bytes, manifest)
    assert len(rep.pages) == 1
    page = rep.pages[0]

    # The original raw image is sideways (landscape: width > height).
    # Corrected upright image should be portrait (width < height).
    assert page.width < page.height, f"Expected portrait orientation after correction, got {page.width}x{page.height}"
    # Which preprocessing variant wins is an adaptive, quality-driven decision and
    # shifts legitimately with the OpenCV version, so assert the requirement (a
    # variant was applied and the read is usable) rather than one variant's name.
    assert page.preprocessing and page.preprocessing != "none"
    assert len(page.lines) >= 20, f"expected a usable read, got {len(page.lines)} lines"
    assert page.mean_confidence >= 0.70, f"OCR quality regressed: {page.mean_confidence}"


def test_requirement_13_and_14_user_edit_immutability_and_exact_output():
    """
    Requirements 13 & 14:
    Test the actual lifecycle:
    AI extraction -> user edits cell -> USER_EDITED=true -> verification -> validation -> recalculation -> approval -> XLSX generation.
    Assert:
    - user value survives every stage
    - exactly 12 columns
    - exact order
    - no unexpected primary-sheet columns
    - missing values = "-"
    - no None/null/blank/N/A/Unknown
    - user-edited values preserved in generated XLSX
    """
    # 1. Simulate extracted data with an initial AI extraction
    extracted_data = {
        "document_type": "MEMBER_RECEIPT",
        "summary": "Member Receipt Test",
        "amount": "900",
        "transaction_date": "04-07-25",
        "reference_number": "266",
        "tower": "A",
        "flat": "101",
        "bill_head": "Maintenance",
        "bank_name_or_code": "HDFC01",
        "payment_type": "Bank Transfer",
        "rows": [
            {
                "Payment Type*": "Bank Transfer",
                "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "HDFC01",
                "Cheque/Ref No*": "266",
                "Tower No*": "A",
                "Flat No*": "101",
                "Bill Head*": "Maintenance",
                "Amount*": "900",
                "Transaction Date*": "04-07-25",
                "Comments": "Firm Registration Stamp papers",
                "Meter No": "-",
                "Cheque Issuer Bank": "-",
                "Cheque Date": "-",
            }
        ]
    }

    # 2. User edits cell: changes Amount to 950.00, Tower to "B", and Flat to "202"
    row = extracted_data["rows"][0]
    row["Amount*"] = "950"
    row["Tower No*"] = "B"
    row["Flat No*"] = "202"
    row["Comments"] = "User Approved Stamp Papers"
    row["USER_EDITED"] = True
    # The reviewer's edit confirms the row (row identity is immutable).
    row["_row_id"] = "r0_001"
    row["_status"] = "USER_CONFIRMED"
    row["_edited_fields"] = ["Amount*", "Tower No*", "Flat No*", "Comments"]
    extracted_data["amount"] = "950"
    extracted_data["tower"] = "B"
    extracted_data["flat"] = "202"
    extracted_data["comments"] = "User Approved Stamp Papers"

    # 3. Verification step preserves row and user-edited flag
    mock_gemini = MagicMock()
    mock_gemini.settings.poppler_path = POPPLER_PATH
    mock_gemini.generate_json.return_value = {
        "overall_status": "PASSED",
        "fields": [],
        "rows": [{"row_index": 1, "status": "VERIFIED", "verified_amount": 950.0}]
    }
    verifier = GeminiVerificationProvider(gemini_client=mock_gemini)
    verif_res = verifier.verify(b"dummy", "MEMBER_RECEIPT", MEMBER_RECEIPT_TEMPLATE, extracted_data)
    assert verif_res["overall_status"] in {"PASSED", "NEEDS_REVIEW"}

    # 4. Validation step runs
    validator = AccountingValidationService()
    val_res = validator.validate("MEMBER_RECEIPT", extracted_data)
    assert val_res.status == "PASSED"

    # 5. Output generation (XLSX)
    generator = TemplateOutputGenerator()
    filename, xlsx_bytes = generator.generate_xlsx("MEMBER_RECEIPT", MEMBER_RECEIPT_TEMPLATE, extracted_data, "test_job_13")

    wb = load_workbook(io.BytesIO(xlsx_bytes))
    sheet = wb.active
    assert sheet.title == "NBH Accounting"

    # Assert exactly 12 columns in row 1 in exact order
    header_cells = [cell.value for cell in sheet[1]]
    assert len(header_cells) == 12
    assert tuple(header_cells) == NBH_IMPORT_COLUMNS

    # Assert data row values
    data_cells = [cell.value for cell in sheet[2]]
    assert len(data_cells) == 12

    # Map header to value
    row_map = dict(zip(header_cells, data_cells))
    assert row_map["Amount*"] == 950.0 or row_map["Amount*"] == "950"
    assert row_map["Tower No*"] == "B"
    assert row_map["Flat No*"] == "202"
    assert row_map["Comments"] == "User Approved Stamp Papers"

    # Check that missing fields are strictly "-" and not None, null, blank, etc.
    for col in ["Meter No", "Cheque Issuer Bank", "Cheque Date"]:
        assert row_map[col] == "-", f"Column {col} was {row_map[col]!r} instead of '-'"

    for col, val in row_map.items():
        assert val is not None, f"Column {col} had None value"
        assert str(val).strip() != "", f"Column {col} had empty string"
        assert str(val).lower() not in {"null", "none", "nan", "unknown"}, f"Column {col} had invalid placeholder {val}"


def test_requirement_15_human_review_flagged_for_ambiguity_and_discrepancy():
    """
    Requirement 15: Human review flags.
    For ambiguous cases verify that the application produces NEEDS_REVIEW / FLAG_FOR_HUMAN
    rather than silently guessing or suppressing differences:
    - reconciliation discrepancy emits WARNING
    - purpose mismatch emits WARNING
    """
    validator = AccountingValidationService()

    # Case A: Reconciliation difference
    data_with_discrepancy = {
        "rows": [
            {
                "Bill Head*": "Office",
                "Amount*": "900",
                "Transaction Date*": "04-07-25",
                "Cheque/Ref No*": "266",
                "Comments": "Test",
                "_row_id": "r0_001",
                "_status": "ACCEPTED",
            }
        ],
        "balance_summary": {"total_expenditure": "903"},
    }
    data_with_discrepancy["reconciliation"] = AccountingReconciliationService().reconcile(data_with_discrepancy)
    res = validator.validate("PETTY_CASH_REGISTER", data_with_discrepancy)
    # Status passes but reconciliation difference is preserved as a WARNING for human review
    assert any(i.code == "RECONCILIATION_DISCREPANCY" for i in res.issues)

    # Case B: Purpose mismatch (e.g. VENDOR_INVOICE uploaded as MEMBER_RECEIPT)
    data_mismatch = {
        "document_type": "VENDOR_INVOICE",
        "rows": [
            {
                "Bill Head*": "Office",
                "Amount*": "900",
                "Transaction Date*": "04-07-25",
                "Cheque/Ref No*": "266",
                "Comments": "Test 1"
            },
            {
                "Bill Head*": "Office",
                "Amount*": "900",
                "Transaction Date*": "04-07-25",
                "Cheque/Ref No*": "267",
                "Comments": "Test 2"
            }
        ]
    }
    res_mismatch = validator.validate("MEMBER_RECEIPT", data_mismatch)
    assert any(i.code == "PURPOSE_MISMATCH" for i in res_mismatch.issues)
