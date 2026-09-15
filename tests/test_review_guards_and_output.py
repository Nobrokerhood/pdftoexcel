import io
from datetime import date, datetime

import pytest
from openpyxl import load_workbook
from pydantic import ValidationError

from app.accounting.output import _amount, _date
from app.accounting.schemas import MemberReceiptExtraction, VendorInvoiceExtraction
from app.core.errors import ExternalServiceUnavailableError
from app.google.sheets_service import GoogleSheetsService
from tests.test_accounting_workflow import (
    MEMBER_DATA,
    VENDOR_DATA,
    client_for,
    headers,
    settings,
    start_job,
    token,
)


class UnavailableVerifier:
    def verify(self, source_bytes, purpose, template, extracted_data):
        raise ExternalServiceUnavailableError("verification service down")


def test_failed_job_with_extracted_data_cannot_be_approved():
    # Reproduces the live incident: extraction produced data, then the run FAILED
    # before verification and mapping, and approval must not generate output.
    client, app, drive, _ = client_for()
    app.state.verification_agent.provider = UnavailableVerifier()
    session_token = token(client)

    job = start_job(client, session_token).json()
    assert job["overall_status"] == "FAILED"
    assert job["extracted_data"]["amount"] == "5000"
    assert job["mapping_status"] == "NOT_STARTED"

    approved = client.post(f"/processing/jobs/{job['job_id']}/approve", headers=headers(session_token))

    assert approved.status_code == 409
    assert approved.json()["detail"] == "JOB_NOT_AWAITING_REVIEW"
    assert [upload["folder_id"] for upload in drive.uploads] == ["member-in"]
    assert drive.moves == []


def test_unverified_job_in_review_cannot_be_approved():
    client, _, drive, _ = client_for(
        verify_results=[{"overall_status": "FAILED", "fields": []}], max_retries=0
    )
    session_token = token(client)

    job = start_job(client, session_token).json()
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["verification_status"] == "FAILED"

    approved = client.post(f"/processing/jobs/{job['job_id']}/approve", headers=headers(session_token))

    assert approved.status_code == 409
    assert approved.json()["detail"] == "VERIFICATION_NOT_PASSED"
    assert len(drive.uploads) == 1


def test_completed_job_cannot_be_approved_again_or_rejected():
    client, _, drive, _ = client_for()
    session_token = token(client)
    job = start_job(client, session_token).json()
    auth = headers(session_token)

    assert client.post(f"/processing/jobs/{job['job_id']}/approve", headers=auth).status_code == 200
    again = client.post(f"/processing/jobs/{job['job_id']}/approve", headers=auth)
    rejected = client.post(f"/processing/jobs/{job['job_id']}/reject", headers=auth)

    assert again.status_code == 409
    assert rejected.status_code == 409
    assert [upload["folder_id"] for upload in drive.uploads] == ["member-in", "member-out"]
    assert len(drive.moves) == 1


def test_failed_job_can_still_be_rejected():
    client, app, _, _ = client_for()
    app.state.verification_agent.provider = UnavailableVerifier()
    session_token = token(client)
    job = start_job(client, session_token).json()

    rejected = client.post(f"/processing/jobs/{job['job_id']}/reject", headers=headers(session_token))

    assert rejected.status_code == 200
    assert rejected.json()["overall_status"] == "REJECTED"


def download_sheet(client, session_token, purpose):
    job = start_job(client, session_token, purpose).json()
    auth = headers(session_token)
    assert client.post(f"/processing/jobs/{job['job_id']}/approve", headers=auth).status_code == 200
    content = client.get(f"/processing/jobs/{job['job_id']}/download", headers=auth).content
    sheet = load_workbook(io.BytesIO(content)).active
    header = [cell.value for cell in sheet[1]]
    return header, list(sheet.iter_rows(min_row=2))


def test_member_receipt_xlsx_writes_numeric_amount_and_real_date():
    client, _, _, _ = client_for()
    header, rows = download_sheet(client, token(client), "MEMBER_RECEIPT")
    row = dict(zip(header, rows[0]))

    assert row["Amount*"].data_type == "n"
    assert row["Amount*"].value == 5000.0
    assert isinstance(row["Transaction Date*"].value, datetime)
    assert row["Transaction Date*"].value.date() == date(2026, 8, 25)
    assert row["Cheque/Ref No*"].value == MEMBER_DATA["reference_number"]


def test_vendor_invoice_xlsx_writes_numeric_taxes_and_expense():
    client, _, _, _ = client_for()
    header, rows = download_sheet(client, token(client), "VENDOR_INVOICE")
    row = dict(zip(header, rows[0]))

    for column in ["CGST Amount", "SGST Amount", "IGST Amount", "TDS Amount", "Expense Amount"]:
        assert row[column].data_type == "n", column
    assert row["Expense Amount"].value == 10000.0
    assert row["CGST Amount"].value == 900.0
    assert row["Bill Date"].value.date() == date(2026, 8, 25)
    assert row["Bill Number"].value == VENDOR_DATA["bill_number"]


def test_amount_and_date_conversion_rules():
    assert _amount("₹ 12,450.75") == 12450.75
    assert _amount("0") == 0.0
    assert _amount(None) == "-"
    assert _amount("see attached") == "see attached"

    assert _date("03-Sep-2026") == date(2026, 9, 3)
    assert _date("03/09/2026") == date(2026, 9, 3)
    assert _date("2026-09-03") == date(2026, 9, 3)
    assert _date("") == "-"
    assert _date("next Tuesday") == "next Tuesday"


class RecordingWorksheet:
    def __init__(self, rows):
        self.rows = rows
        self.batches = []

    def get_all_values(self):
        return self.rows

    def batch_update(self, data, value_input_option=None):
        self.batches.append((data, value_input_option))

    def update_cell(self, *args):
        raise AssertionError("per-cell writes exhaust the Sheets write quota")


def test_row_update_is_a_single_batched_write():
    worksheet = RecordingWorksheet(
        [["Job ID", "Overall Status", "Completed At"], ["job-1", "PROCESSING", ""], ["job-2", "CREATED", ""]]
    )
    service = GoogleSheetsService(settings())
    service._worksheet = lambda spreadsheet_id, worksheet_name="Sheet1": worksheet

    updated = service.update_row_by_key(
        "sheet", "Job ID", "job-2", {"Overall Status": "COMPLETED", "Completed At": "now", "Unknown": "x"}
    )

    assert updated is True
    assert len(worksheet.batches) == 1
    data, input_option = worksheet.batches[0]
    assert data == [
        {"range": "B3", "values": [["COMPLETED"]]},
        {"range": "C3", "values": [["now"]]},
    ]
    assert str(input_option.value if hasattr(input_option, "value") else input_option) == "USER_ENTERED"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("12,450.75", "12450.75"),
        ("\u20b9 12,450.75", "12450.75"),
        ("Rs. 1,00,000", "100000"),
        ("INR 500", "500"),
        ("5000", "5000"),
        (5000, "5000"),
        ("", None),
        (None, None),
    ],
)
def test_extraction_schema_accepts_printed_amount_formats(raw, expected):
    assert MemberReceiptExtraction(amount=raw).model_dump(mode="json")["amount"] == expected


@pytest.mark.parametrize("raw", ["twelve thousand", "12.450,75", "Rs 12 abc", "PRS 100"])
def test_extraction_schema_rejects_ambiguous_amounts(raw):
    with pytest.raises(ValidationError):
        MemberReceiptExtraction(amount=raw)


def test_vendor_schema_normalizes_tax_and_expense_amounts():
    data = VendorInvoiceExtraction(
        cgst_amount="1,125.00", expenses=[{"expense_amount": "\u20b910,000"}]
    ).model_dump(mode="json")

    assert data["cgst_amount"] == "1125.00"
    assert data["expenses"][0]["expense_amount"] == "10000"


# The verifier response Gemini actually returned in the live run: statuses and
# confidence outside the schema.
LIVE_MALFORMED_VERIFICATION = {
    "overall_status": "PASSED",
    "fields": [
        {"field": "amount", "extracted_value": "5000", "verified_value": "5000",
         "status": "PASSED", "confidence": "HIGH", "evidence": "Amount Paid"},
    ],
}


def test_malformed_verification_fails_job_safely_instead_of_500():
    client, app, drive, _ = client_for(verify_results=[LIVE_MALFORMED_VERIFICATION])
    repair = app.state.repair_agent.provider
    session_token = token(client)

    response = start_job(client, session_token)

    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "FAILED"
    assert job["verification_status"] == "FAILED"
    assert job["current_step"] == "HUMAN_REVIEW"
    assert "unexpected format" in job["last_error"]
    assert repair.calls == 0
    approve = client.post(f"/processing/jobs/{job['job_id']}/approve", headers=headers(session_token))
    assert approve.status_code == 409
    assert len(drive.uploads) == 1


def test_repair_output_is_validated_and_normalized():
    repaired = dict(MEMBER_DATA, amount="\u20b9 5,000.00")
    client, _, _, _ = client_for(
        verify_results=[{"overall_status": "FAILED", "fields": []}, {"overall_status": "PASSED", "fields": []}],
        repaired=repaired,
    )

    job = start_job(client, token(client)).json()

    assert job["verification_status"] == "PASSED"
    assert job["extraction_attempt"] == 2
    assert job["extracted_data"]["amount"] == "5000.00"


def test_malformed_repair_output_fails_job_safely():
    client, app, _, _ = client_for(
        verify_results=[{"overall_status": "FAILED", "fields": []}],
        repaired=dict(MEMBER_DATA, amount="about five thousand"),
    )
    verifier = app.state.verification_agent.provider

    response = start_job(client, token(client))

    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "FAILED"
    assert "unexpected format" in job["last_error"]
    assert verifier.calls == 1


def test_verification_accepts_null_evidence_for_absent_fields():
    from app.accounting.schemas import VerificationResult

    result = VerificationResult(
        overall_status="PASSED",
        fields=[{"field": "meter_number", "extracted_value": None, "verified_value": None,
                 "status": "VERIFIED", "confidence": 1.0, "evidence": None, "page_number": None}],
    )

    assert result.fields[0].evidence == ""
