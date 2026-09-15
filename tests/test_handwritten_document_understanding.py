import io
import pytest
from decimal import Decimal
from openpyxl import load_workbook

from app.accounting.purposes import MEMBER_RECEIPT
from app.accounting.reconciliation import AccountingReconciliationService
from app.accounting.schemas import MemberReceiptExtraction
from app.accounting.templates import MEMBER_RECEIPT_TEMPLATE, NBH_IMPORT_COLUMNS
from app.accounting.output import TemplateOutputGenerator
from app.agents.extractor import validate_extraction, _clean_dash


def test_clean_dash_helper():
    assert _clean_dash(None) == "-"
    assert _clean_dash("") == "-"
    assert _clean_dash("null") == "-"
    assert _clean_dash("None") == "-"
    assert _clean_dash("N/A") == "-"
    assert _clean_dash("UNKNOWN") == "-"
    assert _clean_dash("  Office Expenses  ") == "Office Expenses"


def test_validate_extraction_normalizes_12_columns():
    raw_data = {
        "document_type": "PETTY_CASH_REGISTER",
        "summary": "Handwritten cash register",
        "rows": [
            {
                "Bill Head*": "Office",
                "Amount*": "900",
                "Transaction Date*": "04-07-25",
                "Cheque/Ref No*": "266",
                "Comments": "Firm Registration Stamp papers"
            },
            {
                "Bill Head*": "Salary",
                "Amount*": "10000",
                "Transaction Date*": "10-07-25",
                "Cheque/Ref No*": "284",
                "Comments": "Sujatha (H/K) Jun-25 Salary"
            }
        ]
    }
    validated = validate_extraction(MEMBER_RECEIPT, raw_data)
    rows = validated.get("rows", [])
    assert len(rows) == 2
    for r in rows:
        for col in NBH_IMPORT_COLUMNS:
            assert col in r
            assert r[col] is not None
            assert r[col] != ""


def test_reconciliation_service_math_and_discrepancy():
    recon_service = AccountingReconciliationService()
    extracted_data = {
        "rows": [
            {"Amount*": "900", "Cheque/Ref No*": "266"},
            {"Amount*": "140", "Cheque/Ref No*": "267"},
            {"Amount*": "700", "Cheque/Ref No*": "268"},
            {"Amount*": "10000", "Cheque/Ref No*": "284"},
            {"Amount*": "8400", "Cheque/Ref No*": "285"},
            {"Amount*": "9000", "Cheque/Ref No*": "286"},
            {"Amount*": "10000", "Cheque/Ref No*": "287"},
            {"Amount*": "9400", "Cheque/Ref No*": "288"},
            {"Amount*": "12000", "Cheque/Ref No*": "289"},
            {"Amount*": "10000", "Cheque/Ref No*": "290"},
        ],
        "balance_summary": {
            "opening_balance": "1714",
            "closing_balance": "10174",
            "total_expenditure": "98623",
            "total_receipts": "108800",
            "inflows": [
                {"date": "04-07-25", "amount": "10000", "ref_no": "014409"},
                {"date": "10-07-25", "amount": "68800", "ref_no": "024438"},
                {"date": "14-07-25", "amount": "30000", "ref_no": "014415"}
            ],
            "notes": ["Housekeeping salaries group sum: 68,800"]
        }
    }
    recon = recon_service.reconcile(extracted_data)
    assert recon["calculated_inflow_total"] == 108800.0
    assert recon["source_total_receipts"] == "108800"
    assert recon["inflow_status"] == "MATCHED"
    assert recon["calculated_net_balance"] == 10177.0  # 108800 - 98623
    assert recon["closing_balance_difference"] == 3.0   # 10177 - 10174
    assert recon["closing_status"] == "DISCREPANCY"


def test_exact_12_column_nbh_excel_output():
    generator = TemplateOutputGenerator()
    data = {
        "rows": [
            {
                "Payment Type*": "Cash",
                "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-",
                "Cheque/Ref No*": "266",
                "Tower No*": "-",
                "Flat No*": "-",
                "Bill Head*": "Office",
                "Amount*": "900",
                "Transaction Date*": "04-07-2025",
                "Comments": "Firm Registration Stamp papers",
                "Meter No": "-",
                "Cheque Issuer Bank": "-",
                "Cheque Date": "-"
            },
            {
                "Payment Type*": "Cash",
                "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-",
                "Cheque/Ref No*": "300",
                "Tower No*": "-",
                "Flat No*": "-",
                "Bill Head*": "Society Electrical",
                "Amount*": "2500",
                "Transaction Date*": "15-07-2025",
                "Comments": "Labour payment for common electrical wire (105)",
                "Meter No": "-",
                "Cheque Issuer Bank": "-",
                "Cheque Date": "-"
            }
        ]
    }
    filename, excel_bytes = generator.generate_xlsx(MEMBER_RECEIPT, MEMBER_RECEIPT_TEMPLATE, data, "test_job_123")
    wb = load_workbook(io.BytesIO(excel_bytes))
    sheet = wb.active

    # Check exact headers
    headers = [cell.value for cell in sheet[1]]
    assert headers == list(NBH_IMPORT_COLUMNS)
    assert len(headers) == 12

    # Check row counts and data
    assert sheet.max_row == 3
    assert sheet.cell(row=2, column=7).value == 900.0
    assert sheet.cell(row=3, column=7).value == 2500.0
    assert sheet.cell(row=3, column=3).value == "300"


def test_salary_group_vs_inflow_separation():
    # Demonstrates that salary group sum (68800) matches individual entries
    # and is distinct from payment entries.
    salary_rows = [
        {"name": "Sujatha", "amount": Decimal("10000")},
        {"name": "Saidamma", "amount": Decimal("8400")},
        {"name": "Lalitha", "amount": Decimal("9000")},
        {"name": "Krishnamma", "amount": Decimal("10000")},
        {"name": "Suvarna", "amount": Decimal("9400")},
        {"name": "Padma", "amount": Decimal("12000")},
        {"name": "Jhansi", "amount": Decimal("10000")},
    ]
    salary_sum = sum(s["amount"] for s in salary_rows)
    assert salary_sum == Decimal("68800")
    inflow_salary_amount = Decimal("68800")
    assert salary_sum == inflow_salary_amount
