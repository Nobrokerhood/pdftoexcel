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


def test_handwritten_benchmark_full_regression():
    benchmark_rows = [
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "266", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "900", "Transaction Date*": "04-07-2025", "Comments": "Firm Registration Stamp papers", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "267", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "140", "Transaction Date*": "04-07-2025", "Comments": "Stationery items", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "268", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "700", "Transaction Date*": "04-07-2025", "Comments": "Letterhead printing", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "269", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Clubhouse", "Amount*": "1200", "Transaction Date*": "05-07-2025", "Comments": "Housekeeping materials", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "270", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Clubhouse", "Amount*": "500", "Transaction Date*": "05-07-2025", "Comments": "Sanitizer and soaps", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "271", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Garden", "Amount*": "800", "Transaction Date*": "06-07-2025", "Comments": "Lawn mowing petrol", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "272", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Garden", "Amount*": "450", "Transaction Date*": "06-07-2025", "Comments": "Pesticide spray", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "284", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "10000", "Transaction Date*": "10-07-2025", "Comments": "Sujatha (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "285", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "8400", "Transaction Date*": "10-07-2025", "Comments": "Saidamma (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "286", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "9000", "Transaction Date*": "10-07-2025", "Comments": "Lalitha (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "287", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "10000", "Transaction Date*": "10-07-2025", "Comments": "Krishnamma (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "288", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "9400", "Transaction Date*": "10-07-2025", "Comments": "Suvarna (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "289", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "12000", "Transaction Date*": "10-07-2025", "Comments": "Padma (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "290", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary", "Amount*": "10000", "Transaction Date*": "10-07-2025", "Comments": "Jhansi (H/K) Jun-25 Salary", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "291", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Society Electrical", "Amount*": "880", "Transaction Date*": "11-07-2025", "Comments": "Common electrical wire Transport charges", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "292", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "350", "Transaction Date*": "11-07-2025", "Comments": "Electricity bill payment receipt copy", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "293", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Water", "Amount*": "1500", "Transaction Date*": "12-07-2025", "Comments": "Water tanker payment", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "297", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office (Shed)", "Amount*": "2250", "Transaction Date*": "13-07-2025", "Comments": "Krishna Reddy Iron Stand Rent for (Shed)", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "298", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Water", "Amount*": "750", "Transaction Date*": "13-07-2025", "Comments": "Borewell motor repair", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "299", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Garden", "Amount*": "600", "Transaction Date*": "14-07-2025", "Comments": "Tree trimming labor", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "300", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Society Electrical", "Amount*": "2500", "Transaction Date*": "15-07-2025", "Comments": "Labour payment for common electrical wire (105)", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "301", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Shed (Office)", "Amount*": "2400", "Transaction Date*": "15-07-2025", "Comments": "Iron stand Rent purpose for shed 8 days", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "302", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Society", "Amount*": "1789", "Transaction Date*": "16-07-2025", "Comments": "Kiranam & General Stores (water bottles & sunblock)", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "303", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "320", "Transaction Date*": "16-07-2025", "Comments": "Tea & snacks for committee meeting", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "304", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Salary Advance", "Amount*": "3000", "Transaction Date*": "17-07-2025", "Comments": "Jhansi (H/K) Salary Advance (July-25)", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "305", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Clubhouse", "Amount*": "420", "Transaction Date*": "18-07-2025", "Comments": "Gym cleaning supplies", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "306", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Office", "Amount*": "180", "Transaction Date*": "18-07-2025", "Comments": "Postage and courier", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "307", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Shed (Office)", "Amount*": "750", "Transaction Date*": "19-07-2025", "Comments": "Krishna Reddy Iron Stand Rent 1 day", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
        {"Payment Type*": "Cash", "Society Bank Name/Bank code(Given to you by nobrokerhood)*": "-", "Cheque/Ref No*": "308", "Tower No*": "-", "Flat No*": "-", "Bill Head*": "Miscellaneous", "Amount*": "444", "Transaction Date*": "20-07-2025", "Comments": "General hardware nails and screws", "Meter No": "-", "Cheque Issuer Bank": "-", "Cheque Date": "-"},
    ]

    assert len(benchmark_rows) == 29
    assert benchmark_rows[20]["Amount*"] == "2500"
    assert benchmark_rows[20]["Comments"] == "Labour payment for common electrical wire (105)"
    assert benchmark_rows[22]["Amount*"] == "1789"
    assert benchmark_rows[22]["Comments"] == "Kiranam & General Stores (water bottles & sunblock)"
    assert benchmark_rows[27]["Amount*"] == "750"
    assert benchmark_rows[27]["Comments"] == "Krishna Reddy Iron Stand Rent 1 day"
    assert benchmark_rows[13]["Cheque/Ref No*"] == "290"

    inflows = [
        {"date": "04-07-25", "amount": "10000", "ref_no": "014409"},
        {"date": "10-07-25", "amount": "68800", "ref_no": "024438"},
        {"date": "14-07-25", "amount": "30000", "ref_no": "014415"},
    ]
    assert len(inflows) == 3
    inflow_total = sum(float(x["amount"]) for x in inflows)
    assert inflow_total == 108800.0

    extracted_data = {
        "rows": benchmark_rows,
        "balance_summary": {
            "opening_balance": "1714",
            "closing_balance": "10174",
            "total_expenditure": "98623",
            "total_receipts": "108800",
            "inflows": inflows,
        }
    }

    recon = AccountingReconciliationService().reconcile(extracted_data)
    assert recon["calculated_inflow_total"] == 108800.0
    assert recon["source_total_receipts"] == "108800"
    assert recon["inflow_status"] == "MATCHED"
    assert recon["calculated_net_balance"] == 10177.0
    assert recon["closing_balance_difference"] == 3.0
    assert recon["closing_status"] == "DISCREPANCY"

    generator = TemplateOutputGenerator()
    _, excel_bytes = generator.generate_xlsx(MEMBER_RECEIPT, MEMBER_RECEIPT_TEMPLATE, extracted_data, "job_bench")
    wb = load_workbook(io.BytesIO(excel_bytes))
    ws = wb.active
    assert ws.max_row == 30
    assert ws.cell(row=22, column=7).value == 2500.0
    assert ws.cell(row=24, column=7).value == 1789.0
    assert ws.cell(row=29, column=7).value == 750.0
    assert ws.cell(row=15, column=3).value == "290"

