"""
Unit tests for purpose-specific accounting engines and semantic row classification.
"""

from decimal import Decimal
import pytest

from app.accounting.semantic_rows import classify_semantic_row, RowType
from app.accounting.engines import (
    MemberBankReceiptEngine,
    VendorInvoiceEngine,
    PettyCashRegisterEngine,
)


def test_classify_semantic_row_balances_and_totals():
    assert classify_semantic_row("Opening Balance b/f", Decimal("1714")) == RowType.OPENING_BALANCE
    assert classify_semantic_row("Closing Balance c/f", Decimal("10174")) == RowType.CLOSING_BALANCE
    assert classify_semantic_row("Total Expenditure for month", Decimal("98623")) == RowType.TOTAL
    assert classify_semantic_row("Housekeeping Salaries group sum", Decimal("68800")) == RowType.GROUPED_TRANSACTION
    assert classify_semantic_row("Letterhead printing", Decimal("700")) == RowType.EXPENSE


def test_petty_cash_engine_separates_totals_from_transactions():
    engine = PettyCashRegisterEngine()
    raw_data = {
        "balance_summary": {
            "opening_balance": "1714",
            "closing_balance": "10174",
            "total_expenditure": "98623",
            "inflows": [{"date": "04-07-25", "amount": "10000", "ref_no": "014409"}]
        },
        "rows": [
            {"Transaction Date*": "04-07-25", "Cheque/Ref No*": "266", "Amount*": "900", "Comments": "Stamp papers"},
            {"Transaction Date*": "10-07-25", "Cheque/Ref No*": "284", "Amount*": "10000", "Comments": "Sujatha Salary"},
            {"Transaction Date*": "31-07-25", "Cheque/Ref No*": "-", "Amount*": "98623", "Comments": "Total Expenditure"},
            {"Transaction Date*": "31-07-25", "Cheque/Ref No*": "-", "Amount*": "10174", "Comments": "Closing Balance"}
        ]
    }
    doc = engine.process(raw_data)
    
    assert doc.opening_balance == Decimal("1714")
    assert doc.closing_balance == Decimal("10174")
    assert len(doc.inflow_rows) == 1
    
    genuine = doc.get_genuine_transactions()
    assert len(genuine) == 2
    assert genuine[0].amount == Decimal("900")
    assert genuine[1].amount == Decimal("10000")
    
    # Verify non-transaction rows
    assert len(doc.non_transaction_rows) == 2
    non_tx_descs = [t.description for t in doc.non_transaction_rows]
    assert "Total Expenditure" in non_tx_descs
    assert "Closing Balance" in non_tx_descs


def test_member_bank_receipt_engine_tower_flat_extraction():
    engine = MemberBankReceiptEngine()
    raw_data = {
        "rows": [
            {
                "Transaction Date*": "15-05-2025",
                "Cheque/Ref No*": "982341",
                "Amount*": "4500",
                "Comments": "UPI-Maintenance A-402 Mr Sharma"
            },
            {
                "Transaction Date*": "16-05-2025",
                "Cheque/Ref No*": "000214",
                "Amount*": "5000",
                "Comments": "CHQ DEPOSIT FLAT 105 LAKSHMI SAI"
            }
        ]
    }
    doc = engine.process(raw_data)
    genuine = doc.get_genuine_transactions()
    assert len(genuine) == 2
    
    row1 = genuine[0].to_nbh_row()
    assert row1["Tower No*"] == "A"
    assert row1["Flat No*"] == "402"
    assert row1["Payment Type*"] == "UPI"
    
    row2 = genuine[1].to_nbh_row()
    assert row2["Flat No*"] == "105"
    assert row2["Payment Type*"] == "Cheque"


def test_vendor_invoice_engine_tax_line_separation():
    engine = VendorInvoiceEngine()
    raw_data = {
        "vendor_name": "PATEL ELECTRIC SALES & SERVICE",
        "invoice_number": "15",
        "invoice_date": "01-09-2026",
        "total_amount": "8700",
        "items": [
            {"description": "22W DOL G2R", "amount": "6600"},
            {"description": "24 csR", "amount": "2100"},
            {"description": "CGST 9%", "amount": "391.50"},
            {"description": "SGST 9%", "amount": "391.50"}
        ]
    }
    doc = engine.process(raw_data)
    genuine = doc.get_genuine_transactions()
    assert len(genuine) == 2
    assert genuine[0].amount == Decimal("6600")
    assert genuine[1].amount == Decimal("2100")
    
    assert len(doc.non_transaction_rows) == 2
    for non_tx in doc.non_transaction_rows:
        assert non_tx.row_type == RowType.SUBTOTAL
