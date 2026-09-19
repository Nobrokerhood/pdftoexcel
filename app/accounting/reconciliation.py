"""Deterministic accounting reconciliation.

Every check reports three things side by side: the SOURCE-WRITTEN value, the
CALCULATED value, and the DIFFERENCE. Nothing is ever adjusted to make an
equation balance: a real source discrepancy (e.g. a register whose written
closing balance is 3 rupees off its own arithmetic) is reported as a
DISCREPANCY review item and left exactly as written.

Amounts are parsed with `money.parse_amount`; an ambiguous or unreadable amount
is excluded from sums and COUNTED, so a sum is never silently incomplete.
"""

from decimal import Decimal
from typing import Any

from app.accounting.money import format_amount, parse_amount

MATCHED = "MATCHED"
DISCREPANCY = "DISCREPANCY"
NOT_AVAILABLE = "NOT_AVAILABLE"
INCOMPLETE = "INCOMPLETE"   # a sum could not include every row (unreadable amounts)

TOLERANCE = Decimal("0.01")


def _amt(value: Any) -> Decimal | None:
    reading = parse_amount(value)
    return reading.value if reading.found else None


def _sum(rows: list[dict], key: str = "Amount*") -> tuple[Decimal, int, int]:
    total, counted, unreadable = Decimal("0"), 0, 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        value = _amt(row.get(key))
        if value is None:
            unreadable += 1
        else:
            total += value
            counted += 1
    return total, counted, unreadable


def _check(check_id: str, label: str, source, calculated, note: str = "", incomplete: int = 0) -> dict:
    if source is None or calculated is None:
        status, diff = NOT_AVAILABLE, None
    else:
        diff = source - calculated
        status = MATCHED if abs(diff) <= TOLERANCE else DISCREPANCY
        if incomplete and status == DISCREPANCY:
            status = INCOMPLETE
    return {
        "check_id": check_id,
        "label": label,
        "source_value": format_amount(source) if source is not None else "-",
        "calculated_value": format_amount(calculated) if calculated is not None else "-",
        "difference": format_amount(diff) if diff is not None else "-",
        "status": status,
        "note": note,
        "rows_excluded_unreadable": incomplete,
    }


class AccountingReconciliationService:
    def reconcile(self, data: dict[str, Any], purpose: str | None = None) -> dict[str, Any]:
        purpose = (purpose or data.get("purpose") or "").upper()
        if purpose == "VENDOR_INVOICE" or data.get("vendor_detail"):
            return self._vendor(data)
        return self._register(data)

    # -- registers / receipts ---------------------------------------------------
    def _register(self, data: dict) -> dict:
        rows = [r for r in data.get("rows") or [] if isinstance(r, dict)]
        inflow_rows = [r for r in data.get("inflow_rows") or [] if isinstance(r, dict)]
        summary = data.get("balance_summary") or {}
        legacy_inflows = [i for i in summary.get("inflows") or [] if isinstance(i, dict)]

        tx_sum, tx_counted, tx_unreadable = _sum(rows)
        in_sum, in_counted, in_unreadable = _sum(inflow_rows)
        if not inflow_rows and legacy_inflows:
            in_sum, in_counted, in_unreadable = _sum(legacy_inflows, "amount")

        written_exp = _amt(summary.get("total_expenditure"))
        written_rec = _amt(summary.get("total_receipts"))
        written_open = _amt(summary.get("opening_balance"))
        written_close = _amt(summary.get("closing_balance"))
        have_inflows = (in_counted + in_unreadable) > 0

        checks = [
            _check("EXPENDITURE_TOTAL", "Written expenditure total vs sum of transaction rows",
                   written_exp, tx_sum if rows else None, incomplete=tx_unreadable),
            _check("RECEIPTS_TOTAL", "Written receipts total vs sum of cash inflow rows",
                   written_rec, in_sum if have_inflows else None, incomplete=in_unreadable),
        ]
        receipts_basis = written_rec if written_rec is not None else (in_sum if have_inflows else None)
        closing_from_totals = (receipts_basis - written_exp) if (receipts_basis is not None and written_exp is not None) else None
        checks.append(_check(
            "CLOSING_FROM_WRITTEN_TOTALS",
            "Written closing balance vs (receipts total - written expenditure total)",
            written_close, closing_from_totals,
            note="uses the totals written on the document"))

        derived_adjustment = (written_exp - tx_sum) if (written_exp is not None and rows) else None
        if written_open is not None and derived_adjustment is not None:
            checks.append(_check(
                "OPENING_ADJUSTMENT",
                "Opening balance (magnitude) vs adjustment implied by the written expenditure total",
                abs(written_open), derived_adjustment,
                note=("written expenditure total minus the sum of transaction rows; for a register that "
                      "carries a negative opening balance this should equal the deficit"),
                incomplete=tx_unreadable))
        if written_open is not None and rows and have_inflows:
            checks.append(_check(
                "CLOSING_FROM_ROWS",
                "Written closing balance vs (opening + inflow rows - transaction rows)",
                written_close, written_open + in_sum - tx_sum, incomplete=tx_unreadable + in_unreadable))

        status = MATCHED
        if any(c["status"] in (DISCREPANCY, INCOMPLETE) for c in checks):
            status = DISCREPANCY
        elif all(c["status"] == NOT_AVAILABLE for c in checks):
            status = "UNCHECKED"
        return {
            "schema_version": 2,
            "overall_status": status,
            "calculated_transaction_total": format_amount(tx_sum),
            "transaction_rows_counted": tx_counted,
            "transaction_rows_unreadable": tx_unreadable,
            "calculated_inflow_total": format_amount(in_sum) if have_inflows else "-",
            "inflow_rows_counted": in_counted,
            "inflow_rows_unreadable": in_unreadable,
            "source_total_expenditure": format_amount(written_exp) if written_exp is not None else "-",
            "source_total_receipts": format_amount(written_rec) if written_rec is not None else "-",
            "source_opening_balance": format_amount(written_open) if written_open is not None else "-",
            "source_closing_balance": format_amount(written_close) if written_close is not None else "-",
            "derived_opening_adjustment": format_amount(derived_adjustment) if derived_adjustment is not None else "-",
            "checks": checks,
            "notes": list(summary.get("notes") or []),
            "policy": "Source values are never changed to make totals balance; differences are reported for review.",
        }

    # -- vendor invoices -------------------------------------------------------
    def _vendor(self, data: dict) -> dict:
        detail = data.get("vendor_detail") or {}
        rows = [r for r in data.get("rows") or [] if isinstance(r, dict)]
        items_sum, counted, unreadable = _sum(rows)
        taxable = _amt(detail.get("taxable_amount"))
        total = _amt(detail.get("total_amount"))
        taxes = [(_amt(detail.get(k)) or Decimal("0")) for k in ("cgst_amount", "sgst_amount", "igst_amount")]
        tds = _amt(detail.get("tds_amount")) or Decimal("0")
        checks = [
            _check("LINE_ITEMS_VS_TAXABLE", "Taxable value vs sum of line items", taxable,
                   items_sum if rows else None, incomplete=unreadable),
        ]
        base = taxable if taxable is not None else (items_sum if rows else None)
        expected_total = (base + sum(taxes) - tds) if base is not None else None
        checks.append(_check("INVOICE_TOTAL", "Invoice total vs (taxable or line items) + GST - TDS",
                             total, expected_total, incomplete=unreadable))
        status = MATCHED
        if any(c["status"] in (DISCREPANCY, INCOMPLETE) for c in checks):
            status = DISCREPANCY
        elif all(c["status"] == NOT_AVAILABLE for c in checks):
            status = "UNCHECKED"
        return {
            "schema_version": 2,
            "overall_status": status,
            "calculated_line_item_total": format_amount(items_sum),
            "line_items_counted": counted,
            "line_items_unreadable": unreadable,
            "checks": checks,
            "policy": "Source values are never changed to make totals balance; differences are reported for review.",
        }
