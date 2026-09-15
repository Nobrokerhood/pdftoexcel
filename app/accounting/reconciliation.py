import re
from decimal import Decimal, InvalidOperation
from typing import Any


def _parse_num(val: Any) -> Decimal | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in {"-", "null", "None", "N/A", "UNKNOWN"}:
        return None
    cleaned = re.sub(r"[^\d.\-]", "", s)
    try:
        return Decimal(cleaned)
    except (InvalidOperation, ValueError):
        return None


class AccountingReconciliationService:
    """
    Generalized accounting reconciliation service.
    Compares mathematically calculated transaction sums, cash inflows, and net balances
    against source-written totals/balances without silently forcing matches.
    """

    def reconcile(self, extracted_data: dict[str, Any]) -> dict[str, Any]:
        rows = extracted_data.get("rows", [])
        balance_summary = extracted_data.get("balance_summary") or {}

        # 1. Calculated transactions sum
        total_tx_amount = Decimal("0")
        valid_row_count = 0
        for r in rows:
            if isinstance(r, dict):
                amt = _parse_num(r.get("Amount*", r.get("amount")))
                if amt is not None:
                    total_tx_amount += amt
                    valid_row_count += 1

        # 2. Inflows
        inflows = balance_summary.get("inflows", [])
        total_inflow_amount = Decimal("0")
        if isinstance(inflows, list):
            for inf in inflows:
                if isinstance(inf, dict):
                    amt = _parse_num(inf.get("amount"))
                    if amt is not None:
                        total_inflow_amount += amt
                elif isinstance(inf, (int, float, str)):
                    amt = _parse_num(inf)
                    if amt is not None:
                        total_inflow_amount += amt

        # 3. Source-written totals
        source_expenditure_raw = balance_summary.get("total_expenditure")
        source_expenditure = _parse_num(source_expenditure_raw)

        source_receipts_raw = balance_summary.get("total_receipts")
        source_receipts = _parse_num(source_receipts_raw)

        source_opening_raw = balance_summary.get("opening_balance")
        source_opening = _parse_num(source_opening_raw)

        source_closing_raw = balance_summary.get("closing_balance")
        source_closing = _parse_num(source_closing_raw)

        # 4. Expenditure Comparison
        expenditure_diff = None
        expenditure_status = "NOT_AVAILABLE"
        if source_expenditure is not None:
            expenditure_diff = float(total_tx_amount - source_expenditure)
            if abs(expenditure_diff) < 0.01:
                expenditure_status = "MATCHED"
            else:
                expenditure_status = "DISCREPANCY"

        # 5. Inflow / Receipt Comparison
        inflow_diff = None
        inflow_status = "NOT_AVAILABLE"
        if source_receipts is not None and total_inflow_amount > 0:
            inflow_diff = float(total_inflow_amount - source_receipts)
            if abs(inflow_diff) < 0.01:
                inflow_status = "MATCHED"
            else:
                inflow_status = "DISCREPANCY"

        # 6. Net Calculated Balance
        # Net balance = Inflows - Accounted Outflows (or - Transactions)
        effective_outflows = source_expenditure if source_expenditure is not None else total_tx_amount
        effective_inflows = source_receipts if source_receipts is not None else total_inflow_amount
        calculated_net_balance = effective_inflows - effective_outflows

        closing_diff = None
        closing_status = "NOT_AVAILABLE"
        if source_closing is not None and (effective_inflows > 0 or effective_outflows > 0):
            closing_diff = float(calculated_net_balance - source_closing)
            if abs(closing_diff) < 0.01:
                closing_status = "MATCHED"
            else:
                closing_status = "DISCREPANCY"

        # 7. Opening / Carry-forward adjustment
        derived_opening_adjustment = None
        opening_diff = None
        if source_expenditure is not None and total_tx_amount > 0:
            derived_opening_adjustment = float(source_expenditure - total_tx_amount)
            if source_opening is not None:
                opening_diff = float(abs(Decimal(str(derived_opening_adjustment))) - abs(source_opening))

        overall_status = "MATCHED"
        if expenditure_status == "DISCREPANCY" or closing_status == "DISCREPANCY" or inflow_status == "DISCREPANCY":
            overall_status = "DISCREPANCY"
        elif expenditure_status == "NOT_AVAILABLE" and closing_status == "NOT_AVAILABLE":
            overall_status = "UNCHECKED"

        return {
            "overall_status": overall_status,
            "calculated_transaction_total": float(total_tx_amount),
            "source_total_expenditure": str(source_expenditure_raw) if source_expenditure_raw is not None else "-",
            "expenditure_difference": expenditure_diff,
            "expenditure_status": expenditure_status,
            "calculated_inflow_total": float(total_inflow_amount) if total_inflow_amount > 0 else "-",
            "source_total_receipts": str(source_receipts_raw) if source_receipts_raw is not None else "-",
            "inflow_difference": inflow_diff,
            "inflow_status": inflow_status,
            "source_opening_balance": str(source_opening_raw) if source_opening_raw is not None else "-",
            "derived_opening_adjustment": derived_opening_adjustment,
            "opening_balance_difference": opening_diff,
            "calculated_net_balance": float(calculated_net_balance) if (effective_inflows > 0 or effective_outflows > 0) else "-",
            "source_closing_balance": str(source_closing_raw) if source_closing_raw is not None else "-",
            "closing_balance_difference": closing_diff,
            "closing_status": closing_status,
            "notes": balance_summary.get("notes", []),
        }
