"""
Petty Cash Register Accounting Engine.
Processes petty cash registers, cash books, and monthly expense sheets.
Understands cash inflows, expense vouchers, salary groups (e.g. ₹68,800 housekeeping pool),
subtotals, totals, opening balance, and closing balance.
Guarantees non-transaction rows (like 'Total Expenditure ₹98,623' or 'Closing Balance ₹10,174')
are kept in balance summaries and never converted into NBH transactions.
"""

from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Optional

from app.accounting.amounts import resolve_amount_candidates
from app.accounting.semantic_rows import classify_semantic_row, RowType
from app.accounting.intermediate import IntermediateTransaction, IntermediateDocument


class PettyCashRegisterEngine:
    """Deterministic domain logic for Petty Cash Registers and Cash Books."""

    def process(self, raw_data: Dict[str, Any]) -> IntermediateDocument:
        doc = IntermediateDocument(
            purpose="PETTY_CASH_REGISTER",
            document_type=raw_data.get("document_type", "CASH_BOOK") or "CASH_BOOK",
            accounting_content="EXPENSE_LOG",
            metadata=raw_data.get("metadata") or {}
        )

        bal_summary = raw_data.get("balance_summary") or {}
        opg_raw = bal_summary.get("opening_balance") or raw_data.get("opening_balance")
        clg_raw = bal_summary.get("closing_balance") or raw_data.get("closing_balance")
        tot_exp_raw = bal_summary.get("total_expenditure") or raw_data.get("total_expenditure")
        tot_rec_raw = bal_summary.get("total_receipts") or raw_data.get("total_receipts")

        if opg_raw not in {None, "", "-"}:
            try:
                doc.opening_balance = Decimal(str(opg_raw).replace(",", "").strip())
            except (InvalidOperation, ValueError):
                pass

        if clg_raw not in {None, "", "-"}:
            try:
                doc.closing_balance = Decimal(str(clg_raw).replace(",", "").strip())
            except (InvalidOperation, ValueError):
                pass

        if tot_exp_raw not in {None, "", "-"}:
            try:
                doc.total_expenditure = Decimal(str(tot_exp_raw).replace(",", "").strip())
            except (InvalidOperation, ValueError):
                pass

        if tot_rec_raw not in {None, "", "-"}:
            try:
                doc.total_inflows = Decimal(str(tot_rec_raw).replace(",", "").strip())
            except (InvalidOperation, ValueError):
                pass

        # Parse Inflows
        raw_inflows = bal_summary.get("inflows") or raw_data.get("inflows") or []
        for idx, inf in enumerate(raw_inflows, start=1):
            amt_raw = inf.get("amount")
            amt = Decimal("0")
            if amt_raw not in {None, "", "-"}:
                try:
                    amt = Decimal(str(amt_raw).replace(",", "").strip())
                except (InvalidOperation, ValueError):
                    amt = Decimal("0")

            inflow_tx = IntermediateTransaction(
                transaction_id=f"inflow_{idx}",
                date=str(inf.get("date", "-")),
                description=str(inf.get("note") or f"Cash Inflow Ref {inf.get('ref_no', '')}"),
                amount=amt,
                reference=str(inf.get("ref_no", "-")),
                row_type=RowType.INFLOW,
                payment_type="Cash",
                comments=f"Inflow received: {inf.get('ref_no', '-')}"
            )
            doc.inflow_rows.append(inflow_tx)

        # Parse Expense / Transaction rows
        raw_rows = raw_data.get("rows") or []
        for idx, r in enumerate(raw_rows, start=1):
            particulars = str(r.get("Comments") or r.get("particulars") or r.get("description") or "").strip()
            date_val = str(r.get("Transaction Date*") or r.get("date") or "-").strip()
            ref_val = str(r.get("Cheque/Ref No*") or r.get("voucher_no") or r.get("ref_no") or "-").strip()
            amt_raw = r.get("Amount*") or r.get("amount") or r.get("expense") or r.get("paid")

            # None (not 0) when the source amount is unreadable, so an abstention
            # is never exported as a zero-value transaction.
            amt: Optional[Decimal] = None
            if amt_raw not in {None, "", "-"}:
                try:
                    amt = Decimal(str(amt_raw).replace(",", "").strip())
                except (InvalidOperation, ValueError):
                    amt = None

            amount_status = str(r.get("amount_status") or ("FOUND" if amt is not None else "MISSING"))
            amount_candidates = [str(c) for c in (r.get("amount_candidates") or [])]

            # OCR abstained but offered competing readings. Resolve only the
            # well-understood rupee-suffix artefact; anything else stays unresolved.
            disambiguation_note = ""
            if amt is None and amount_candidates:
                resolved, disambiguation_note = resolve_amount_candidates(amount_candidates)
                if resolved is not None:
                    amt = resolved
                    amount_status = "RESOLVED_FROM_CANDIDATES"

            # A voucher number, a date or a narration is enough to prove this row
            # refers to a real transaction even when the amount is unreadable.
            has_identity = bool(
                (ref_val and ref_val != "-")
                or (date_val and date_val != "-")
                or particulars
            )

            row_type = classify_semantic_row(
                particulars,
                amount=amt,
                document_purpose="PETTY_CASH_REGISTER",
                has_transaction_identity=has_identity,
            )

            tx = IntermediateTransaction(
                transaction_id=f"petty_{idx}",
                date=date_val,
                description=particulars,
                amount=amt,
                reference=ref_val,
                row_type=row_type,
                page=int(r.get("page", 1)),
                source_row_idx=idx,
                payment_type=str(r.get("Payment Type*") or r.get("payment_type") or "Cash").strip(),
                bank_code=str(r.get("Society Bank Name/Bank code(Given to you by nobrokerhood)*") or r.get("bank_name_or_code") or "-").strip(),
                tower=str(r.get("Tower No*") or r.get("tower") or "-").strip(),
                flat=str(r.get("Flat No*") or r.get("flat") or "-").strip(),
                bill_head=str(r.get("Bill Head*") or r.get("bill_head") or "Society Expense").strip(),
                comments=particulars,
                meter_no=str(r.get("Meter No") or r.get("meter_number") or "-").strip(),
                cheque_issuer_bank=str(r.get("Cheque Issuer Bank") or r.get("cheque_issuer_bank") or "-").strip(),
                cheque_date=str(r.get("Cheque Date") or r.get("cheque_date") or "-").strip(),
                user_edited=bool(r.get("USER_EDITED", False)),
                amount_candidates=amount_candidates,
                amount_status=amount_status,
            )

            if disambiguation_note:
                tx.verification_status = "FLAG_FOR_HUMAN"
                tx.review_reasons.append(disambiguation_note)

            if row_type == RowType.UNRESOLVED_TRANSACTION:
                tx.verification_status = "FLAG_FOR_HUMAN"
                detail = (
                    f" OCR read candidates: {', '.join(amount_candidates)}."
                    if amount_candidates else ""
                )
                tx.review_reasons.append(
                    f"Amount could not be read confidently from the source "
                    f"(status {amount_status}).{detail}"
                )

            # Separate non-transaction rows (balances, totals, subtotals)
            if row_type in [RowType.OPENING_BALANCE, RowType.CLOSING_BALANCE, RowType.TOTAL, RowType.SUBTOTAL, RowType.CARRY_FORWARD]:
                doc.non_transaction_rows.append(tx)
            elif row_type == RowType.INFLOW:
                doc.inflow_rows.append(tx)
            else:
                doc.transactions.append(tx)

        return doc
