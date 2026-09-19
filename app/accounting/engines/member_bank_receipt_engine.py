"""
Member Bank Receipt Accounting Engine.
Processes bank statements, member receipts, and ledger extracts.
Extracts transaction date, narration, debits, credits, balances, cheque/UTR numbers.
Separates opening/closing balances and bank charges from regular member transactions.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Any, List, Optional

from app.accounting.semantic_rows import classify_semantic_row, RowType
from app.accounting.intermediate import IntermediateTransaction, IntermediateDocument


TOWER_FLAT_REGEX = re.compile(r"\b([A-Za-z])-?\s*(\d{2,4})\b|\b(?:flat|unit|flt)\s*#?:?\s*([A-Za-z0-9-]+)\b", re.IGNORECASE)
UTR_CHQ_REGEX = re.compile(r"\b(?:utr|chq|cheque|ref|tran\s*id|txn)\s*[:#-]?\s*([A-Za-z0-9]+)\b", re.IGNORECASE)


class MemberBankReceiptEngine:
    """Deterministic domain logic for Member Bank Receipts and Bank Statements."""

    def process(self, raw_data: Dict[str, Any]) -> IntermediateDocument:
        doc = IntermediateDocument(
            purpose="MEMBER_RECEIPT",
            document_type=raw_data.get("document_type", "BANK_STATEMENT"),
            accounting_content="BANK_TRANSACTIONS",
            metadata=raw_data.get("metadata", {})
        )

        raw_rows = raw_data.get("rows", [])
        for idx, r in enumerate(raw_rows, start=1):
            narration = str(r.get("Comments") or r.get("narration") or r.get("description") or "").strip()
            date_val = str(r.get("Transaction Date*") or r.get("date") or "-").strip()
            ref_val = str(r.get("Cheque/Ref No*") or r.get("ref_no") or r.get("chq_no") or "-").strip()
            amt_raw = r.get("Amount*") or r.get("amount") or r.get("credit") or r.get("debit")
            
            amt: Optional[Decimal] = None
            if amt_raw not in {None, "", "-"}:
                try:
                    amt = Decimal(str(amt_raw).replace(",", "").strip())
                except (InvalidOperation, ValueError):
                    amt = None

            is_credit = bool(r.get("credit")) or "credit" in narration.lower() or "cr" in narration.lower()
            is_debit = bool(r.get("debit")) or "debit" in narration.lower() or "dr" in narration.lower()

            row_type = classify_semantic_row(
                narration,
                amount=amt,
                is_credit=is_credit,
                is_debit=is_debit,
                document_purpose="MEMBER_RECEIPT"
            )

            # Detect Tower / Flat from input or narration
            tower = str(r.get("Tower No*") or r.get("tower") or "-").strip()
            flat = str(r.get("Flat No*") or r.get("flat") or "-").strip()
            if tower == "-" or flat == "-":
                tf_match = TOWER_FLAT_REGEX.search(narration)
                if tf_match:
                    if tf_match.group(1) and tf_match.group(2):
                        if tower == "-":
                            tower = tf_match.group(1).upper()
                        if flat == "-":
                            flat = tf_match.group(2)
                    elif tf_match.group(3) and flat == "-":
                        flat = tf_match.group(3)

            # Detect Payment Type (UPI, NEFT, RTGS, Cheque, Transfer)
            pay_type = str(r.get("Payment Type*") or r.get("payment_type") or "Transfer").strip()
            if pay_type in {"-", "Transfer"}:
                narr_upper = narration.upper()
                if "UPI" in narr_upper:
                    pay_type = "UPI"
                elif "NEFT" in narr_upper:
                    pay_type = "NEFT"
                elif "RTGS" in narr_upper:
                    pay_type = "RTGS"
                elif "CHQ" in narr_upper or "CHEQUE" in narr_upper or (ref_val.isdigit() and len(ref_val) == 6):
                    pay_type = "Cheque"
                elif "CASH" in narr_upper:
                    pay_type = "Cash"

            # Check Cheque/UTR ref in narration if missing in ref_val
            if ref_val in {"-", ""}:
                utr_match = UTR_CHQ_REGEX.search(narration)
                if utr_match:
                    ref_val = utr_match.group(1)

            tx = IntermediateTransaction(
                transaction_id=f"mbr_{idx}",
                date=date_val,
                description=narration,
                amount=amt if amt is not None else Decimal("0"),
                reference=ref_val,
                row_type=row_type,
                page=int(r.get("page", 1)),
                source_row_idx=idx,
                payment_type=pay_type,
                bank_code=str(r.get("Society Bank Name/Bank code(Given to you by nobrokerhood)*") or r.get("bank_name_or_code") or "-").strip(),
                tower=tower,
                flat=flat,
                bill_head=str(r.get("Bill Head*") or r.get("bill_head") or "Maintenance").strip(),
                comments=narration,
                meter_no=str(r.get("Meter No") or r.get("meter_number") or "-").strip(),
                cheque_issuer_bank=str(r.get("Cheque Issuer Bank") or r.get("cheque_issuer_bank") or "-").strip(),
                cheque_date=str(r.get("Cheque Date") or r.get("cheque_date") or "-").strip(),
                user_edited=bool(r.get("USER_EDITED", False)),
            )

            if row_type in [RowType.OPENING_BALANCE, RowType.CLOSING_BALANCE, RowType.TOTAL]:
                doc.non_transaction_rows.append(tx)
                if row_type == RowType.OPENING_BALANCE and amt:
                    doc.opening_balance = amt
                elif row_type == RowType.CLOSING_BALANCE and amt:
                    doc.closing_balance = amt
            else:
                doc.transactions.append(tx)

        return doc
