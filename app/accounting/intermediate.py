"""
Intermediate Accounting Representation.
Serves as the decoupled intermediate schema between document understanding/OCR
and the final 12-column NoBrokerHood format.
Preserves line-level bounding box evidence, verification statuses, and protects USER_EDITED values.
"""

from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import List, Dict, Any, Optional, Tuple

from app.accounting.semantic_rows import RowType


NBH_COLUMNS_ORDER = [
    "Payment Type*",
    "Society Bank Name/Bank code(Given to you by nobrokerhood)*",
    "Cheque/Ref No*",
    "Tower No*",
    "Flat No*",
    "Bill Head*",
    "Amount*",
    "Transaction Date*",
    "Comments",
    "Meter No",
    "Cheque Issuer Bank",
    "Cheque Date",
]


@dataclass
class IntermediateTransaction:
    transaction_id: str
    date: str
    description: str
    # None means the source amount could not be read. It is exported as "-" and
    # never coerced to 0, which would fabricate a zero-value transaction.
    amount: Optional[Decimal]
    reference: str = "-"
    row_type: str = RowType.GENUINE_TRANSACTION
    page: int = 1
    source_row_idx: int = 0
    evidence_bbox: Optional[Tuple[int, int, int, int]] = None
    ocr_confidence: float = 1.0
    extraction_confidence: float = 1.0
    verification_status: str = "PENDING"  # "VERIFIED", "CORRECTED", "FLAG_FOR_HUMAN"
    review_reasons: List[str] = field(default_factory=list)
    user_edited: bool = False
    # Raw amount readings OCR considered when it could not commit to one value.
    amount_candidates: List[str] = field(default_factory=list)
    amount_status: str = "FOUND"  # FOUND | MISSING | AMBIGUOUS
    
    # NBH-specific fields
    payment_type: str = "Cash"
    bank_code: str = "-"
    tower: str = "-"
    flat: str = "-"
    bill_head: str = "-"
    comments: str = "-"
    meter_no: str = "-"
    cheque_issuer_bank: str = "-"
    cheque_date: str = "-"

    def to_nbh_row(self) -> Dict[str, Any]:
        """Converts to exact 12-column NBH row format with '-' for empty values."""
        def clean(val: Any) -> str:
            if val is None:
                return "-"
            s = str(val).strip()
            if s in {"", "None", "null", "N/A", "UNKNOWN", "undefined"}:
                return "-"
            return s

        if self.amount is None:
            # Unreadable amount stays "-"; it is never invented or zero-filled.
            amt_str = "-"
        else:
            amt_str = f"{self.amount:.2f}".rstrip("0").rstrip(".") or "-"

        row = {
            "Payment Type*": clean(self.payment_type),
            "Society Bank Name/Bank code(Given to you by nobrokerhood)*": clean(self.bank_code),
            "Cheque/Ref No*": clean(self.reference),
            "Tower No*": clean(self.tower),
            "Flat No*": clean(self.flat),
            "Bill Head*": clean(self.bill_head),
            "Amount*": amt_str,
            "Transaction Date*": clean(self.date),
            "Comments": clean(self.comments or self.description),
            "Meter No": clean(self.meter_no),
            "Cheque Issuer Bank": clean(self.cheque_issuer_bank),
            "Cheque Date": clean(self.cheque_date),
        }

        # Keep tracking metadata separate from primary export
        if self.user_edited:
            row["USER_EDITED"] = True

        return row


@dataclass
class IntermediateDocument:
    purpose: str
    document_type: str
    accounting_content: str
    opening_balance: Optional[Decimal] = None
    closing_balance: Optional[Decimal] = None
    total_inflows: Optional[Decimal] = None
    total_expenditure: Optional[Decimal] = None
    inflow_rows: List[IntermediateTransaction] = field(default_factory=list)
    transactions: List[IntermediateTransaction] = field(default_factory=list)
    non_transaction_rows: List[IntermediateTransaction] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_genuine_transactions(self) -> List[IntermediateTransaction]:
        """Returns rows eligible for NBH import based on purpose.

        UNRESOLVED_TRANSACTION is always eligible: the source identified a real
        transaction whose amount could not be read, so it is carried through with
        Amount* = "-" and flagged for human review. Dropping it here is exactly
        the silent row loss this pipeline must never do.
        """
        if self.purpose == "MEMBER_RECEIPT":
            eligible = {RowType.GENUINE_TRANSACTION, RowType.INFLOW}
        elif self.purpose in {"PETTY_CASH_REGISTER", "VENDOR_INVOICE"}:
            eligible = {RowType.GENUINE_TRANSACTION, RowType.EXPENSE}
        else:
            eligible = {RowType.GENUINE_TRANSACTION, RowType.INFLOW, RowType.EXPENSE}

        eligible = eligible | {RowType.UNRESOLVED_TRANSACTION}
        return [t for t in self.transactions if t.row_type in eligible]

    def get_excluded_transactions(self) -> List[IntermediateTransaction]:
        """Rows held in `transactions` that will NOT be exported.

        Exposed so every exclusion can be reported with a reason instead of
        disappearing between the engine and the NBH dataset.
        """
        exported = {id(t) for t in self.get_genuine_transactions()}
        return [t for t in self.transactions if id(t) not in exported]

    def to_nbh_dataset(self) -> List[Dict[str, Any]]:
        """Maps genuine transactions to exact 12-column NBH import dataset."""
        return [t.to_nbh_row() for t in self.get_genuine_transactions()]
