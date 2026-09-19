"""
Semantic Row Classification for Accounting Documents.
Classifies every row before NBH mapping into explicit accounting categories:
GENUINE_TRANSACTION, INFLOW, EXPENSE, SUBTOTAL, TOTAL, OPENING_BALANCE,
CLOSING_BALANCE, CARRY_FORWARD, CONTINUATION, ANNOTATION, GROUPED_TRANSACTION, UNKNOWN.
Ensures non-transaction rows (like 'Total Expenditure ₹98,623' or 'Closing Balance ₹10,174')
are never converted into final NBH transactions.
"""

import re
from decimal import Decimal
from typing import Optional, Dict, Any


class RowType:
    GENUINE_TRANSACTION = "GENUINE_TRANSACTION"
    INFLOW = "INFLOW"
    EXPENSE = "EXPENSE"
    # A row that identifies a real transaction (voucher/date/narration) but whose
    # amount OCR could not read confidently. Preserved and sent to human review;
    # never silently dropped and never exported with a fabricated amount.
    UNRESOLVED_TRANSACTION = "UNRESOLVED_TRANSACTION"
    SUBTOTAL = "SUBTOTAL"
    TOTAL = "TOTAL"
    OPENING_BALANCE = "OPENING_BALANCE"
    CLOSING_BALANCE = "CLOSING_BALANCE"
    CARRY_FORWARD = "CARRY_FORWARD"
    CONTINUATION = "CONTINUATION"
    ANNOTATION = "ANNOTATION"
    GROUPED_TRANSACTION = "GROUPED_TRANSACTION"
    UNKNOWN = "UNKNOWN"


TOTAL_KEYWORDS = re.compile(
    r"\b(total|grand\s*total|net\s*total|total\s*expenditure|total\s*receipts|total\s*payment|bill\s*total)\b",
    re.IGNORECASE
)
SUBTOTAL_KEYWORDS = re.compile(
    r"\b(sub\s*total|subtotal|taxable\s*value|pool\s*sum|group\s*sum)\b",
    re.IGNORECASE
)
OPENING_BALANCE_KEYWORDS = re.compile(
    r"\b(opening\s*balance|opg\.?\s*bal|balance\s*b/?f|brought\s*forward|opening\s*cash)\b",
    re.IGNORECASE
)
CLOSING_BALANCE_KEYWORDS = re.compile(
    r"\b(closing\s*balance|clg\.?\s*bal|balance\s*c/?f|carried\s*forward|cash\s*in\s*hand|net\s*balance)\b",
    re.IGNORECASE
)
CARRY_FORWARD_KEYWORDS = re.compile(
    r"\b(c/f|b/f|carried\s*over|brought\s*down)\b",
    re.IGNORECASE
)
INFLOW_KEYWORDS = re.compile(
    r"\b(received|cash\s*received|deposit|deposited|inflow|cr\.?|credit|from\s*bank|cash\s*withdrawn\s*for\s*petty)\b",
    re.IGNORECASE
)
SALARY_GROUP_KEYWORDS = re.compile(
    r"\b(h/?k\s*salary|housekeeping\s*salaries|staff\s*salary\s*pool|salary\s*group)\b",
    re.IGNORECASE
)


def classify_semantic_row(
    text: str,
    amount: Optional[Decimal] = None,
    is_credit: bool = False,
    is_debit: bool = False,
    document_purpose: str = "PETTY_CASH_REGISTER",
    has_transaction_identity: bool = False,
) -> str:
    """Classifies a single row deterministically based on text patterns and accounting position.

    `has_transaction_identity` tells the classifier that the row carries a
    voucher/reference number or a date even though `amount` is unreadable. Such a
    row is a real transaction with a missing field, not an annotation, so it is
    classified UNRESOLVED_TRANSACTION and kept for human review.
    """
    if not text:
        # An amount-less, text-less row can still be a real transaction if the
        # source gave it a voucher number or date.
        return RowType.UNRESOLVED_TRANSACTION if has_transaction_identity else RowType.UNKNOWN

    desc = text.strip().lower()

    if OPENING_BALANCE_KEYWORDS.search(desc):
        return RowType.OPENING_BALANCE

    if CLOSING_BALANCE_KEYWORDS.search(desc):
        return RowType.CLOSING_BALANCE

    if TOTAL_KEYWORDS.search(desc):
        return RowType.TOTAL

    if SALARY_GROUP_KEYWORDS.search(desc):
        return RowType.GROUPED_TRANSACTION

    if SUBTOTAL_KEYWORDS.search(desc):
        return RowType.SUBTOTAL

    if is_credit or INFLOW_KEYWORDS.search(desc):
        return RowType.INFLOW

    # If amount is present and not balance/total, it is a genuine transaction or expense
    if amount is not None and amount > 0:
        if document_purpose in ["PETTY_CASH_REGISTER", "VENDOR_INVOICE"]:
            return RowType.EXPENSE
        return RowType.GENUINE_TRANSACTION

    # No usable amount. If the source still identifies the transaction (voucher
    # number, date, narration), preserve it as unresolved rather than discarding
    # it as an annotation -- losing a real transaction is the worse error.
    if amount is None or amount == Decimal("0"):
        if has_transaction_identity:
            return RowType.UNRESOLVED_TRANSACTION
        return RowType.ANNOTATION

    return RowType.GENUINE_TRANSACTION
