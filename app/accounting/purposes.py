from dataclasses import dataclass


MEMBER_RECEIPT = "MEMBER_RECEIPT"
VENDOR_INVOICE = "VENDOR_INVOICE"
PETTY_CASH_REGISTER = "PETTY_CASH_REGISTER"

# Document types the classifier can report. Only some are selectable purposes.
GENERAL_LEDGER = "GENERAL_LEDGER"
CASH_BOOK = "CASH_BOOK"
BANK_STATEMENT = "BANK_STATEMENT"
JOURNAL = "JOURNAL"
PAYMENT_VOUCHER = "PAYMENT_VOUCHER"
EXPENSE_REGISTER = "EXPENSE_REGISTER"
UNKNOWN = "UNKNOWN"

DOCUMENT_TYPES = (
    MEMBER_RECEIPT,
    VENDOR_INVOICE,
    GENERAL_LEDGER,
    CASH_BOOK,
    PETTY_CASH_REGISTER,
    BANK_STATEMENT,
    JOURNAL,
    PAYMENT_VOUCHER,
    EXPENSE_REGISTER,
    UNKNOWN,
)


@dataclass(frozen=True)
class PurposeDefinition:
    code: str
    label: str
    enabled: bool = True
    # Detected document types this purpose can legitimately process.
    accepts: tuple[str, ...] = ()
    # Multi-row register documents are extracted as rows, not one record.
    row_based: bool = False


PURPOSES = [
    PurposeDefinition(MEMBER_RECEIPT, "Member Bank Receipt", accepts=(MEMBER_RECEIPT,)),
    PurposeDefinition(VENDOR_INVOICE, "Vendor Invoice", accepts=(VENDOR_INVOICE,)),
    PurposeDefinition(
        PETTY_CASH_REGISTER,
        "Petty Cash / Expense Register",
        accepts=(PETTY_CASH_REGISTER, CASH_BOOK, EXPENSE_REGISTER),
        row_based=True,
    ),
]

DOCUMENT_TYPE_LABELS = {
    MEMBER_RECEIPT: "Member Receipt",
    VENDOR_INVOICE: "Vendor Invoice",
    GENERAL_LEDGER: "General Ledger",
    CASH_BOOK: "Cash Book",
    PETTY_CASH_REGISTER: "Petty Cash Register",
    BANK_STATEMENT: "Bank Statement",
    JOURNAL: "Journal",
    PAYMENT_VOUCHER: "Payment Voucher",
    EXPENSE_REGISTER: "Expense Register",
    UNKNOWN: "Unknown",
}


def supported_purpose_codes() -> set[str]:
    return {purpose.code for purpose in PURPOSES if purpose.enabled}


def purpose_definition(code: str) -> PurposeDefinition | None:
    code = (code or "").strip().upper()
    return next((purpose for purpose in PURPOSES if purpose.code == code), None)


def purpose_for_document_type(document_type: str) -> PurposeDefinition | None:
    return next((purpose for purpose in PURPOSES if document_type in purpose.accepts), None)
