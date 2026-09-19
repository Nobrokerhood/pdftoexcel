"""Purpose-specific Gemini extraction contracts: prompt, response models, and
strict response normalisation.

Previously every non-MEMBER_RECEIPT purpose was sent the vendor-invoice schema
(so a petty cash register was asked for `bill_number` / `cgst_amount`), and the
raw response was used without checking its shape. Two live runs returned a list
(crash) and a flat 12-column object (silently zero rows).

Now:
* each purpose has its own prompt and response model;
* the response is normalised by explicit, documented rules only;
* anything else raises `ExtractionContractError`, which the pipeline turns into
  NEEDS_REVIEW with the OCR source candidates preserved; never zero rows and
  never a fabricated result.
"""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.templates import NBH_IMPORT_COLUMNS


class ExtractionContractError(ValueError):
    """Gemini returned a response that does not satisfy the purpose contract."""

    def __init__(self, purpose: str, problem: str, shape: str = ""):
        super().__init__(f"{purpose}: {problem}")
        self.purpose = purpose
        self.problem = problem
        self.shape = shape


Scalar = str | int | float | None


def _scalar_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid field value")
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return value.strip()
    raise ValueError(f"expected a text or number value, got {type(value).__name__}")


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_line_ids: list[str] = Field(default_factory=list)

    @field_validator("source_line_ids", mode="before")
    @classmethod
    def _ids(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [p.strip() for p in v.replace(";", ",").split(",") if p.strip()]
        if not isinstance(v, list):
            raise ValueError("source_line_ids must be a list")
        return [str(x).strip() for x in v if str(x).strip()]


class PettyCashRow(_Row):
    row_kind: str = "EXPENSE"          # EXPENSE | INFLOW
    serial_no: str | None = None
    voucher_no: str | None = None
    date: str | None = None
    particulars: str | None = None
    category: str | None = None
    amount: str | None = None

    @field_validator("serial_no", "voucher_no", "date", "particulars", "category", "amount", mode="before")
    @classmethod
    def _text(cls, v):
        return _scalar_to_text(v)

    @field_validator("row_kind", mode="before")
    @classmethod
    def _kind(cls, v):
        text = str(v or "EXPENSE").strip().upper()
        if text not in {"EXPENSE", "INFLOW"}:
            raise ValueError("row_kind must be EXPENSE or INFLOW")
        return text


class BalanceSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    opening_balance: str | None = None
    opening_balance_note: str | None = None
    closing_balance: str | None = None
    total_expenditure: str | None = None
    total_receipts: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("opening_balance", "opening_balance_note", "closing_balance", "total_expenditure",
                     "total_receipts", mode="before")
    @classmethod
    def _text(cls, v):
        return _scalar_to_text(v)

    @field_validator("notes", mode="before")
    @classmethod
    def _notes(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]


class PettyCashResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    document_type: str | None = None
    period: str | None = None
    rows: list[PettyCashRow]
    balance_summary: BalanceSummary = Field(default_factory=BalanceSummary)



class MemberReceiptResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    document_type: str | None = None
    summary: str | None = None
    period: str | None = None
    rows: list[dict]
    balance_summary: BalanceSummary = Field(default_factory=BalanceSummary)


class VendorLineItem(_Row):
    description: str | None = None
    quantity: str | None = None
    rate: str | None = None
    amount: str | None = None

    @field_validator("description", "quantity", "rate", "amount", mode="before")
    @classmethod
    def _text(cls, v):
        return _scalar_to_text(v)


class VendorInvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    document_type: str | None = None
    vendor_name: str | None = None
    vendor_gstin: str | None = None
    bill_number: str | None = None
    bill_date: str | None = None
    due_date: str | None = None
    narration: str | None = None
    taxable_amount: str | None = None
    cgst_amount: str | None = None
    sgst_amount: str | None = None
    igst_amount: str | None = None
    tds_amount: str | None = None
    total_amount: str | None = None
    line_items: list[VendorLineItem]

    @field_validator("vendor_name", "vendor_gstin", "bill_number", "bill_date", "due_date", "narration",
                     "taxable_amount", "cgst_amount", "sgst_amount", "igst_amount", "tds_amount",
                     "total_amount", mode="before")
    @classmethod
    def _text(cls, v):
        return _scalar_to_text(v)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

_COMMON_RULES = (
    "RULES:\n"
    "- The page IMAGE is the source of truth. The OCR lines are hints with ids and positions; OCR may be wrong.\n"
    "- Never invent a value. If a value is not written or not legible, use \"-\".\n"
    "- Copy numbers exactly as written; do not correct them to make totals balance.\n"
    "- For every row, list in source_line_ids the ids of the OCR lines that belong to that row\n"
    "  (reference, date, narration, amount). This is how your row is matched to the source; list at\n"
    "  least one id whenever any OCR line lies on that row.\n"
    "- Handwriting: distinguish 1 from 7 (a crossed 7), 2 from 7, 9 from 7 in reference numbers;\n"
    "  read every digit up to the column border (e.g. 1789 not 178); a trailing '/-' closes an amount\n"
    "  and is not a digit; a parenthetical note such as '(105)' is not part of the amount.\n"
    "- Return ONE JSON object and nothing else.\n"
)

PETTY_CASH_PROMPT = (
    "You extract a PETTY CASH / EXPENSE REGISTER (cash book) for NoBrokerHood accounting.\n"
    "Report EVERY ruled row of the register in the order written, top to bottom, on every page:\n"
    "- EXPENSE rows: a payment (voucher number, date, particulars, amount in the payment column).\n"
    "- INFLOW rows: cash received / withdrawn into the register (amount in the receipts column).\n"
    "Do not report the column header, balance lines, carry-forward lines or written totals as rows;\n"
    "put those in balance_summary. A row whose amount is illegible is still reported, with amount \"-\".\n"
    + _COMMON_RULES +
    "JSON SHAPE:\n"
    "{\"document_type\": \"PETTY_CASH_REGISTER\", \"period\": \"...\",\n"
    " \"rows\": [{\"row_kind\": \"EXPENSE\"|\"INFLOW\", \"serial_no\": \"...\", \"voucher_no\": \"...\",\n"
    "            \"date\": \"DD-MM-YYYY or as written\", \"particulars\": \"...\", \"category\": \"...\",\n"
    "            \"amount\": \"digits as written, no /-\", \"source_line_ids\": [\"L12\", \"L13\"]}],\n"
    " \"balance_summary\": {\"opening_balance\": \"...\", \"opening_balance_note\": \"...\",\n"
    "            \"closing_balance\": \"...\", \"total_expenditure\": \"...\", \"total_receipts\": \"...\",\n"
    "            \"notes\": [\"...\"]}}\n"
    "Write a negative opening balance (deficit) with a leading minus sign.\n"
)

MEMBER_RECEIPT_PROMPT = (
    "You extract MEMBER RECEIPTS / BANK TRANSACTIONS for NoBrokerHood accounting.\n"
    "The document may be a single payment receipt, a bank statement, a member ledger or a register.\n"
    "Report every individual transaction as one row with exactly these 12 keys:\n"
    + ", ".join(f'"{c}"' for c in NBH_IMPORT_COLUMNS) + ",\n"
    "plus \"source_line_ids\". Balance, opening/closing and total lines are NOT rows; put them in balance_summary.\n"
    "Amount*: digits only, no currency or '/-'. Transaction Date*: DD-MM-YYYY. Use \"-\" for anything absent.\n"
    "Do not invent Tower/Flat/Bank values that are not written.\n"
    + _COMMON_RULES +
    "JSON SHAPE:\n"
    "{\"document_type\": \"MEMBER_RECEIPT\"|\"BANK_STATEMENT\"|\"SOCIETY_MEMBER_LEDGER\"|\"PETTY_CASH_REGISTER\"|\"OTHER\",\n"
    " \"summary\": \"...\", \"period\": \"...\",\n"
    " \"rows\": [{<the 12 keys>, \"source_line_ids\": [\"L3\"]}],\n"
    " \"balance_summary\": {\"opening_balance\": \"...\", \"closing_balance\": \"...\", \"total_expenditure\": \"...\",\n"
    "                      \"total_receipts\": \"...\", \"notes\": []}}\n"
)

VENDOR_INVOICE_PROMPT = (
    "You extract a VENDOR INVOICE / BILL / CASH MEMO for NoBrokerHood accounting.\n"
    "Report every purchased line item (not tax lines, not totals) in line_items.\n"
    + _COMMON_RULES +
    "JSON SHAPE:\n"
    "{\"document_type\": \"TAX_INVOICE\"|\"CASH_MEMO\"|\"LABOUR_BILL\"|\"OTHER\", \"vendor_name\": \"...\",\n"
    " \"vendor_gstin\": \"...\", \"bill_number\": \"...\", \"bill_date\": \"DD-MM-YYYY\", \"due_date\": \"...\",\n"
    " \"narration\": \"...\", \"taxable_amount\": \"...\", \"cgst_amount\": \"...\", \"sgst_amount\": \"...\",\n"
    " \"igst_amount\": \"...\", \"tds_amount\": \"...\", \"total_amount\": \"...\",\n"
    " \"line_items\": [{\"description\": \"...\", \"quantity\": \"...\", \"rate\": \"...\", \"amount\": \"...\",\n"
    "                 \"source_line_ids\": [\"L7\"]}]}\n"
    "If the document is not legible enough to identify any line item, return \"line_items\": [].\n"
)

PROMPTS = {
    PETTY_CASH_REGISTER: PETTY_CASH_PROMPT,
    MEMBER_RECEIPT: MEMBER_RECEIPT_PROMPT,
    VENDOR_INVOICE: VENDOR_INVOICE_PROMPT,
}


def prompt_for(purpose: str) -> str:
    try:
        return PROMPTS[purpose]
    except KeyError as exc:
        raise ValueError(f"No extraction contract for purpose {purpose}") from exc


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

_ROW_HINT_KEYS = set(NBH_IMPORT_COLUMNS) | {"amount", "date", "voucher_no", "particulars", "Amount*"}


def _looks_like_row(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(set(obj) & _ROW_HINT_KEYS)


def normalize_response(purpose: str, raw: Any) -> tuple[BaseModel, list[str]]:
    """Validate a Gemini response against the purpose contract.

    Returns (model, shape_notes). Accepted equivalents, each recorded in
    shape_notes: a top-level list of row objects for a multi-row purpose; a
    single row object for MEMBER_RECEIPT (single-receipt documents are valid).
    Anything else raises ExtractionContractError.
    """
    notes: list[str] = []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExtractionContractError(purpose, "response is not valid JSON", "text") from exc

    if purpose in (PETTY_CASH_REGISTER, MEMBER_RECEIPT):
        if isinstance(raw, list):
            if raw and all(_looks_like_row(item) for item in raw):
                raw = {"rows": raw}
                notes.append("SHAPE_NORMALIZED: top-level list of row objects wrapped as rows[]")
            else:
                raise ExtractionContractError(purpose, "top-level list is not a list of row objects", "list")
        if not isinstance(raw, dict):
            raise ExtractionContractError(purpose, f"expected a JSON object, got {type(raw).__name__}", type(raw).__name__)
        if "rows" not in raw:
            if purpose == MEMBER_RECEIPT and _looks_like_row(raw):
                raw = {**{k: v for k, v in raw.items() if k in ("document_type", "summary", "balance_summary")},
                       "rows": [raw]}
                notes.append("SHAPE_NORMALIZED: single transaction object wrapped as one row (single-receipt document)")
            else:
                raise ExtractionContractError(
                    purpose, "response has no 'rows' array; a register must report its rows as a list", "object_without_rows")
        if not isinstance(raw["rows"], list):
            raise ExtractionContractError(purpose, "'rows' is not a list", "rows_not_list")
        if any(not isinstance(r, dict) for r in raw["rows"]):
            raise ExtractionContractError(purpose, "'rows' contains a non-object entry", "row_not_object")

    if purpose == VENDOR_INVOICE:
        if not isinstance(raw, dict):
            raise ExtractionContractError(purpose, f"expected a JSON object, got {type(raw).__name__}", type(raw).__name__)
        if "line_items" not in raw:
            if isinstance(raw.get("expenses"), list):
                raw = {**raw, "line_items": [
                    {"description": e.get("expense_description") or e.get("description"),
                     "amount": e.get("expense_amount") or e.get("amount"),
                     "source_line_ids": e.get("source_line_ids", [])}
                    for e in raw["expenses"] if isinstance(e, dict)]}
                notes.append("SHAPE_NORMALIZED: legacy 'expenses' array mapped to line_items")
            else:
                raise ExtractionContractError(purpose, "response has no 'line_items' array", "object_without_line_items")

    try:
        if purpose == PETTY_CASH_REGISTER:
            model = PettyCashResponse.model_validate(raw)
        elif purpose == MEMBER_RECEIPT:
            model = MemberReceiptResponse.model_validate(raw)
            for idx, row in enumerate(model.rows):
                for key, value in row.items():
                    if key == "source_line_ids":
                        continue
                    if isinstance(value, (dict, list)) and key in NBH_IMPORT_COLUMNS:
                        raise ExtractionContractError(purpose, f"row {idx + 1} field '{key}' is not a scalar", "bad_field")
        elif purpose == VENDOR_INVOICE:
            model = VendorInvoiceResponse.model_validate(raw)
        else:
            raise ExtractionContractError(purpose, "unsupported purpose", "")
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ()))
        raise ExtractionContractError(purpose, f"field {loc}: {first.get('msg')}", "schema_mismatch") from exc
    return model, notes


def member_row_values(row: dict) -> dict[str, str]:
    """Exactly the 12 NBH columns from a MEMBER_RECEIPT row object (NBH names,
    unstarred names, or the documented snake_case synonyms)."""
    from app.accounting.document_result import _SYNONYMS

    out = {}
    lower = {str(k).strip().lower(): v for k, v in row.items()}
    for col in NBH_IMPORT_COLUMNS:
        value = row.get(col)
        if value is None:
            value = row.get(col.rstrip("*"))
        if value is None:
            value = lower.get(col.lower()) or lower.get(col.rstrip("*").lower())
        if value is None:
            value = next((row.get(alt) for alt in _SYNONYMS.get(col, ()) if row.get(alt) not in (None, "")), None)
        text = _scalar_to_text(value) if not isinstance(value, (dict, list)) else None
        out[col] = text if text not in (None, "") else "-"
    return out
