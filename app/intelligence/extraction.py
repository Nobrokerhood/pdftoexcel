"""Accounting extraction agents that work only from the OCR document representation.

Each extractor returns (data, details):
- data: values in the purpose's schema (what mapping, validation and Excel use)
- details: per-field FieldResult with evidence, confidence and candidates
"""

import re
import statistics
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.documents.ocr import DocumentRepresentation, OcrLine, OcrPage, group_rows
from app.intelligence import confidence as conf
from app.intelligence.classification import GSTIN, MONTH_HEADER
from app.intelligence.evidence import AMBIGUOUS, FOUND, MISSING, Evidence, FieldResult, missing
from app.intelligence.knowledge import KnowledgeBase
from app.intelligence.parsing import (
    MONTHS,
    amount_from_words,
    find_printed_dates,
    fuzzy_contains,
    normalise_reference,
    parse_printed_amount,
    parse_printed_date,
    read_handwritten_amount,
    read_handwritten_date,
    similarity,
    split_merged_amounts,
)


LABEL_SEPARATORS = " :-=#."


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, date) else None


def _money(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


@dataclass
class LabelMatch:
    row_index: int
    line: OcrLine
    label: str
    remainder: str


def _label_prefix(text: str, label: str) -> str | None:
    """Text after `label` when the line starts with it (tolerating OCR noise), else None."""
    lowered = text.lower()
    if lowered.startswith(label):
        return text[len(label):].strip(LABEL_SEPARATORS)
    if len(label) >= 6:
        head = lowered[: len(label)]
        if similarity(head, label) >= 0.85:
            return text[len(label):].strip(LABEL_SEPARATORS)
    return None


def find_label(rows: list[list[OcrLine]], labels: list[str], start: int = 0) -> LabelMatch | None:
    for label in labels:
        for index, row in enumerate(rows[start:], start=start):
            for line in row:
                remainder = _label_prefix(line.text.strip(), label)
                if remainder is not None:
                    return LabelMatch(index, line, label, remainder)
    return None


def value_for_label(rows: list[list[OcrLine]], match: LabelMatch, all_labels: set[str]) -> tuple[str, OcrLine, bool] | None:
    """(value text, evidence line, same_row) to the right of a label, or directly below it."""
    if match.remainder:
        return match.remainder, match.line, True
    row = rows[match.row_index]
    right = [line for line in row if line.x0 > match.line.x0 and line is not match.line]
    if right:
        return " ".join(line.text for line in right), right[0], True
    for below in rows[match.row_index + 1: match.row_index + 3]:
        text = " ".join(line.text for line in below)
        if any(_label_prefix(text, label) is not None for label in all_labels):
            return None
        return text, below[0], False
    return None


def _all_labels(groups: dict[str, list[str]]) -> set[str]:
    return {label for labels in groups.values() for label in labels}


class MemberReceiptExtractor:
    fields = (
        "payment_type", "bank_name_or_code", "reference_number", "tower", "flat", "bill_head",
        "amount", "transaction_date", "comments", "meter_number", "cheque_issuer_bank", "cheque_date",
    )

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.labels = kb.vocabulary["member_receipt_labels"]

    def extract(self, representation: DocumentRepresentation) -> tuple[dict, dict]:
        rows = group_rows(representation.all_lines)
        known = _all_labels(self.labels)
        results: dict[str, FieldResult] = {}

        def labelled(field_name: str) -> tuple[str, OcrLine, bool, str] | None:
            match = find_label(rows, self.labels[field_name])
            if not match:
                return None
            found = value_for_label(rows, match, known)
            if not found:
                return None
            return found[0], found[1], found[2], match.label

        # Amount
        hit = labelled("amount")
        amount_value = parse_printed_amount(hit[0]) if hit else None
        if hit and amount_value is not None:
            results["amount"] = conf.assess(
                FieldResult("amount", amount_value, FOUND, evidence=Evidence.from_line(hit[1], hit[3])),
                ocr_confidence=hit[1].confidence, pattern_ok=True, label_anchored=True,
            )
        else:
            results["amount"] = missing("amount", "no amount found next to an amount label")
        words_line = next((line for line in representation.all_lines if "rupees" in line.text.lower()), None)
        words_value = amount_from_words(words_line.text) if words_line else None

        # Transaction date
        hit = labelled("transaction_date")
        parsed = parse_printed_date(hit[0]) if hit else None
        if hit and parsed:
            results["transaction_date"] = conf.assess(
                FieldResult("transaction_date", parsed, FOUND, evidence=Evidence.from_line(hit[1], hit[3])),
                ocr_confidence=hit[1].confidence, pattern_ok=True, label_anchored=True,
            )
        else:
            dates = {d for d, _ in find_printed_dates(representation.text)}
            results["transaction_date"] = missing(
                "transaction_date",
                "no date next to a date label" + (f"; {len(dates)} unlabelled date(s) in document not used" if dates else ""),
            )

        # Payment mode
        hit = labelled("payment_type")
        mode = self._payment_mode(hit[0]) if hit else None
        if hit and mode:
            results["payment_type"] = conf.assess(
                FieldResult("payment_type", mode, FOUND, evidence=Evidence.from_line(hit[1], hit[3])),
                ocr_confidence=hit[1].confidence, pattern_ok=True, label_anchored=True,
            )
        else:
            modes = {m: line for line in representation.all_lines if (m := self._payment_mode(line.text))}
            if len(modes) == 1:
                (mode, line), = modes.items()
                results["payment_type"] = conf.assess(
                    FieldResult("payment_type", mode, FOUND, evidence=Evidence.from_line(line), reasons=["payment mode word found without a label"]),
                    ocr_confidence=line.confidence, pattern_ok=True,
                )
            elif len(modes) > 1:
                results["payment_type"] = FieldResult(
                    "payment_type", None, AMBIGUOUS, candidates=sorted(modes), reasons=["several payment modes mentioned"]
                )
                conf.assess(results["payment_type"], ocr_confidence=0.5)
            else:
                results["payment_type"] = missing("payment_type", "no payment mode found")

        # Reference number
        hit = labelled("reference_number")
        reference = normalise_reference(hit[0]) if hit else None
        if hit and reference:
            pattern = self.kb.vocabulary["reference_patterns"].get(results["payment_type"].value or "")
            pattern_ok = bool(re.match(pattern, reference)) if pattern else None
            results["reference_number"] = conf.assess(
                FieldResult("reference_number", reference, FOUND, evidence=Evidence.from_line(hit[1], hit[3])),
                ocr_confidence=hit[1].confidence, pattern_ok=pattern_ok, label_anchored=True,
            )
        else:
            results["reference_number"] = missing("reference_number", "no reference/UTR/cheque number label found")

        # Society bank (beneficiary bank), comments, meter, cheque details: label-only fields.
        for field_name in ("bank_name_or_code", "comments", "meter_number", "cheque_issuer_bank", "cheque_date"):
            hit = labelled(field_name)
            if not hit:
                results[field_name] = missing(field_name, "label not present in document")
                continue
            value = parse_printed_date(hit[0]) if field_name == "cheque_date" else hit[0].strip()
            if not value:
                results[field_name] = missing(field_name, "label present but value unreadable")
                continue
            results[field_name] = conf.assess(
                FieldResult(field_name, value, FOUND, evidence=Evidence.from_line(hit[1], hit[3])),
                ocr_confidence=hit[1].confidence, label_anchored=True,
            )

        # Tower / flat
        tower_flat = self._tower_flat(representation)
        if tower_flat:
            tower, flat, line = tower_flat
            for field_name, value in (("tower", tower), ("flat", flat)):
                results[field_name] = conf.assess(
                    FieldResult(field_name, value, FOUND, evidence=Evidence.from_line(line)),
                    ocr_confidence=line.confidence, pattern_ok=True,
                )
        else:
            results["tower"] = missing("tower", "no tower/flat reference found")
            results["flat"] = missing("flat", "no tower/flat reference found")

        # Bill head: a known bill head named in the source text.
        heads = self._bill_heads(representation)
        if len(heads) == 1:
            (head, line), = heads.items()
            results["bill_head"] = conf.assess(
                FieldResult("bill_head", head, FOUND, evidence=Evidence.from_line(line), reasons=[f"source mentions '{head}'"]),
                ocr_confidence=line.confidence, pattern_ok=True,
            )
        elif len(heads) > 1:
            results["bill_head"] = conf.assess(
                FieldResult("bill_head", None, AMBIGUOUS, candidates=sorted(heads), reasons=["several bill heads mentioned"]),
                ocr_confidence=0.5,
            )
        else:
            results["bill_head"] = missing("bill_head", "no known bill head mentioned")

        data = {name: self._data_value(results[name]) for name in self.fields}
        details = {
            "fields": {name: result.to_dict() for name, result in results.items()},
            "cross_checks": {
                "amount_in_words": _money(words_value),
                "amount_in_words_evidence": Evidence.from_line(words_line).__dict__ if words_line else None,
            },
        }
        return data, details

    @staticmethod
    def _data_value(result: FieldResult):
        if result.status != FOUND:
            return None
        if isinstance(result.value, Decimal):
            return _money(result.value)
        if isinstance(result.value, date):
            return result.value.strftime("%d-%m-%Y")
        return result.value

    def _payment_mode(self, text: str) -> str | None:
        lowered = f" {text.lower()} "
        found = [
            mode for mode, words in self.kb.vocabulary["payment_modes"].items()
            if any(re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered) for word in words)
        ]
        return found[0] if len(found) == 1 else None

    def _tower_flat(self, representation: DocumentRepresentation):
        for line in representation.all_lines:
            for pattern in self.kb.vocabulary["flat_patterns"]:
                match = re.search(pattern, line.text, re.IGNORECASE)
                if match:
                    return match.group("tower").upper(), match.group("flat").upper(), line
        return None

    def _bill_heads(self, representation: DocumentRepresentation) -> dict[str, OcrLine]:
        found = {}
        for line in representation.all_lines:
            lowered = line.text.lower()
            for head, words in self.kb.vocabulary["bill_heads"].items():
                if any(word in lowered for word in words):
                    found.setdefault(head, line)
        return found


class VendorInvoiceExtractor:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.labels = kb.vocabulary["vendor_invoice_labels"]

    def extract(self, representation: DocumentRepresentation) -> tuple[dict, dict]:
        rows = group_rows(representation.all_lines)
        known = _all_labels(self.labels)
        results: dict[str, FieldResult] = {}

        def text_field(field_name: str, parser=None) -> FieldResult:
            match = find_label(rows, self.labels[field_name])
            if not match:
                return missing(field_name, "label not present in document")
            found = value_for_label(rows, match, known)
            if not found:
                return missing(field_name, "label present but no value next to it")
            text, line, _ = found
            value = parser(text) if parser else text.strip()
            if value in (None, ""):
                return missing(field_name, f"value '{text}' could not be read")
            return conf.assess(
                FieldResult(field_name, value, FOUND, evidence=Evidence.from_line(line, match.label)),
                ocr_confidence=line.confidence, pattern_ok=True if parser else None, label_anchored=True,
            )

        results["bill_number"] = text_field("bill_number", lambda t: normalise_reference(t.split()[0]) if t.split() else None)
        results["bill_date"] = text_field("bill_date", lambda t: parse_printed_date(t) or (find_printed_dates(t) or [(None,)])[0][0])
        results["due_date"] = text_field("due_date", lambda t: parse_printed_date(t) or (find_printed_dates(t) or [(None,)])[0][0])
        for field_name in ("taxable_value", "cgst_amount", "sgst_amount", "igst_amount", "cess_amount", "total_amount", "tds_amount"):
            results[field_name] = self._amount_row(rows, field_name)
        results["vendor_name"] = self._vendor_name(representation)
        gstins = self._gstins(representation)
        results["vendor_gstin"] = gstins[0] if gstins else missing("vendor_gstin", "no GSTIN found")
        results["recipient_gstin"] = gstins[1] if len(gstins) > 1 else missing("recipient_gstin", "no second GSTIN found")
        expenses, expense_details = self._line_items(rows)

        def amount_or_zero(name: str) -> str | None:
            result = results[name]
            if result.status == FOUND:
                return _money(result.value)
            return "0" if result.status == MISSING and name in {"cgst_amount", "sgst_amount", "igst_amount", "tds_amount"} else None

        data = {
            "bill_number": results["bill_number"].value if results["bill_number"].status == FOUND else None,
            "bill_date": results["bill_date"].value.strftime("%d-%m-%Y") if results["bill_date"].status == FOUND else None,
            "vendor_code": None,
            "vendor_name": results["vendor_name"].value if results["vendor_name"].status == FOUND else None,
            "due_date": results["due_date"].value.strftime("%d-%m-%Y") if results["due_date"].status == FOUND else None,
            "narration": None,
            "cgst_amount": amount_or_zero("cgst_amount"),
            "sgst_amount": amount_or_zero("sgst_amount"),
            "igst_amount": amount_or_zero("igst_amount"),
            "tds_amount": amount_or_zero("tds_amount"),
            "expenses": expenses,
        }
        details = {
            "fields": {name: result.to_dict() for name, result in results.items()},
            "line_items": expense_details,
            "tax_rates": self._tax_rates(rows),
        }
        return data, details

    def _amount_row(self, rows, field_name: str) -> FieldResult:
        for label in self.labels[field_name]:
            for index, row in enumerate(rows):
                label_line = next((line for line in row if _label_prefix(line.text.strip(), label) is not None), None)
                if not label_line:
                    continue
                if field_name == "total_amount" and any(
                    other in label_line.text.lower() for other in ("sub total", "subtotal", "taxable")
                ):
                    continue
                candidates = [line for line in row if line.x0 >= label_line.x0]
                for line in sorted(candidates, key=lambda item: item.x1, reverse=True):
                    tail = line.text.split()[-1] if line.text.split() else ""
                    value = parse_printed_amount(tail)
                    if value is not None and not re.search(r"%\s*$", line.text):
                        return conf.assess(
                            FieldResult(field_name, value, FOUND, evidence=Evidence.from_line(line, label, row=index)),
                            ocr_confidence=line.confidence, pattern_ok=True, label_anchored=True,
                        )
        return missing(field_name, "no amount on a row with this label")

    def _vendor_name(self, representation: DocumentRepresentation) -> FieldResult:
        page = representation.pages[0]
        skip = ("invoice", "synthetic", "gstin", "bill to", "phone", "email", "@", "address", "state", "pan")
        top = [
            line for line in page.lines
            if line.bbox[1] < page.height * 0.25
            and not any(word in line.text.lower() for word in skip)
            and sum(ch.isalpha() for ch in line.text) >= 4
        ]
        if not top:
            return missing("vendor_name", "no supplier name in the header area")
        line = max(top, key=lambda item: (item.height, -item.bbox[1]))
        result = FieldResult("vendor_name", line.text.strip(), FOUND, evidence=Evidence.from_line(line), reasons=["largest text in the invoice header"])
        return conf.assess(result, ocr_confidence=min(line.confidence, 0.84))

    def _gstins(self, representation: DocumentRepresentation) -> list[FieldResult]:
        results = []
        for line in representation.all_lines:
            match = GSTIN.search(line.text.upper().replace(" ", ""))
            if match and all(r.value != match.group(0) for r in results):
                name = "vendor_gstin" if not results else "recipient_gstin"
                results.append(conf.assess(
                    FieldResult(name, match.group(0), FOUND, evidence=Evidence.from_line(line)),
                    ocr_confidence=line.confidence, pattern_ok=True,
                ))
        return results

    def _tax_rates(self, rows) -> dict:
        rates = {}
        for row in rows:
            text = " ".join(line.text for line in row).lower()
            for tax in ("cgst", "sgst", "igst"):
                if tax in text:
                    match = re.search(r"(\d{1,2}(?:\.\d{1,2})?)\s*%", text)
                    if match:
                        rates[tax] = match.group(1)
        return rates

    def _line_items(self, rows) -> tuple[list[dict], list[dict]]:
        header_index = next(
            (i for i, row in enumerate(rows)
             if any(w in " ".join(l.text for l in row).lower() for w in ("description", "particulars"))
             and "amount" in " ".join(l.text for l in row).lower()),
            None,
        )
        if header_index is None:
            return [], []
        stop_words = ("taxable", "sub total", "subtotal", "cgst", "sgst", "igst", "total")
        expenses, details = [], []
        for index, row in enumerate(rows[header_index + 1:], start=header_index + 1):
            text = " ".join(line.text for line in row).lower()
            if any(word in text for word in stop_words):
                break
            amount_line = row[-1]
            amount = parse_printed_amount(amount_line.text.split()[-1]) if amount_line.text.split() else None
            description_lines = [line for line in row if not re.fullmatch(r"[\d,.%\s]+", line.text)]
            if amount is None or not description_lines:
                continue
            description = description_lines[0].text.strip()
            expenses.append({"expense_code": None, "expense_description": description, "expense_amount": _money(amount)})
            item = conf.assess(
                FieldResult("expense_amount", amount, FOUND, evidence=Evidence.from_line(amount_line, row=index)),
                ocr_confidence=amount_line.confidence, pattern_ok=True,
            )
            details.append({"description": description, "description_evidence": Evidence.from_line(description_lines[0], row=index).__dict__, "amount": item.to_dict()})
        return expenses, details


@dataclass
class _Cell:
    line: OcrLine
    kind: str
    reading: object = None
    digits: str = ""


@dataclass
class _Row:
    page: int
    y: float
    cells: dict[str, list[_Cell]] = field(default_factory=dict)

    def add(self, kind: str, cell: _Cell):
        self.cells.setdefault(kind, []).append(cell)

    def first(self, kind: str) -> _Cell | None:
        items = self.cells.get(kind) or []
        return items[0] if items else None


HEADER_WORDS = ("particulars", "expenditure", "voucher", "s.no", "sno", "date", "amount")
BALANCE_WORDS = ("balance", "carry forward", "carried forward", "brought forward", "b/f", "c/f", "c/d", "b/d")
REGISTER_DATE = re.compile(r"^[\dOoIlB|]{1,3}\s*[-./]\s*[\dOoIl|]{1,3}\s*[-./]?\s*[\d.]{1,5}$|^\d{5,8}$")


class RegisterExtractor:
    """Petty cash / cash / expense registers: every row, with page and row position."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.letters = kb.vocabulary["handwriting"]["letter_to_digit"]

    def extract(self, representation: DocumentRepresentation) -> tuple[dict, dict]:
        period = self._period(representation)
        rows_out: list[dict] = []
        row_details: list[dict] = []
        notes: list[str] = []
        page_balances: list[dict] = []
        opening = closing = None
        totals: dict[str, FieldResult] = {}

        for page in representation.pages:
            page_rows, page_info = self._page(page)
            notes += page_info["notes"]
            page_balances.append({"page": page.page_number, "opening": page_info["opening"].to_dict() if page_info["opening"] else None,
                                  "closing": page_info["closing"].to_dict() if page_info["closing"] else None})
            if opening is None and page_info["opening"]:
                opening = page_info["opening"]
            if page_info["closing"]:
                closing = page_info["closing"]
            totals.update(page_info["totals"])
            for number, (data, detail) in enumerate(page_rows, start=1):
                data["source_page"] = page.page_number
                data["source_row"] = number
                detail["source_page"] = page.page_number
                detail["source_row"] = number
                rows_out.append(data)
                row_details.append(detail)

        data = {
            "register_period": period[0] if period else None,
            "opening_balance": _money(opening.value) if opening and opening.status == FOUND else None,
            "closing_balance": _money(closing.value) if closing and closing.status == FOUND else None,
            "written_payment_total": _money(totals["payment_total"].value) if "payment_total" in totals else None,
            "written_receipt_total": _money(totals["receipt_total"].value) if "receipt_total" in totals else None,
            "rows": rows_out,
        }
        details = {
            "rows": row_details,
            "document_fields": {
                "register_period": {"value": period[0], "evidence": Evidence.from_line(period[1]).__dict__} if period else None,
                "opening_balance": opening.to_dict() if opening else missing("opening_balance", "no readable opening balance").to_dict(),
                "closing_balance": closing.to_dict() if closing else missing("closing_balance", "no readable closing balance").to_dict(),
                **{name: result.to_dict() for name, result in totals.items()},
            },
            "page_balances": page_balances,
            "notes": notes,
        }
        return data, details

    def _period(self, representation: DocumentRepresentation):
        for page in representation.pages:
            for line in sorted(page.lines, key=lambda item: item.bbox[1]):
                match = MONTH_HEADER.search(line.text)
                if match and line.bbox[1] < page.height * 0.25:
                    month = MONTHS.get(match.group(1).lower()[:3]) or MONTHS.get(match.group(1).lower())
                    year = int(match.group(2))
                    year = year + 2000 if year < 100 else year
                    if month:
                        return f"{year:04d}-{month:02d}", line
        return None

    def _classify(self, line: OcrLine, page: OcrPage) -> _Cell:
        text = line.text.strip()
        lowered = text.lower()
        if any(fuzzy_contains(lowered, word, 0.75) for word in BALANCE_WORDS if len(word) > 3) or any(w in lowered for w in ("b/f", "c/f", "c/d", "b/d")):
            return _Cell(line, "BALANCE")
        if REGISTER_DATE.match(text.replace(" ", "")):
            reading = read_handwritten_date(text, self.letters)
            return _Cell(line, "DATE", reading)
        if len(text) <= 12 and sum(ch.isalpha() for ch in text) <= 3 and any(fuzzy_contains(lowered, w, 0.8) for w in HEADER_WORDS):
            return _Cell(line, "HEADER")
        if len(text) > 3 and any(fuzzy_contains(lowered, w, 0.75) for w in HEADER_WORDS[:3]):
            return _Cell(line, "HEADER")
        digits = re.sub(r"\D", "", text.replace("B", "8").replace("O", "0").replace("o", "0"))
        non_digits = len(re.sub(r"[\d\s]", "", text))
        if 1 <= len(digits) <= 2 and len(text) <= 3 and non_digits <= 1:
            return _Cell(line, "SERIAL", digits=digits)
        if len(digits) == 3 and len(text) <= 4 and non_digits <= 1 and line.x0 < page.width * 0.5:
            return _Cell(line, "VOUCHER", digits=digits)
        if len(digits) >= 2 and line.x0 > page.width * 0.55 and non_digits <= 4:
            merged = split_merged_amounts(text)
            if len(merged) >= 2:
                return _Cell(line, "TOTALS", merged)
            return _Cell(line, "AMOUNT", read_handwritten_amount(text, self.letters))
        return _Cell(line, "TEXT")

    def _page(self, page: OcrPage):
        info = {"notes": [], "opening": None, "closing": None, "totals": {}}
        cells = [self._classify(line, page) for line in page.lines]
        dates = [c for c in cells if c.kind == "DATE"]
        if len(dates) < 2:
            info["notes"].append(f"Page {page.page_number}: fewer than two dates found; no register rows extracted.")
            return [], info

        median_height = statistics.median(c.line.height for c in cells)
        date_x = statistics.median(c.line.x0 for c in dates)
        date_width = statistics.median(c.line.x1 - c.line.x0 for c in dates)
        payments_x, receipts_x = self._amount_columns([c for c in cells if c.kind == "AMOUNT"], page, info)

        anchors = sorted(dates, key=lambda c: c.line.y_center)
        spacing = statistics.median(
            [b.line.y_center - a.line.y_center for a, b in zip(anchors, anchors[1:]) if b.line.y_center - a.line.y_center > median_height * 0.5]
            or [median_height * 1.2]
        )
        rows = [_Row(page.page_number, a.line.y_center) for a in anchors]
        for row, anchor in zip(rows, anchors):
            row.add("DATE", anchor)

        def column_of(cell: _Cell) -> str | None:
            x0 = cell.line.x0
            if cell.kind == "AMOUNT":
                if receipts_x is not None and abs(x0 - receipts_x) < abs(x0 - payments_x):
                    return "RECEIPT_AMOUNT"
                if abs(x0 - payments_x) <= page.width * 0.06:
                    return "PAYMENT_AMOUNT"
                return None
            if cell.kind in {"SERIAL", "VOUCHER"}:
                return cell.kind if x0 < date_x else None
            if cell.kind == "TEXT":
                if cell.line.x1 < date_x - date_width * 0.2 and x0 < date_x * 0.45:
                    return "CATEGORY"
                if x0 >= date_x + date_width * 0.5 and x0 < payments_x - page.width * 0.02:
                    return "PARTICULARS"
            return None

        assignable = [(c, column_of(c)) for c in cells if c.kind in {"AMOUNT", "SERIAL", "VOUCHER", "TEXT"}]
        assignable = [(c, col) for c, col in assignable if col]

        # Per-column vertical offset (skewed photos, serial numbers written lower, etc.).
        offsets: dict[str, float] = {}
        for column in {col for _, col in assignable}:
            deltas = []
            for cell, col in assignable:
                if col != column:
                    continue
                nearest = min(rows, key=lambda r: abs(r.y - cell.line.y_center))
                delta = cell.line.y_center - nearest.y
                if abs(delta) < spacing * 0.6:
                    deltas.append(delta)
            offsets[column] = statistics.median(deltas) if deltas else 0.0

        leftovers: list[tuple[_Cell, str]] = []
        for cell, column in assignable:
            adjusted = cell.line.y_center - offsets.get(column, 0.0)
            nearest = min(rows, key=lambda r: abs(r.y - adjusted))
            if abs(nearest.y - adjusted) <= spacing * 0.5:
                nearest.add(column, cell)
            else:
                leftovers.append((cell, column))

        last_date_y = max(r.y for r in rows)
        first_date_y = min(r.y for r in rows)
        for cell, column in leftovers:
            if column == "RECEIPT_AMOUNT" and first_date_y - spacing <= cell.line.y_center <= last_date_y + spacing:
                extra = _Row(page.page_number, cell.line.y_center)
                extra.add(column, cell)
                rows.append(extra)
                info["notes"].append(f"Page {page.page_number}: receipt amount \"{cell.line.text}\" has no readable date on its row.")

        self._balances_and_totals(cells, page, rows, payments_x, receipts_x, info)
        rows.sort(key=lambda r: r.y)
        extracted = [self._row(row, page) for row in rows if self._is_transaction(row)]
        return extracted, info

    def _amount_columns(self, amounts: list[_Cell], page: OcrPage, info: dict) -> tuple[float, float | None]:
        if not amounts:
            return page.width * 0.85, None
        xs = sorted(c.line.x0 for c in amounts)
        clusters = [[xs[0]]]
        for x in xs[1:]:
            if x - clusters[-1][-1] > page.width * 0.04:
                clusters.append([x])
            else:
                clusters[-1].append(x)
        payments = max(clusters, key=len)
        payments_x = statistics.median(payments)
        right = [c for c in clusters if statistics.median(c) > payments_x + page.width * 0.03]
        receipts_x = statistics.median(right[-1]) if right else None
        if receipts_x is not None:
            info["notes"].append(
                f"Page {page.page_number}: two amount columns found; the larger column is read as payments and the column to its right as cash received."
            )
        ignored = [c for c in clusters if c is not payments and (not right or c is not right[-1])]
        if ignored:
            info["notes"].append(f"Page {page.page_number}: {sum(len(c) for c in ignored)} amount-like annotation(s) outside the amount columns were not used.")
        return payments_x, receipts_x

    def _balances_and_totals(self, cells, page, rows, payments_x, receipts_x, info):
        for cell in cells:
            if cell.kind == "BALANCE":
                reading = read_handwritten_amount(cell.line.text, self.letters)
                if not reading.candidates:
                    neighbours = [c for c in cells if c.kind in {"AMOUNT", "TEXT"} and c.line.x0 > cell.line.x1
                                  and abs(c.line.y_center - cell.line.y_center) < cell.line.height]
                    if neighbours:
                        reading = read_handwritten_amount(neighbours[0].line.text, self.letters)
                result = FieldResult(
                    "balance", reading.value, reading.status, evidence=Evidence.from_line(cell.line, "balance line"),
                    candidates=reading.candidates, reasons=list(reading.notes),
                )
                if reading.status == MISSING:
                    result.reasons.append("balance line present but amount unreadable")
                conf.assess(result, ocr_confidence=cell.line.confidence, corrections=len(reading.notes))
                top = cell.line.bbox[1] < page.height * 0.3 or any(w in cell.line.text.lower() for w in ("brought", "b/f", "b/d"))
                result.field = "opening_balance" if top else "closing_balance"
                info["opening" if top else "closing"] = result
            elif cell.kind == "TOTALS" and cell.line.bbox[1] > max(r.y for r in rows):
                values = cell.reading
                names = ["payment_total", "receipt_total"]
                for name, value in zip(names, values[:2]):
                    info["totals"][name] = conf.assess(
                        FieldResult(name, value, FOUND, evidence=Evidence.from_line(cell.line, "written total"),
                                    reasons=["two totals were read as one text line and split by digit grouping"]),
                        ocr_confidence=cell.line.confidence, corrections=1,
                    )

    @staticmethod
    def _is_transaction(row: _Row) -> bool:
        has_amount = bool(row.cells.get("PAYMENT_AMOUNT") or row.cells.get("RECEIPT_AMOUNT"))
        has_context = bool(row.cells.get("VOUCHER") or row.cells.get("PARTICULARS"))
        return has_amount or (bool(row.cells.get("DATE")) and has_context)

    def _row(self, row: _Row, page: OcrPage) -> tuple[dict, dict]:
        fields: dict[str, FieldResult] = {}

        date_cell = row.first("DATE")
        if date_cell:
            reading = date_cell.reading
            fields["date"] = conf.assess(
                FieldResult("date", reading.value, reading.status, evidence=Evidence.from_line(date_cell.line),
                            candidates=reading.candidates, reasons=list(reading.notes)),
                ocr_confidence=date_cell.line.confidence, pattern_ok=reading.status == FOUND, corrections=len(reading.notes),
            )
            if reading.status == MISSING:
                fields["date"].reasons.append(f"date text \"{date_cell.line.text}\" is not a valid date")
        else:
            fields["date"] = missing("date", "no date on this row")

        amount_cell = row.first("PAYMENT_AMOUNT") or row.first("RECEIPT_AMOUNT")
        direction = "RECEIPT" if row.first("RECEIPT_AMOUNT") and not row.first("PAYMENT_AMOUNT") else "PAYMENT"
        if amount_cell:
            reading = amount_cell.reading
            fields["amount"] = conf.assess(
                FieldResult("amount", reading.value, reading.status, evidence=Evidence.from_line(amount_cell.line),
                            candidates=reading.candidates, reasons=list(reading.notes)),
                ocr_confidence=amount_cell.line.confidence, pattern_ok=reading.status == FOUND, corrections=len(reading.notes),
            )
        else:
            fields["amount"] = missing("amount", "no amount on this row")

        for kind, name in (("VOUCHER", "voucher_no"), ("SERIAL", "serial_no")):
            cell = row.first(kind)
            if cell:
                fields[name] = conf.assess(
                    FieldResult(name, cell.digits, FOUND, evidence=Evidence.from_line(cell.line)),
                    ocr_confidence=cell.line.confidence, pattern_ok=cell.digits == re.sub(r"\D", "", cell.line.text),
                )
            else:
                fields[name] = missing(name, f"no {name.replace('_', ' ')} on this row")

        for kind, name in (("PARTICULARS", "particulars"), ("CATEGORY", "category")):
            items = sorted(row.cells.get(kind) or [], key=lambda c: c.line.x0)
            if items:
                text = " ".join(c.line.text.strip() for c in items)
                lowest = min(items, key=lambda c: c.line.confidence)
                fields[name] = conf.assess(
                    FieldResult(name, text, FOUND, evidence=Evidence.from_line(items[0].line)),
                    ocr_confidence=lowest.line.confidence,
                )
            else:
                fields[name] = missing(name, f"no {name} text on this row")

        if not row.first("RECEIPT_AMOUNT") and not row.first("VOUCHER") and fields["particulars"].status == FOUND:
            lowered = str(fields["particulars"].value).lower()
            if any(word in lowered for word in ("cash with", "withdrawn", "withdrawal", "received")):
                direction = "RECEIPT"

        suggestion = self.kb.suggest_category(f"{fields['particulars'].value or ''} {fields['category'].value or ''}")
        levels = [fields["amount"].confidence, fields["date"].confidence]
        if direction == "PAYMENT":
            levels.append(fields["voucher_no"].confidence)
        row_confidence = "LOW" if "LOW" in levels else ("MEDIUM" if "MEDIUM" in levels else "HIGH")

        data = {
            "serial_no": fields["serial_no"].value if fields["serial_no"].status == FOUND else None,
            "voucher_no": fields["voucher_no"].value if fields["voucher_no"].status == FOUND else None,
            "date": _iso(fields["date"].value) if fields["date"].status == FOUND else None,
            "particulars": fields["particulars"].value if fields["particulars"].status == FOUND else None,
            "category": fields["category"].value if fields["category"].status == FOUND else None,
            "debit_credit": direction,
            "amount": _money(fields["amount"].value) if fields["amount"].status == FOUND else None,
            "amount_status": fields["amount"].status,
            "amount_candidates": [_money(c) for c in fields["amount"].candidates if c is not None],
            "running_balance": None,
            "confidence": row_confidence,
        }
        detail = {
            "fields": {name: result.to_dict() for name, result in fields.items()},
            "suggested_category": suggestion,
        }
        return data, detail


def extractor_for(purpose: str, kb: KnowledgeBase):
    if purpose == MEMBER_RECEIPT:
        return MemberReceiptExtractor(kb)
    if purpose == VENDOR_INVOICE:
        return VendorInvoiceExtractor(kb)
    if purpose == PETTY_CASH_REGISTER:
        return RegisterExtractor(kb)
    raise ValueError(f"No local extractor for {purpose}")
