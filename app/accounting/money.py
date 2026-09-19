"""Financial amount normalisation shared by every stage (engines, validation,
reconciliation, export).

The previous universal parser was `re.sub(r"[^\\d.\\-]", "", value)`, which turned
`Rs.900` into 0.9, `2500 (105)` into 2500105 and `1.500,00` into 1.5. This module
replaces it with a parser that either reads a value it can justify or reports
why it cannot. It never guesses:

* FOUND      -- one unambiguous value.
* AMBIGUOUS  -- the text supports more than one reading; `candidates` lists them.
* INVALID    -- the text is not an amount.
* MISSING    -- a placeholder ("", "-", None, "N/A", ...).

Only FOUND values may be exported as numbers.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

FOUND = "FOUND"
AMBIGUOUS = "AMBIGUOUS"
INVALID = "INVALID"
MISSING = "MISSING"

PLACEHOLDERS = {"", "-", "--", "—", "null", "none", "n/a", "na", "unknown", "nil", "undefined", "nan"}

# Currency markers. "Rs" must not eat the start of a word ("Rsvp").
_CURRENCY = re.compile(r"₹|\bINR\b|\bRs\b\.?|\bRupees?\b|\bRs\.(?=\s*\d)", re.IGNORECASE)
# Indian "only"/"/-"/"/=" suffixes that close a written amount.
_SUFFIX = re.compile(r"(?:/-|/=|/|\bonly\b)+\s*$", re.IGNORECASE)
_DRCR = re.compile(r"\s*\b(cr|dr)\b\.?\s*$", re.IGNORECASE)

# Valid digit groupings: plain, western thousands, Indian lakh/crore.
_PLAIN = r"\d+"
_WESTERN = r"\d{1,3}(?:,\d{3})+"
_INDIAN = r"\d{1,2}(?:,\d{2})*,\d{3}"
_NUMBER = re.compile(rf"^(?:{_INDIAN}|{_WESTERN}|{_PLAIN})(?:\.\d{{1,2}})?$")


@dataclass(frozen=True)
class AmountReading:
    raw: Any
    status: str
    value: Decimal | None = None
    candidates: tuple = field(default_factory=tuple)
    reason: str = ""
    direction: str | None = None  # "CR" / "DR" when the source said so

    @property
    def found(self) -> bool:
        return self.status == FOUND


def is_placeholder(value: Any) -> bool:
    return value is None or str(value).strip().lower() in PLACEHOLDERS


def _to_decimal(text: str) -> Decimal | None:
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def parse_amount(value: Any) -> AmountReading:
    """Parse one financial amount. Never returns a value it cannot justify."""
    if is_placeholder(value):
        return AmountReading(value, MISSING, reason="no amount")
    if isinstance(value, bool):
        return AmountReading(value, INVALID, reason="boolean is not an amount")
    if isinstance(value, (int, Decimal)):
        dec = _to_decimal(str(value))
        return AmountReading(value, FOUND, dec) if dec is not None else AmountReading(value, INVALID, reason="not finite")
    if isinstance(value, float):
        dec = _to_decimal(repr(value))
        return AmountReading(value, FOUND, dec) if dec is not None else AmountReading(value, INVALID, reason="not finite")

    text = str(value).strip()
    direction = None
    m = _DRCR.search(text)
    if m:
        direction = m.group(1).upper()
        text = text[: m.start()].strip()

    negative = False
    # Accounting negative: the WHOLE value in parentheses, e.g. "(1,200.00)".
    if re.fullmatch(r"\(\s*[^()]+\s*\)", text):
        negative = True
        text = text.strip()[1:-1].strip()
    elif "(" in text or ")" in text:
        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
        return AmountReading(
            value, AMBIGUOUS,
            candidates=tuple(n.replace(",", "") for n in numbers),
            reason="amount text contains a parenthetical note; the note may be a flat/reference number, not part of the amount",
            direction=direction,
        )

    text = _CURRENCY.sub(" ", text).strip()
    text = _SUFFIX.sub("", text).strip()
    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()
    text = text.strip(" :=")

    if not text:
        return AmountReading(value, MISSING, reason="only currency markers, no digits")

    # A trailing "1-" where the source likely wrote "/-" (handwriting OCR artefact).
    if re.fullmatch(r"[\d,]+1-", text):
        body = text[:-2]
        with_one = text[:-1]
        cands = [c for c in (body, with_one) if _NUMBER.match(c)]
        return AmountReading(
            value, AMBIGUOUS, candidates=tuple(c.replace(",", "") for c in cands),
            reason="trailing '1-' may be the '/-' rupee suffix read as a digit",
            direction=direction,
        )
    text = text.rstrip("-").strip()

    compact = text.replace(" ", "")
    tokens = [t for t in re.split(r"\s+", text) if t]
    if len(tokens) > 1 and not all(_NUMBER.match(t) is None for t in tokens):
        # "1 00 000" style spacing or two separate numbers: not decidable safely.
        if not re.fullmatch(r"\d{1,3}(?: \d{2,3})+", text):
            return AmountReading(
                value, AMBIGUOUS,
                candidates=tuple(t.replace(",", "") for t in tokens if _NUMBER.match(t)),
                reason="more than one number in the amount text",
                direction=direction,
            )
        return AmountReading(value, AMBIGUOUS, candidates=(compact,),
                             reason="space-separated digit groups", direction=direction)

    if re.search(r"[A-Za-z]", compact):
        return AmountReading(value, INVALID, reason="letters inside the amount", direction=direction)

    # Decimal comma with dot thousands ("1.500,00") or any mixed-order separators.
    if "," in compact and "." in compact and compact.rfind(",") > compact.rfind("."):
        return AmountReading(
            value, AMBIGUOUS,
            candidates=(compact.replace(".", "").replace(",", "."),),
            reason="decimal-comma notation (e.g. 1.500,00) is not used on Indian documents; confirm the value",
            direction=direction,
        )
    # A dot followed by exactly three digits and nothing else ("1.500") may be a
    # thousands separator; the value would differ by 1000x.
    if re.fullmatch(r"\d{1,3}\.\d{3}", compact):
        return AmountReading(
            value, AMBIGUOUS,
            candidates=(compact, compact.replace(".", "")),
            reason="dot followed by three digits may be a thousands separator",
            direction=direction,
        )
    if not _NUMBER.match(compact):
        return AmountReading(value, INVALID, reason="digit grouping is not a valid amount format", direction=direction)

    dec = _to_decimal(compact.replace(",", ""))
    if dec is None:
        return AmountReading(value, INVALID, reason="not a number", direction=direction)
    if negative:
        dec = -dec
    return AmountReading(value, FOUND, dec, (str(dec),), direction=direction)


def amount_or_none(value: Any) -> Decimal | None:
    """The value only when it is unambiguous; otherwise None."""
    reading = parse_amount(value)
    return reading.value if reading.found else None


def format_amount(value: Decimal | None) -> str:
    """Canonical text form used in NBH rows: no grouping, no trailing zeros."""
    if value is None:
        return "-"
    text = format(value.quantize(Decimal("0.01")), "f")
    if text.endswith(".00"):
        text = text[:-3]
    return text
