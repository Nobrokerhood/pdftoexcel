"""Transaction date normalisation with one documented output policy.

Policy (applies to every date column of the NBH sheet):

* Indian documents are day-first. `04/07/2025` is 4 July 2025, never 7 April.
* A readable date is exported as a real Excel date formatted `DD-MM-YYYY`.
* A placeholder is exported as "-".
* Anything else is NOT exported: it becomes a blocking review item so the column
  never mixes text and dates.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.accounting.money import is_placeholder

FOUND = "FOUND"
INVALID = "INVALID"
MISSING = "MISSING"

EXCEL_DATE_FORMAT = "DD-MM-YYYY"

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_name) if name})
_MONTHS["sept"] = 9

_NUMERIC = re.compile(r"^(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2}|\d{4})$")
_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?$")
_NAMED = re.compile(r"^(\d{1,2})\s*[-/. ]\s*([A-Za-z]{3,9})\.?\s*[-/., ]\s*(\d{2}|\d{4})$")
_NAMED_US = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})$")


@dataclass(frozen=True)
class DateReading:
    raw: Any
    status: str
    value: date | None = None
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.status == FOUND


def _year(text: str) -> int:
    year = int(text)
    return year + 2000 if year < 100 else year


def _build(raw, y: int, m: int, d: int) -> DateReading:
    try:
        value = date(y, m, d)
    except ValueError:
        return DateReading(raw, INVALID, reason=f"{d:02d}-{m:02d}-{y} is not a calendar date")
    if not (1990 <= value.year <= 2100):
        return DateReading(raw, INVALID, reason=f"year {value.year} is outside the plausible range")
    return DateReading(raw, FOUND, value)


def parse_date(value: Any) -> DateReading:
    if is_placeholder(value):
        return DateReading(value, MISSING, reason="no date")
    if isinstance(value, datetime):
        return DateReading(value, FOUND, value.date())
    if isinstance(value, date):
        return DateReading(value, FOUND, value)

    text = str(value).strip()
    m = _ISO.match(text)
    if m:
        return _build(value, int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _NUMERIC.match(text)
    if m:
        return _build(value, _year(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = _NAMED.match(text)
    if m:
        month = _MONTHS.get(m.group(2).lower()) or _MONTHS.get(m.group(2).lower()[:3])
        if not month:
            return DateReading(value, INVALID, reason=f"unknown month '{m.group(2)}'")
        return _build(value, _year(m.group(3)), month, int(m.group(1)))
    m = _NAMED_US.match(text)
    if m:
        month = _MONTHS.get(m.group(1).lower()) or _MONTHS.get(m.group(1).lower()[:3])
        if not month:
            return DateReading(value, INVALID, reason=f"unknown month '{m.group(1)}'")
        return _build(value, int(m.group(3)), month, int(m.group(2)))
    return DateReading(value, INVALID, reason=f"'{text}' is not a recognised date format")


def canonical_date_text(value: Any) -> str:
    """DD-MM-YYYY for a readable date, "-" for a placeholder, otherwise the raw text."""
    reading = parse_date(value)
    if reading.found:
        return reading.value.strftime("%d-%m-%Y")
    if reading.status == MISSING:
        return "-"
    return str(value).strip()
