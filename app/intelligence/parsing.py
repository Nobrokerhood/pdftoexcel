"""Deterministic readers for amounts, dates and amounts-in-words.

Readers never pick a value silently: an unclear reading returns status AMBIGUOUS
with every plausible candidate and the reason, for the verifier/repair agents and
the human reviewer to resolve.
"""

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher


FOUND = "FOUND"
AMBIGUOUS = "AMBIGUOUS"
MISSING = "MISSING"

CURRENCY = re.compile(r"₹|(?<![A-Za-z])(?:INR|Rs\.?)(?![A-Za-z])|=(?=\s*\d)", re.IGNORECASE)
PRINTED_AMOUNT = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?(?![\w])")
GROUPED = re.compile(r"^(?:\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")

MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_name) if name})
MONTHS["sept"] = 9

PRINTED_DATE_FORMATS = (
    "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%Y-%m-%d", "%d-%b-%y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y", "%b %d, %Y", "%B %d, %Y",
)
DATE_TOKEN = re.compile(
    r"\b(\d{1,2}[-/. ](?:\d{1,2}|[A-Za-z]{3,9})[-/. ,]*\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Za-z]{3,9} \d{1,2}, \d{4})\b"
)


@dataclass
class Reading:
    raw: str
    status: str
    value: object = None
    candidates: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.status == FOUND


def to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def parse_printed_amount(text: str | None) -> Decimal | None:
    """Strict reading of a printed amount such as '₹ 12,450.75' or 'Rs. 1,00,000'."""
    if text is None:
        return None
    cleaned = CURRENCY.sub("", str(text)).replace(" ", "").strip().rstrip("/-").strip()
    if not cleaned or not GROUPED.match(cleaned):
        return None
    return to_decimal(cleaned.replace(",", ""))


def find_printed_amounts(text: str) -> list[Decimal]:
    values = []
    for match in PRINTED_AMOUNT.finditer(CURRENCY.sub(" ", text)):
        value = to_decimal(match.group(0).replace(",", ""))
        if value is not None:
            values.append(value)
    return values


def _letters_to_digits(token: str, mapping: dict[str, str]) -> tuple[str, bool]:
    """Swap a single stray letter that sits next to digits ('B801' -> '8801'); words are dropped."""
    token = re.sub(r"[A-Za-z]{2,}", " ", token).strip(" -:")
    chars = list(token)
    changed = False
    for i, ch in enumerate(chars):
        if ch not in mapping:
            continue
        before = token[i - 1] if i > 0 else ""
        after = token[i + 1] if i + 1 < len(token) else ""
        if before.isdigit() or after.isdigit() or (before in ",." and before) or (after in ",." and after):
            chars[i] = mapping[ch]
            changed = True
    return "".join(chars).replace(" ", ""), changed


def read_handwritten_amount(text: str | None, letter_to_digit: dict[str, str] | None = None) -> Reading:
    """Read an amount written by hand in an Indian register (e.g. '900/-', OCR '9001-')."""
    raw = (text or "").strip()
    if not raw or not any(ch.isdigit() for ch in raw):
        return Reading(raw, MISSING)
    notes: list[str] = []
    token = CURRENCY.sub("", raw).replace(" ", "")
    token, substituted = _letters_to_digits(token, letter_to_digit or {})
    if substituted:
        notes.append("OCR letter read as digit")
    token = re.sub(r"[^\d,./\-]", "", token).lstrip("-/.,")

    dash_suffix = bool(re.search(r"[/\-]+$", token))
    token = re.sub(r"[/\-]+$", "", token)
    if not token:
        return Reading(raw, MISSING, notes=notes)

    # '/-' read as '1-' : the trailing 1 before the dash is the slash.
    if dash_suffix and token.endswith("1") and len(re.sub(r"\D", "", token)) > 1:
        body = token[:-1].rstrip(",.")
        with_one = _grouped_value(token)
        without_one = _grouped_value(body)
        if without_one is not None and (with_one is None or not _valid_grouping(token)):
            notes.append("trailing '1-' read as the '/-' suffix")
            return Reading(raw, FOUND, without_one, [without_one], notes)
        if without_one is not None and with_one is not None:
            notes.append("trailing '1' before '-' may be the '/-' suffix")
            return Reading(raw, AMBIGUOUS, None, [without_one, with_one], notes)

    value = _grouped_value(token)
    if value is not None and _valid_grouping(token):
        if not dash_suffix and token.endswith("1") and "," not in token and "." not in token and len(token) >= 3:
            without = _grouped_value(token[:-1])
            notes.append("trailing '1' may be a misread '/-' suffix")
            return Reading(raw, AMBIGUOUS, None, [without, value], notes)
        return Reading(raw, FOUND, value, [value], notes)

    # Grouping broken by one extra trailing digit, e.g. '8,4001' = '8,400/'.
    if token[-1] == "1" and _valid_grouping(token[:-1]) and ("," in token or "." in token):
        fixed = _grouped_value(token[:-1])
        notes.append("extra trailing '1' breaks digit grouping; read as the '/' suffix")
        return Reading(raw, FOUND, fixed, [fixed], notes)

    digits = re.sub(r"\D", "", token)
    if digits:
        notes.append("digit grouping not recognised")
        candidates = [to_decimal(digits)]
        if digits.endswith("1") and len(digits) > 1:
            candidates.insert(0, to_decimal(digits[:-1]))
        return Reading(raw, AMBIGUOUS, None, [c for c in candidates if c is not None], notes)
    return Reading(raw, MISSING, notes=notes)


def _valid_grouping(token: str) -> bool:
    normalised = token.replace(".", ",") if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token) else token
    return bool(GROUPED.match(normalised))


def _grouped_value(token: str) -> Decimal | None:
    if not token:
        return None
    # Handwritten '9.400' uses '.' as a thousands separator; '12.50' is a decimal.
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", token):
        token = token.replace(".", "")
    if not GROUPED.match(token):
        return None
    return to_decimal(token.replace(",", ""))


def split_merged_amounts(text: str) -> list[Decimal]:
    """'98,6231,08,300' (two totals OCR'd as one token) -> [98623, 108300]."""
    token = re.sub(r"[^\d,]", "", text or "")
    found = []
    pattern = re.compile(r"\d{1,3}(?:,\d{2})*,\d{3}|\d{1,3}(?:,\d{3})+")
    position = 0
    while position < len(token):
        match = pattern.match(token, position)
        if not match:
            position += 1
            continue
        found.append(to_decimal(match.group(0).replace(",", "")))
        position = match.end()
    return [value for value in found if value is not None]


def _expand_year(year: int) -> int:
    return 2000 + year if year < 100 else year


def parse_printed_date(text: str | None) -> date | None:
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", str(text).strip().rstrip(".,"))
    for fmt in PRINTED_DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def find_printed_dates(text: str) -> list[tuple[date, str]]:
    results = []
    for match in DATE_TOKEN.finditer(text or ""):
        parsed = parse_printed_date(match.group(1))
        if parsed:
            results.append((parsed, match.group(1)))
    return results


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(_expand_year(year), month, day)
    except ValueError:
        return None


def read_handwritten_date(text: str | None, letter_to_digit: dict[str, str] | None = None) -> Reading:
    """Read 'dd-mm-yy' as written in registers; OCR noise like '10-07-2.5' or '15107.25' is tolerated."""
    raw = (text or "").strip()
    if not raw:
        return Reading(raw, MISSING)
    printed = parse_printed_date(raw)
    if printed:
        return Reading(raw, FOUND, printed, [printed])
    notes: list[str] = []
    token, substituted = _letters_to_digits(raw.replace(" ", ""), letter_to_digit or {})
    if substituted:
        notes.append("OCR letter read as digit")
    token = re.sub(r"(?<=\d)\.(?=\d$)", "", token)  # '2.5' -> '25' at the end
    parts = [p for p in re.split(r"[-./]+", token) if p]
    candidates: list[date] = []
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        d, m, y = (int(p) for p in parts)
        if len(parts[2]) in (2, 4):
            parsed = _safe_date(y, m, d)
            if parsed:
                candidates.append(parsed)
    if not candidates:
        digits = re.sub(r"\D", "", token)
        candidates = _date_candidates_from_digits(digits)
        if candidates:
            notes.append("date separators unclear")
        if not candidates and len(digits) == 7:
            for i, ch in enumerate(digits):
                if ch == "1":
                    candidates += _date_candidates_from_digits(digits[:i] + digits[i + 1:])
            if candidates:
                notes.append("a '1' was read in place of a date separator")
    unique = sorted(set(candidates))
    if not unique:
        return Reading(raw, MISSING, notes=notes + ["no valid calendar date"])
    if len(unique) == 1:
        return Reading(raw, FOUND, unique[0], unique, notes)
    return Reading(raw, AMBIGUOUS, None, unique, notes)


def _date_candidates_from_digits(digits: str) -> list[date]:
    options = []
    if len(digits) == 6:
        options.append((digits[0:2], digits[2:4], digits[4:6]))
    elif len(digits) == 8:
        options.append((digits[0:2], digits[2:4], digits[4:8]))
    elif len(digits) == 5:
        options += [(digits[0:1], digits[1:3], digits[3:5]), (digits[0:2], digits[2:3], digits[3:5])]
    elif len(digits) == 4:
        options.append((digits[0:1], digits[1:2], digits[2:4]))
    found = []
    for d, m, y in options:
        parsed = _safe_date(int(y), int(m), int(d))
        if parsed:
            found.append(parsed)
    return found


def date_digit_variants(value: date, confusions: list[list[str]]) -> set[date]:
    """Dates reachable by swapping one commonly confused handwritten digit in dd-mm-yy."""
    text = value.strftime("%d%m%y")
    variants = set()
    for i, ch in enumerate(text):
        for a, b in confusions:
            if ch == a:
                swapped = text[:i] + b + text[i + 1:]
                parsed = _safe_date(int(swapped[4:6]), int(swapped[2:4]), int(swapped[0:2]))
                if parsed:
                    variants.add(parsed)
    return variants


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "lacs": 100000, "crore": 10000000, "crores": 10000000, "million": 1000000}


def _words_to_int(words: list[str]) -> int | None:
    total, current, seen = 0, 0, False
    for word in words:
        if word in _UNITS:
            current += _UNITS[word]
            seen = True
        elif word == "hundred":
            current = (current or 1) * 100
            seen = True
        elif word in _SCALES:
            total += (current or 1) * _SCALES[word]
            current = 0
            seen = True
        elif word in {"and", "rupees", "rupee", "rs", "inr", "only"}:
            continue
        else:
            return None
    return total + current if seen else None


def amount_from_words(text: str | None) -> Decimal | None:
    """'Rupees Twelve Thousand Four Hundred Fifty and Seventy Five Paise Only' -> 12450.75."""
    if not text:
        return None
    words = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    if "rupees" not in words and "rupee" not in words and "only" not in words:
        return None
    words = [w for w in words if w not in {"rupees", "rupee", "only", "inr", "rs"}]
    paise = 0
    if "paise" in words:
        index = words.index("paise")
        before = words[:index]
        split = max((i for i, w in enumerate(before) if w == "and"), default=None)
        if split is None:
            return None
        rupee_words, paise_words = before[:split], before[split + 1:]
        paise_value = _words_to_int(paise_words)
        if paise_value is None or paise_value >= 100:
            return None
        paise = paise_value
    else:
        rupee_words = words
    rupees = _words_to_int(rupee_words)
    if rupees is None:
        return None
    return Decimal(rupees) + (Decimal(paise) / 100)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fuzzy_contains(text: str, phrase: str, threshold: float = 0.8) -> bool:
    """True when `phrase` appears in `text` exactly or as a near match (OCR noise)."""
    lowered = (text or "").lower()
    phrase = phrase.lower().strip()
    if not phrase:
        return False
    if phrase in lowered:
        return True
    if len(phrase) < 5:
        return False
    words = re.findall(r"[a-z0-9/.]+", lowered)
    size = len(phrase.split())
    for i in range(len(words)):
        window = " ".join(words[i:i + size])
        if window and similarity(window, phrase) >= threshold:
            return True
    return False


def normalise_reference(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = re.sub(r"[\s:]+", "", str(text)).strip(".-")
    return cleaned.upper() or None
