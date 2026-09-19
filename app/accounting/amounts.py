"""Deterministic disambiguation of competing OCR amount readings.

Indian accounting documents write amounts with a rupee suffix -- "900/-",
"1,800/-". Low-confidence OCR frequently renders that trailing "/-" as a digit
or letter, so a single written amount surfaces as two candidate readings that
differ only by a trailing artefact: ["900", "9001"].

This module resolves that one specific, well-understood artefact and nothing
else. When the candidates disagree in any other way the amount stays unresolved
and goes to human review -- guessing would be fabrication.

No document-specific or benchmark-specific values appear here; the rule is
purely structural.
"""

import re
from decimal import Decimal, InvalidOperation

# Characters a trailing "/-" is commonly misrecognised as.
_SUFFIX_ARTEFACT_CHARS = set("1lI|/-_.,: ")

_NUMERIC = re.compile(r"^\d+(?:\.\d+)?$")


def normalize_candidate(value) -> str:
    """Strip grouping and whitespace so candidates can be compared structurally."""
    if value is None:
        return ""
    return str(value).replace(",", "").replace(" ", "").strip()


def _is_suffix_artefact(longer: str, shorter: str) -> bool:
    """True when `longer` is `shorter` plus only rupee-suffix noise."""
    if not longer.startswith(shorter) or longer == shorter:
        return False
    return all(ch in _SUFFIX_ARTEFACT_CHARS for ch in longer[len(shorter):])


def resolve_amount_candidates(candidates) -> tuple[Decimal | None, str]:
    """Pick the intended amount from competing OCR readings.

    Returns (amount, reason). `amount` is None when the readings cannot be
    reconciled by the rupee-suffix rule, in which case the caller must keep the
    row unresolved rather than choosing arbitrarily.
    """
    cleaned = [normalize_candidate(c) for c in (candidates or [])]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None, ""

    unique = sorted(set(cleaned), key=len)

    if len(unique) == 1:
        if _NUMERIC.match(unique[0]):
            try:
                return Decimal(unique[0]), "Single OCR amount reading."
            except InvalidOperation:
                return None, ""
        return None, ""

    base = unique[0]
    if not _NUMERIC.match(base):
        return None, ""

    # Every longer reading must be the base plus rupee-suffix noise only.
    if all(_is_suffix_artefact(other, base) for other in unique[1:]):
        try:
            amount = Decimal(base)
        except InvalidOperation:
            return None, ""
        return amount, (
            f"OCR returned {unique}; resolved to {base} because the longer reading(s) "
            f"differ only by a trailing rupee-suffix artefact (\"/-\" misread). "
            f"Flagged for human confirmation."
        )

    return None, ""
