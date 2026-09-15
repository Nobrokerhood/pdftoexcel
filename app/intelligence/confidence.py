"""Confidence engine: combines OCR quality, pattern checks, cross-checks and verification into HIGH/MEDIUM/LOW."""

from app.intelligence.evidence import AMBIGUOUS, HIGH, LOW, MEDIUM, MISSING, FieldResult


HIGH_THRESHOLD = 0.85
MEDIUM_THRESHOLD = 0.6


def level(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return HIGH
    if score >= MEDIUM_THRESHOLD:
        return MEDIUM
    return LOW


def assess(
    result: FieldResult,
    *,
    ocr_confidence: float | None,
    pattern_ok: bool | None = None,
    label_anchored: bool = False,
    corrections: int = 0,
) -> FieldResult:
    """Initial confidence after extraction."""
    reasons = list(result.reasons)
    if result.status == MISSING:
        result.score, result.confidence = 0.0, LOW
        return result
    score = ocr_confidence if ocr_confidence is not None else 0.5
    reasons.append(f"OCR confidence {score:.2f}")
    if label_anchored:
        score += 0.05
        reasons.append("value found next to its label")
    if pattern_ok is True:
        reasons.append("value matches the expected format")
    elif pattern_ok is False:
        score -= 0.25
        reasons.append("value does not match the expected format")
    if corrections:
        score -= 0.15 * corrections
        reasons.append(f"{corrections} OCR correction(s) applied while reading")
    if result.status == AMBIGUOUS:
        score = min(score, 0.4)
        reasons.append("reading is ambiguous")
    result.score = round(max(0.0, min(1.0, score)), 3)
    result.confidence = level(result.score)
    result.reasons = reasons
    return result


def apply_check(result: FieldResult, passed: bool | None, description: str, weight: float = 0.1) -> FieldResult:
    """Adjust confidence after an independent check (second OCR engine, arithmetic, rules)."""
    if passed is None or result.status == MISSING:
        return result
    if passed:
        result.score = min(1.0, result.score + weight)
        result.reasons.append(f"confirmed: {description}")
    else:
        result.score = max(0.0, result.score - max(0.4, weight))
        result.reasons.append(f"contradicted: {description}")
    if result.status == AMBIGUOUS:
        result.score = min(result.score, 0.4)
    result.confidence = level(result.score)
    return result
