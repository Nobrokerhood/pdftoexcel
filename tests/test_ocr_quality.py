"""
Unit tests for OcrQualityAssessor.
"""

import pytest
from app.documents.ocr_quality import OcrQualityAssessor


def test_ocr_quality_high_rating():
    assessor = OcrQualityAssessor()
    lines = [
        ("STATEMENT OF ACCOUNT", 0.99, (10, 10, 200, 30)),
        ("ACCOUNT NO 10178263032", 0.98, (10, 40, 200, 60)),
        ("Date 01-04-2025 Amount 50000.00", 0.97, (10, 70, 200, 90))
    ]
    report = assessor.evaluate(lines)
    assert report.quality_rating == "HIGH"
    assert report.numeric_confidence >= 0.95
    assert report.garbage_ratio == 0.0
    assert report.needs_human_review is False


def test_ocr_quality_flags_abnormal_digit_mixing():
    assessor = OcrQualityAssessor()
    lines = [
        ("Cash payment 2500O to Kiranam", 0.82, (10, 10, 200, 30)),  # '2500O' mixed O with digits
        ("Ref 284 Amount 10000", 0.88, (10, 40, 200, 60))
    ]
    report = assessor.evaluate(lines)
    assert "2500O" in report.abnormal_tokens
    assert report.needs_human_review is True
    assert any("Abnormal alphanumeric tokens" in r for r in report.review_reasons)


def test_ocr_quality_low_numeric_confidence_triggers_fallback():
    assessor = OcrQualityAssessor(min_acceptable_numeric_conf=0.85)
    lines = [
        ("Faint handwritten payment 750", 0.60, (10, 10, 200, 30)),
        ("Total 1200", 0.65, (10, 40, 200, 60))
    ]
    report = assessor.evaluate(lines)
    assert report.numeric_confidence < 0.70
    assert report.needs_adaptive_fallback is True
    assert report.recommended_variant == "variant_b"
    assert report.needs_human_review is True
