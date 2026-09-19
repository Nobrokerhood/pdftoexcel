"""
Multi-Dimensional OCR Quality Assessor.
Evaluates OCR quality across numeric confidence, date/ref confidence, garbage ratios,
abnormal token patterns, and spatial consistency.
Flags suspicious extractions to trigger adaptive preprocessing variants or human review.
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any


DATE_REGEX = re.compile(r"\b\d{1,2}[-/.](?:\d{1,2}|[A-Za-z]{3})[-/.]\d{2,4}\b")
NUMERIC_REGEX = re.compile(r"^[₹$Rs.]*\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?$")
# Abnormal tokens like digits mixed with letters: '2500O', '8161Z', '1O00'
SUSPICIOUS_DIGIT_MIX_REGEX = re.compile(r"\b\d+[A-Za-z]+\d*|\b[A-Za-z]+\d{2,}\b")


@dataclass
class OcrQualityReport:
    mean_confidence: float
    numeric_confidence: float
    date_ref_confidence: float
    garbage_ratio: float
    total_lines: int
    numeric_tokens_count: int
    abnormal_tokens: List[str] = field(default_factory=list)
    quality_rating: str = "ACCEPTABLE"  # "HIGH", "ACCEPTABLE", "SUSPICIOUS", "POOR"
    needs_adaptive_fallback: bool = False
    recommended_variant: str = "variant_a"
    needs_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OcrQualityAssessor:
    """Evaluates multi-dimensional OCR quality on extracted lines."""

    def __init__(
        self,
        min_acceptable_numeric_conf: float = 0.85,
        max_acceptable_garbage_ratio: float = 0.08
    ):
        self.min_numeric_conf = min_acceptable_numeric_conf
        self.max_garbage_ratio = max_acceptable_garbage_ratio

    def evaluate(
        self,
        lines: List[Tuple[str, float, Tuple[int, int, int, int]]]
    ) -> OcrQualityReport:
        normalized_lines = []
        for item in lines:
            if hasattr(item, "text") and hasattr(item, "confidence") and hasattr(item, "bbox"):
                normalized_lines.append((item.text, float(item.confidence), item.bbox))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                normalized_lines.append((str(item[0]), float(item[1]), item[2]))

        if not normalized_lines:
            return OcrQualityReport(
                mean_confidence=0.0,
                numeric_confidence=0.0,
                date_ref_confidence=0.0,
                garbage_ratio=1.0,
                total_lines=0,
                numeric_tokens_count=0,
                quality_rating="POOR",
                needs_adaptive_fallback=True,
                recommended_variant="variant_b",
                needs_human_review=True,
                review_reasons=["No OCR lines detected"]
            )

        total_confs = [l[1] for l in normalized_lines]
        mean_conf = round(sum(total_confs) / len(total_confs), 3)

        numeric_confs: List[float] = []
        date_confs: List[float] = []
        abnormal_tokens: List[str] = []
        garbage_token_count = 0
        total_tokens = 0

        for text, conf, bbox in normalized_lines:
            words = text.split()
            total_tokens += len(words)

            # Check dates
            if DATE_REGEX.search(text):
                date_confs.append(conf)

            for w in words:
                clean_w = w.strip("₹$,./-()[]{}")
                # Pure numeric check
                if clean_w.isdigit():
                    numeric_confs.append(conf)
                # Check for digit/letter confusion e.g. 2500O
                elif SUSPICIOUS_DIGIT_MIX_REGEX.match(w):
                    # Ignore normal alphanumeric codes like GSTIN or HSN if valid length
                    if not (len(w) == 15 and w.isalnum()):  # not standard GSTIN
                        abnormal_tokens.append(w)
                        garbage_token_count += 1
                
                # Check for high density of strange symbols
                symbol_count = sum(1 for ch in w if not ch.isalnum() and ch not in ".,/-₹")
                if symbol_count >= 2:
                    garbage_token_count += 1

        numeric_conf = round(sum(numeric_confs) / len(numeric_confs), 3) if numeric_confs else mean_conf
        date_ref_conf = round(sum(date_confs) / len(date_confs), 3) if date_confs else mean_conf
        garbage_ratio = round(garbage_token_count / max(1, total_tokens), 3)

        review_reasons = []
        needs_fallback = False
        recommended_variant = "variant_a"

        if numeric_conf < self.min_numeric_conf:
            review_reasons.append(f"Low numeric confidence: {numeric_conf:.2f} < {self.min_numeric_conf:.2f}")
            needs_fallback = True
            recommended_variant = "variant_b"  # contrast CLAHE

        if garbage_ratio > self.max_garbage_ratio:
            review_reasons.append(f"High garbage token ratio: {garbage_ratio:.2f}")
            needs_fallback = True
            recommended_variant = "variant_c"  # adaptive threshold

        if abnormal_tokens:
            review_reasons.append(f"Abnormal alphanumeric tokens detected: {abnormal_tokens[:3]}")

        # Quality Rating
        if mean_conf >= 0.95 and numeric_conf >= 0.92 and garbage_ratio <= 0.03:
            quality_rating = "HIGH"
        elif mean_conf >= 0.82 and numeric_conf >= 0.80 and garbage_ratio <= 0.08:
            quality_rating = "ACCEPTABLE"
        elif mean_conf >= 0.65:
            quality_rating = "SUSPICIOUS"
        else:
            quality_rating = "POOR"

        needs_human = len(review_reasons) > 0 or quality_rating in ["SUSPICIOUS", "POOR"]

        return OcrQualityReport(
            mean_confidence=mean_conf,
            numeric_confidence=numeric_conf,
            date_ref_confidence=date_ref_conf,
            garbage_ratio=garbage_ratio,
            total_lines=len(lines),
            numeric_tokens_count=len(numeric_confs),
            abnormal_tokens=abnormal_tokens[:5],
            quality_rating=quality_rating,
            needs_adaptive_fallback=needs_fallback,
            recommended_variant=recommended_variant,
            needs_human_review=needs_human,
            review_reasons=review_reasons
        )
