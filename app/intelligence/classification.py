"""Document classification agent: what is this document, and does it fit the selected purpose?"""

import math
import re
from dataclasses import asdict, dataclass, field

from app.accounting.purposes import (
    DOCUMENT_TYPE_LABELS,
    UNKNOWN,
    purpose_definition,
    purpose_for_document_type,
)
from app.documents.ocr import DocumentRepresentation, group_rows
from app.intelligence.knowledge import KnowledgeBase
from app.intelligence.parsing import (
    FOUND,
    find_printed_amounts,
    fuzzy_contains,
    read_handwritten_amount,
    read_handwritten_date,
)


PURPOSE_MATCH = "MATCH"
PURPOSE_MISMATCH = "DOCUMENT_PURPOSE_MISMATCH"
TYPE_UNCERTAIN = "DOCUMENT_TYPE_UNCERTAIN"

MONTH_HEADER = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s\-/'.,]*(20\d{2}|\d{2})\b", re.IGNORECASE
)
GSTIN = re.compile(r"\b[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
DATE_LIKE = re.compile(r"\d{1,2}\s*[-./]\s*\d{1,2}\s*[-./]?\s*\d{1,2}(?:\.?\d)?|\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}")
SHORT_KEYWORD_BOUNDARY = 4


@dataclass
class StructuralSignals:
    dated_amount_rows: int = 0
    voucher_numbers: list[int] = field(default_factory=list)
    voucher_sequence: bool = False
    balance_line: str | None = None
    month_header: str | None = None
    flat_reference: str | None = None
    gstin: str | None = None
    tax_breakup: bool = False
    dr_cr_columns: bool = False
    distinct_amounts: int = 0
    handwritten: bool = False

    def flags(self, many_rows: int) -> dict[str, bool]:
        return {
            "many_dated_amount_rows": self.dated_amount_rows >= many_rows,
            "voucher_sequence": self.voucher_sequence,
            "balance_line": bool(self.balance_line),
            "month_header": bool(self.month_header),
            "flat_reference": bool(self.flat_reference),
            "gstin_present": bool(self.gstin),
            "tax_breakup": self.tax_breakup,
            "dr_cr_columns": self.dr_cr_columns,
            "single_primary_amount": self.dated_amount_rows < 3 and 0 < self.distinct_amounts <= 6,
            "handwritten": self.handwritten,
        }


@dataclass
class ClassificationResult:
    selected_purpose: str
    detected_type: str
    detected_label: str
    confidence: float
    purpose_status: str
    scores: dict[str, float]
    evidence: list[str]
    recommended_action: str
    recommended_purpose: str | None
    signals: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _keyword_present(text: str, keyword: str) -> bool:
    keyword = keyword.strip().lower()
    if len(keyword) <= SHORT_KEYWORD_BOUNDARY:
        return bool(re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", text))
    return fuzzy_contains(text, keyword, threshold=0.8)


def structural_signals(representation: DocumentRepresentation, kb: KnowledgeBase) -> StructuralSignals:
    letters = kb.vocabulary["handwriting"]["letter_to_digit"]
    signals = StructuralSignals(handwritten=any(page.script == "HANDWRITTEN_LIKELY" for page in representation.pages))
    text = representation.text
    amounts = set()
    voucher_by_row = []

    for page in representation.pages:
        for row in group_rows(page.lines, overlap=0.3):
            has_date = False
            has_amount = False
            for line in row:
                token = line.text.strip()
                if DATE_LIKE.search(token) and read_handwritten_date(DATE_LIKE.search(token).group(0), letters).status != "MISSING":
                    has_date = True
                    continue
                if re.fullmatch(r"\d{3,4}", token):
                    voucher_by_row.append((line.page, line.x0, int(token)))
                reading = read_handwritten_amount(token, letters) if re.search(r"\d{2,}", token) else None
                if reading and reading.candidates and not re.fullmatch(r"\d{1,4}", token):
                    has_amount = True
            if has_date and has_amount:
                signals.dated_amount_rows += 1
        top = [line.text for line in page.lines if line.bbox[1] < page.height * 0.2]
        for candidate in top:
            match = MONTH_HEADER.search(candidate)
            if match and not signals.month_header:
                signals.month_header = candidate.strip()

    for line in representation.all_lines:
        lowered = line.text.lower()
        if not signals.balance_line and (fuzzy_contains(lowered, "balance", 0.7) or "carry forward" in lowered or "b/f" in lowered):
            signals.balance_line = line.text.strip()
        amounts.update(find_printed_amounts(line.text))

    # Voucher numbers share one column; amounts like 1800 sit elsewhere on the page.
    columns: dict[tuple[int, int], list[int]] = {}
    for page_number, x0, value in voucher_by_row:
        columns.setdefault((page_number, x0 // 80), []).append(value)
    numbers = max(columns.values(), key=len) if columns else []
    signals.voucher_numbers = numbers
    if len(numbers) >= 5:
        increasing = sum(1 for a, b in zip(numbers, numbers[1:]) if 0 < b - a <= 10)
        signals.voucher_sequence = increasing >= 0.6 * (len(numbers) - 1)

    for pattern in kb.vocabulary["flat_patterns"]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            signals.flat_reference = match.group(0)
            break
    gstin = GSTIN.search(text.upper().replace(" ", ""))
    signals.gstin = gstin.group(0) if gstin else None
    lowered = text.lower()
    signals.tax_breakup = sum(1 for tax in ("cgst", "sgst", "igst") if tax in lowered) >= 1 and bool(amounts)
    signals.dr_cr_columns = ("debit" in lowered and "credit" in lowered) or bool(
        re.search(r"(?<![a-z])dr(?![a-z])", lowered) and re.search(r"(?<![a-z])cr(?![a-z])", lowered)
    )
    signals.distinct_amounts = len(amounts)
    return signals


class DocumentClassifier:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def classify(self, representation: DocumentRepresentation, selected_purpose: str) -> ClassificationResult:
        config = self.kb.document_types
        thresholds = config["thresholds"]
        text = representation.text.lower()
        signals = structural_signals(representation, self.kb)
        flags = signals.flags(thresholds["many_rows"])

        scores: dict[str, float] = {}
        matched_keywords: dict[str, list[str]] = {}
        for doc_type, definition in config["types"].items():
            score = 0.0
            hits = []
            for keyword, weight in definition["keywords"].items():
                if _keyword_present(text, keyword):
                    score += weight
                    hits.append(keyword)
            for flag, weight in definition.get("structure", {}).items():
                if flags.get(flag):
                    score += weight
            scores[doc_type] = round(score, 2)
            matched_keywords[doc_type] = hits

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_type, best_score = ranked[0]
        confidence = self._confidence([score for _, score in ranked])
        if best_score < thresholds["unknown_below_score"]:
            best_type, confidence = UNKNOWN, round(1 - confidence, 2) if best_score > 0 else 1.0

        evidence = self._evidence(best_type, signals, matched_keywords.get(best_type, []), representation)
        selected = purpose_definition(selected_purpose)
        recommended = purpose_for_document_type(best_type)

        if best_type == UNKNOWN:
            status = TYPE_UNCERTAIN
            action = "Document type could not be determined. Review the source carefully before approving."
        elif selected and best_type in selected.accepts:
            status = PURPOSE_MATCH
            action = "None."
        elif confidence >= thresholds["minimum_confidence_for_mismatch"]:
            status = PURPOSE_MISMATCH
            if recommended:
                action = f"Reject this job and upload the document again with purpose '{recommended.label}'."
            else:
                action = (
                    f"'{DOCUMENT_TYPE_LABELS.get(best_type, best_type)}' is not supported by any configured purpose. "
                    "Reject this job."
                )
        else:
            status = TYPE_UNCERTAIN
            action = "Document type is uncertain. Confirm that it really is the selected purpose before approving."

        return ClassificationResult(
            selected_purpose=selected_purpose,
            detected_type=best_type,
            detected_label=DOCUMENT_TYPE_LABELS.get(best_type, best_type),
            confidence=confidence,
            purpose_status=status,
            scores=scores,
            evidence=evidence,
            recommended_action=action,
            recommended_purpose=recommended.code if recommended and status == PURPOSE_MISMATCH else None,
            signals={**asdict(signals), "voucher_numbers": signals.voucher_numbers[:60]},
        )

    @staticmethod
    def _confidence(scores: list[float], temperature: float = 3.0) -> float:
        exps = [math.exp(min(score, 60) / temperature) for score in scores]
        return round(exps[0] / sum(exps), 2)

    @staticmethod
    def _evidence(doc_type: str, signals: StructuralSignals, keywords: list[str], representation: DocumentRepresentation) -> list[str]:
        evidence = [f"{len(representation.pages)} page(s), {len(representation.all_lines)} text lines read by OCR"]
        if signals.dated_amount_rows:
            evidence.append(f"{signals.dated_amount_rows} rows contain both a date and an amount")
        if signals.voucher_sequence:
            numbers = signals.voucher_numbers
            evidence.append(f"voucher numbers in sequence ({min(numbers)}–{max(numbers)}, {len(numbers)} found)")
        if signals.balance_line:
            evidence.append(f"balance line detected: \"{signals.balance_line}\"")
        if signals.month_header:
            evidence.append(f"period header detected: \"{signals.month_header}\"")
        if signals.handwritten:
            evidence.append("handwriting likely (low OCR confidence pattern)")
        if signals.gstin:
            evidence.append(f"GSTIN present: {signals.gstin}")
        if signals.tax_breakup:
            evidence.append("GST breakup (CGST/SGST/IGST) present")
        if signals.flat_reference:
            evidence.append(f"flat reference found: \"{signals.flat_reference}\"")
        elif doc_type != "MEMBER_RECEIPT":
            evidence.append("no member flat/tower reference found")
        if keywords:
            evidence.append("keywords: " + ", ".join(sorted(keywords)[:8]))
        return evidence
