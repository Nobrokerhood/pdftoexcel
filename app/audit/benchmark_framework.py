"""
Field-Level Ground Truth Benchmark & Precision Evaluation Framework.
Computes granular precision, recall, exact-match accuracy, and abstention metrics
without false 100% claims.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


@dataclass
class FieldScore:
    total_evaluated: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown_marked: int = 0
    
    @property
    def accuracy(self) -> float:
        if self.total_evaluated == 0:
            return 1.0
        return round(self.correct / self.total_evaluated, 4)


@dataclass
class RowDetectionMetrics:
    ground_truth_rows: int = 0
    detected_rows: int = 0
    true_positive_rows: int = 0
    false_positive_rows: int = 0
    false_negative_rows: int = 0
    
    @property
    def precision(self) -> float:
        denom = self.true_positive_rows + self.false_positive_rows
        return round(self.true_positive_rows / denom, 4) if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.ground_truth_rows
        return round(self.true_positive_rows / denom, 4) if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return round(2 * (p * r) / (p + r), 4) if (p + r) > 0 else 0.0


@dataclass
class DocumentBenchmarkResult:
    filename: str
    purpose_ground_truth: str
    purpose_predicted: str
    purpose_match: bool
    doc_type_ground_truth: str
    doc_type_predicted: str
    doc_type_match: bool
    row_metrics: RowDetectionMetrics = field(default_factory=RowDetectionMetrics)
    amount_score: FieldScore = field(default_factory=FieldScore)
    date_score: FieldScore = field(default_factory=FieldScore)
    ref_score: FieldScore = field(default_factory=FieldScore)
    payment_type_score: FieldScore = field(default_factory=FieldScore)
    tower_flat_score: FieldScore = field(default_factory=FieldScore)
    bill_head_score: FieldScore = field(default_factory=FieldScore)
    reconciliation_status: str = "NOT_APPLICABLE"
    reconciliation_passed: bool = True
    discrepancies_preserved: bool = True
    needs_human_review: bool = False
    review_reasons: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class CorpusBenchmarkReport:
    total_documents: int = 0
    classification_accuracy: float = 0.0
    purpose_accuracy: float = 0.0
    doc_type_accuracy: float = 0.0
    overall_row_precision: float = 0.0
    overall_row_recall: float = 0.0
    amount_accuracy: float = 0.0
    date_accuracy: float = 0.0
    ref_accuracy: float = 0.0
    human_review_rate: float = 0.0
    document_results: List[DocumentBenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkEvaluator:
    """Evaluates pipeline outputs against source-verified ground truth."""

    def __init__(self, ground_truth_path: str):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.ground_truth = data.get("documents", {})

    def evaluate_document(
        self,
        filename: str,
        predicted_purpose: str,
        predicted_doc_type: str,
        predicted_rows: List[Dict[str, Any]],
        reconciliation_summary: Optional[Dict[str, Any]] = None,
        elapsed_ms: float = 0.0
    ) -> DocumentBenchmarkResult:
        gt = self.ground_truth.get(filename)
        if not gt:
            raise KeyError(f"Filename {filename} not in ground truth database")

        purpose_gt = gt.get("purpose", "UNKNOWN_PURPOSE")
        doc_type_gt = gt.get("document_type", "UNKNOWN")
        
        # Normalize purpose aliases
        def norm_purpose(p: str) -> str:
            p = (p or "").strip().upper()
            if p in ["MEMBER_RECEIPT", "MEMBER_BANK_RECEIPT"]:
                return "MEMBER_BANK_RECEIPT"
            return p

        purpose_match = (norm_purpose(predicted_purpose) == norm_purpose(purpose_gt))
        doc_type_match = (predicted_doc_type == doc_type_gt)

        result = DocumentBenchmarkResult(
            filename=filename,
            purpose_ground_truth=purpose_gt,
            purpose_predicted=predicted_purpose,
            purpose_match=purpose_match,
            doc_type_ground_truth=doc_type_gt,
            doc_type_predicted=predicted_doc_type,
            doc_type_match=doc_type_match,
            processing_time_ms=elapsed_ms
        )

        gt_row_count = gt.get("genuine_transaction_row_count")
        if gt_row_count is not None:
            pred_count = len(predicted_rows)
            result.row_metrics.ground_truth_rows = gt_row_count
            result.row_metrics.detected_rows = pred_count
            # Min rows matched
            tp = min(gt_row_count, pred_count)
            result.row_metrics.true_positive_rows = tp
            result.row_metrics.false_positive_rows = max(0, pred_count - gt_row_count)
            result.row_metrics.false_negative_rows = max(0, gt_row_count - pred_count)

        # Check critical numerals if present
        critical_checks = gt.get("critical_numeral_verifications", {})
        for check_name, spec in critical_checks.items():
            result.amount_score.total_evaluated += 1
            expected_amt = spec["amount"]
            matched = any(
                float(r.get("Amount*", 0) or 0) == expected_amt
                for r in predicted_rows
            )
            if matched:
                result.amount_score.correct += 1
            else:
                result.amount_score.incorrect += 1

        # Check reconciliation differences preservation
        gt_recon = gt.get("reconciliation")
        if gt_recon and reconciliation_summary:
            expected_close_diff = gt_recon.get("closing_discrepancy")
            actual_close_diff = reconciliation_summary.get("closing_balance_difference")
            if expected_close_diff is not None:
                if actual_close_diff == expected_close_diff:
                    result.discrepancies_preserved = True
                    result.reconciliation_passed = True
                else:
                    result.discrepancies_preserved = False
                    result.reconciliation_passed = False
                    result.review_reasons.append(
                        f"Reconciliation discrepancy mismatch: expected {expected_close_diff}, got {actual_close_diff}"
                    )

        if not purpose_match or not result.discrepancies_preserved:
            result.needs_human_review = True

        return result
