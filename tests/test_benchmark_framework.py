"""
Unit tests for benchmark framework and source-verified ground truth integrity.
"""

import os
import pytest
from app.audit.benchmark_framework import BenchmarkEvaluator, RowDetectionMetrics, FieldScore


GROUND_TRUTH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "testing_folder", "testing_folder_ground_truth.json")


def test_ground_truth_file_exists_and_valid():
    assert os.path.exists(GROUND_TRUTH_PATH)
    evaluator = BenchmarkEvaluator(GROUND_TRUTH_PATH)
    assert "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf" in evaluator.ground_truth
    assert "sample-radhakrishna.pdf" in evaluator.ground_truth
    assert "unnamed.jpg" in evaluator.ground_truth


def test_benchmark_evaluator_golden_document():
    evaluator = BenchmarkEvaluator(GROUND_TRUTH_PATH)
    doc_name = "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf"
    
    # Mock extracted rows containing the critical numerals
    mock_rows = [
        {"Amount*": "2500.0", "Cheque/Ref No*": "300"},
        {"Amount*": "1789.0", "Cheque/Ref No*": "302"},
        {"Amount*": "750.0", "Cheque/Ref No*": "307"},
        {"Amount*": "10000.0", "Cheque/Ref No*": "290"},
    ]
    # Add dummy rows to reach 29
    for _ in range(25):
        mock_rows.append({"Amount*": "100.0"})

    mock_recon = {
        "closing_balance_difference": 3.0,
        "inflow_status": "MATCHED"
    }

    result = evaluator.evaluate_document(
        filename=doc_name,
        predicted_purpose="PETTY_CASH_REGISTER",
        predicted_doc_type="CASH_BOOK",
        predicted_rows=mock_rows,
        reconciliation_summary=mock_recon,
        elapsed_ms=120.0
    )

    assert result.purpose_match is True
    assert result.doc_type_match is True
    assert result.row_metrics.ground_truth_rows == 29
    assert result.row_metrics.detected_rows == 29
    assert result.row_metrics.precision == 1.0
    assert result.row_metrics.recall == 1.0
    assert result.amount_score.correct == 4
    assert result.discrepancies_preserved is True
    assert result.needs_human_review is False


def test_benchmark_evaluator_catches_silently_reconciled_discrepancy():
    evaluator = BenchmarkEvaluator(GROUND_TRUTH_PATH)
    doc_name = "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf"
    
    # Incorrect closing diff: silently forced to 0.0 instead of 3.0
    mock_recon = {
        "closing_balance_difference": 0.0,
        "inflow_status": "MATCHED"
    }

    result = evaluator.evaluate_document(
        filename=doc_name,
        predicted_purpose="PETTY_CASH_REGISTER",
        predicted_doc_type="CASH_BOOK",
        predicted_rows=[],
        reconciliation_summary=mock_recon
    )

    assert result.discrepancies_preserved is False
    assert result.needs_human_review is True
    assert any("Reconciliation discrepancy mismatch" in r for r in result.review_reasons)
