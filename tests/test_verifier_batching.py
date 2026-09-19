"""Row-id-keyed verification batching: every row is sent, and results attach
to rows only by row_id (batch numbering can no longer collide)."""

import json
from unittest.mock import MagicMock

from app.accounting.templates import MEMBER_RECEIPT_TEMPLATE
from app.agents.verifier import MODEL_BATCH, GeminiVerificationProvider, VerificationAgent


def _rows(n):
    return [{"Amount*": f"{i * 100}", "Cheque/Ref No*": f"{260 + i}", "Transaction Date*": "04-07-2025",
             "_row_id": f"r1_{i:03d}", "_status": "UNVERIFIED"} for i in range(1, n + 1)]


def test_verifier_sends_every_row_and_attributes_only_by_row_id():
    mock_gemini = MagicMock()
    mock_gemini.settings.gemini_api_key = "fake_key"
    mock_gemini.settings.poppler_path = None
    sent_ids = []

    def generate_json(parts, purpose="json", **_):
        payload = json.loads(parts[0].split("ROWS: ", 1)[1])
        ids = [r["row_id"] for r in payload]
        sent_ids.extend(ids)
        # The model skips the last row of each batch and invents an unknown id.
        return {"overall_status": "PASSED",
                "rows": [{"row_id": rid, "status": "VERIFIED"} for rid in ids[:-1]] +
                        [{"row_id": "r9_999", "status": "VERIFIED"}]}

    mock_gemini.generate_json.side_effect = generate_json
    provider = GeminiVerificationProvider(mock_gemini, ocr_service=None)
    import app.agents.extractor as extractor_module
    extractor_module_source_parts = extractor_module.source_parts
    extractor_module.source_parts = lambda *a, **k: []
    try:
        data = {"rows": _rows(29), "document_type": "PETTY_CASH_REGISTER"}
        result = VerificationAgent(provider).verify(b"%PDF-mock", "PETTY_CASH_REGISTER", MEMBER_RECEIPT_TEMPLATE, data)
    finally:
        extractor_module.source_parts = extractor_module_source_parts

    assert sorted(sent_ids) == sorted(r["_row_id"] for r in _rows(29))       # 100% coverage
    assert mock_gemini.generate_json.call_count == -(-29 // MODEL_BATCH)       # bounded batches
    skipped = {"r1_015", "r1_029"}                                             # last of each batch
    assert {rid for rid, s in result.rows.items() if s == "UNVERIFIED"} == skipped
    assert all(result.rows[r] == "VERIFIED" for r in result.rows if r not in skipped)
    assert "r9_999" not in result.rows and any("unknown row_id" in n for n in result.notes)
    assert result.overall_status == "NEEDS_REVIEW"                             # never PASSED with gaps
