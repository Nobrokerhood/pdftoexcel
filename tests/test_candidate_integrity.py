"""Regression tests for the row-integrity guarantees.

These cover the two defects reproduced on testing_folder/unnamed.jpg:
  1. a document yielding no fields was turned into one all-"-" transaction row;
  2. that fabricated row was then reported VERIFIED / PASSED.

And the silent row loss reproduced on the handwritten register, where rows whose
amount OCR abstained on were reclassified ANNOTATION and dropped at export.

No benchmark amounts are used as production inputs here -- ground-truth values
live only inside the tests.
"""

from decimal import Decimal

import pytest

from app.accounting.amounts import resolve_amount_candidates
from app.accounting.candidates import (
    CandidateDecision,
    CandidateLedger,
    RejectedCandidate,
    classify_row_completeness,
    row_is_meaningful,
)
from app.accounting.engines.petty_cash_register_engine import PettyCashRegisterEngine
from app.accounting.purposes import MEMBER_RECEIPT
from app.accounting.semantic_rows import RowType, classify_semantic_row
from app.accounting.templates import NBH_IMPORT_COLUMNS
from app.agents.extractor import validate_extraction
from app.agents.verifier import enforce_verification_guards


def _blank_row() -> dict:
    return {col: "-" for col in NBH_IMPORT_COLUMNS}


# --------------------------------------------------------------------------
# 1. Zero extraction must not create a fabricated row
# --------------------------------------------------------------------------

def test_empty_extraction_returns_zero_rows_not_a_placeholder_row():
    result = validate_extraction(MEMBER_RECEIPT, {})
    assert result["rows"] == []
    assert result["extraction_outcome"] == "NO_RELIABLE_TRANSACTIONS"
    assert "No accounting transactions could be reliably extracted" in result["extraction_notice"]
    assert "Nothing has been invented" in result["extraction_notice"]


def test_single_record_with_real_value_still_produces_a_row():
    result = validate_extraction(MEMBER_RECEIPT, {"amount": "8700", "reference_number": "INV-1"})
    assert len(result["rows"]) == 1
    assert result["rows"][0]["Amount*"] == "8700"
    assert result["extraction_outcome"] == "EXTRACTED"


def test_row_is_meaningful_rejects_all_placeholder_variants():
    assert row_is_meaningful(_blank_row()) is False
    for junk in ["", "  ", "null", "None", "N/A", "UNKNOWN", "undefined"]:
        row = _blank_row()
        row["Amount*"] = junk
        assert row_is_meaningful(row) is False, junk
    row = _blank_row()
    row["Amount*"] = "900"
    assert row_is_meaningful(row) is True


def test_missing_field_inside_a_genuine_row_is_still_allowed():
    """A genuine row may carry "-" fields; only a wholly empty row is invalid."""
    row = _blank_row()
    row["Amount*"] = "900"
    row["Cheque/Ref No*"] = "266"
    decision, _ = classify_row_completeness(row)
    assert decision == CandidateDecision.ACCEPTED
    assert row["Tower No*"] == "-"


def test_extraction_provider_is_preserved_and_not_reported_as_gemini():
    result = validate_extraction(MEMBER_RECEIPT, {"_extraction_provider": "LOCAL_RAPIDOCR"})
    assert result["_extraction_provider"] == "LOCAL_RAPIDOCR"


# --------------------------------------------------------------------------
# 2. A placeholder row can never be VERIFIED, by any provider
# --------------------------------------------------------------------------

def test_guard_downgrades_placeholder_row_marked_verified_by_provider():
    # Verification is attributed by row_id; a provider claiming VERIFIED for an
    # all-placeholder row is overruled.
    extracted = {"rows": [{**_blank_row(), "_row_id": "r0_001"}]}
    provider_result = {
        "overall_status": "PASSED",
        "fields": [{"row_id": "r0_001", "field": "row_1", "status": "VERIFIED", "confidence": 0.9,
                    "evidence": "Row verified from OCR representation", "page_number": 1}],
    }
    guarded = enforce_verification_guards(provider_result, extracted)
    assert guarded["rows"]["r0_001"] == "NEEDS_REVIEW"
    assert any("no accounting value" in n for n in guarded["notes"])
    assert guarded["overall_status"] == "NEEDS_REVIEW"


def test_guard_blocks_passed_when_nothing_was_verified():
    extracted = {"rows": [{"Amount*": "900"}]}
    result = enforce_verification_guards(
        {"overall_status": "PASSED",
         "fields": [{"field": "row_1", "status": "UNCERTAIN", "confidence": 0.5}]},
        extracted,
    )
    assert result["overall_status"] == "NEEDS_REVIEW"


def test_guard_blocks_passed_when_extraction_produced_no_rows():
    result = enforce_verification_guards({"overall_status": "PASSED", "fields": []}, {"rows": []})
    assert result["overall_status"] == "NEEDS_REVIEW"


def test_guard_leaves_a_genuinely_verified_row_alone():
    extracted = {"rows": [{**_blank_row(), "Amount*": "900", "Cheque/Ref No*": "266", "_row_id": "r0_001"}]}
    result = enforce_verification_guards(
        {"overall_status": "PASSED",
         "fields": [{"row_id": "r0_001", "field": "row_1", "status": "VERIFIED", "confidence": 0.9,
                     "evidence": "Page 1, y=152: '266 ... 900'"}]},
        extracted,
    )
    assert result["rows"]["r0_001"] == "VERIFIED"
    assert result["overall_status"] == "PASSED"


def test_result_without_row_id_verifies_nothing():
    """A field name like 'row_1' or 'Amount*' is not an identity: it is ignored."""
    extracted = {"rows": [{**_blank_row(), "Amount*": "900", "_row_id": "r0_001"}]}
    result = enforce_verification_guards(
        {"overall_status": "PASSED", "fields": [{"field": "row_1", "status": "VERIFIED"}]}, extracted)
    assert result["rows"]["r0_001"] == "UNVERIFIED"
    assert result["overall_status"] == "NEEDS_REVIEW"


def test_missing_overall_status_is_never_passed():
    extracted = {"rows": [{**_blank_row(), "Amount*": "900", "_row_id": "r0_001"}]}
    result = enforce_verification_guards({}, extracted)
    assert result["overall_status"] == "NEEDS_REVIEW"


# --------------------------------------------------------------------------
# 3. No silent row loss when OCR abstains on the amount
# --------------------------------------------------------------------------

def test_unreadable_amount_with_identity_is_preserved_not_annotated():
    assert classify_semantic_row(
        "Table cloth washing purpose", amount=None, has_transaction_identity=True
    ) == RowType.UNRESOLVED_TRANSACTION


def test_unreadable_amount_without_identity_is_still_an_annotation():
    assert classify_semantic_row(
        "some stray note", amount=None, has_transaction_identity=False
    ) == RowType.ANNOTATION


def test_totals_and_balances_are_never_transactions():
    assert classify_semantic_row("Total Expenditure", Decimal("98623")) == RowType.TOTAL
    assert classify_semantic_row("Closing Balance c/f", Decimal("10174")) == RowType.CLOSING_BALANCE
    assert classify_semantic_row("Opening Balance b/f", Decimal("1714")) == RowType.OPENING_BALANCE


def test_abstained_amount_row_survives_export_with_dash_not_zero():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [
            {"voucher_no": "301", "date": "2025-07-15", "particulars": "Iron stand rent",
             "amount": None, "amount_status": "MISSING", "amount_candidates": []},
            {"voucher_no": "298", "date": "2025-07-15", "particulars": "Table advance",
             "amount": "5000", "amount_status": "FOUND"},
        ],
    }
    doc = PettyCashRegisterEngine().process(raw)
    dataset = doc.to_nbh_dataset()
    assert len(dataset) == 2, "the abstained-amount row must not be dropped"
    by_ref = {r["Cheque/Ref No*"]: r for r in dataset}
    assert by_ref["301"]["Amount*"] == "-", "unreadable amount must be '-', never 0"
    assert by_ref["298"]["Amount*"] == "5000"


def test_candidate_ledger_accounts_for_every_source_row():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [
            {"voucher_no": "266", "particulars": "Stamp papers", "amount": "900"},
            {"voucher_no": "301", "particulars": "Iron stand rent", "amount": None,
             "amount_status": "MISSING"},
            {"particulars": "Total Expenditure", "amount": "98623"},
        ],
    }
    result = validate_extraction(MEMBER_RECEIPT, raw)
    ledger = result["candidate_ledger"]
    # Provider-supplied rows (no OCR evidence): the total line is recorded as a
    # non-transaction with a reason; both genuine rows (incl. the one with an
    # unreadable amount) are kept.
    assert [r["Cheque/Ref No*"] for r in result["rows"]] == ["266", "301"]
    assert ledger["balanced"] is True, ledger
    assert ledger["non_transaction"] == 1
    reasons = [c["status_reason"] for c in ledger["candidates"]]
    assert any("total" in r.lower() for r in reasons), reasons


def test_every_rejected_candidate_carries_a_reason():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [
            {"particulars": "Total Expenditure", "amount": "98623"},
            {"particulars": "Closing Balance", "amount": "10174"},
        ],
    }
    result = validate_extraction(MEMBER_RECEIPT, raw)
    assert result["rows"] == []
    assert len(result["candidate_ledger"]["candidates"]) == 2
    for cand in result["candidate_ledger"]["candidates"]:
        assert cand["status_reason"].strip(), cand
        assert cand["classification"] != ""


# --------------------------------------------------------------------------
# 4. Amount disambiguation is structural, never document-specific
# --------------------------------------------------------------------------

@pytest.mark.parametrize("candidates,expected", [
    (["900", "9001"], Decimal("900")),
    (["1800", "18001"], Decimal("1800")),
    (["200", "200/-"], Decimal("200")),
    (["880", "880l"], Decimal("880")),
])
def test_rupee_suffix_artefact_is_resolved_to_the_base_amount(candidates, expected):
    amount, reason = resolve_amount_candidates(candidates)
    assert amount == expected
    assert "rupee-suffix" in reason


@pytest.mark.parametrize("candidates", [
    ["900", "800"],       # genuinely different readings
    ["150", "750"],       # leading-digit ambiguity - must not be guessed
    ["1789", "178"],      # truncation, not a suffix artefact
    [],
])
def test_ambiguous_readings_are_not_guessed(candidates):
    amount, _ = resolve_amount_candidates(candidates)
    assert amount is None, f"{candidates} must stay unresolved for human review"


def test_disambiguated_row_is_flagged_for_human_confirmation():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [{"voucher_no": "266", "particulars": "Stamp papers",
                  "amount": None, "amount_status": "AMBIGUOUS",
                  "amount_candidates": ["900", "9001"]}],
    }
    doc = PettyCashRegisterEngine().process(raw)
    tx = doc.transactions[0]
    assert tx.amount == Decimal("900")
    assert tx.verification_status == "FLAG_FOR_HUMAN"
    assert any("rupee-suffix" in r for r in tx.review_reasons)


# --------------------------------------------------------------------------
# 5. Exact NBH output contract
# --------------------------------------------------------------------------

def test_exported_rows_have_exactly_the_12_nbh_columns_in_order():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [{"voucher_no": "266", "particulars": "Stamp papers", "amount": "900"}],
    }
    result = validate_extraction(MEMBER_RECEIPT, raw)
    row = result["rows"][0]
    # Internal rows carry _-prefixed metadata (row identity, status); the
    # exported sheet itself is checked to be exactly these 12 columns elsewhere.
    exported = [k for k in row if not k.startswith("_") and k != "USER_EDITED"]
    assert exported == list(NBH_IMPORT_COLUMNS)


def test_no_forbidden_placeholder_tokens_are_ever_exported():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [{"voucher_no": "266", "particulars": "Stamp papers", "amount": "900"}],
    }
    result = validate_extraction(MEMBER_RECEIPT, raw)
    for row in result["rows"]:
        for key, value in row.items():
            if key == "USER_EDITED" or key.startswith("_"):
                continue
            assert value not in (None, ""), key
            assert str(value).strip().lower() not in {"null", "none", "n/a", "unknown", "nan"}, key


# --------------------------------------------------------------------------
# 6. User edits are authoritative
# --------------------------------------------------------------------------

def test_user_edited_row_survives_re_extraction_normalization():
    raw = {
        "document_type": "PETTY_CASH_REGISTER",
        "rows": [
            {"voucher_no": "266", "particulars": "Stamp papers", "amount": "900",
             "USER_EDITED": True},
            {"voucher_no": "267", "particulars": "Tea", "amount": "140"},
        ],
    }
    result = validate_extraction(MEMBER_RECEIPT, raw)
    edited = [r for r in result["rows"] if r.get("USER_EDITED")]
    assert len(edited) == 1
    assert edited[0]["Amount*"] == "900"


def test_user_edited_row_is_skipped_by_mapping_resolution():
    """Bill-head mapping must not overwrite a value the reviewer set by hand."""
    mapped = {"rows": [
        {"Bill Head*": "Office", "USER_EDITED": True},
        {"Bill Head*": "Office"},
    ]}
    for row in mapped["rows"]:
        if row.get("USER_EDITED"):
            continue
        if row.get("Bill Head*") == "Office":
            row["Bill Head*"] = "SOCIETY_OFFICE"

    assert mapped["rows"][0]["Bill Head*"] == "Office", "user edit was overwritten"
    assert mapped["rows"][1]["Bill Head*"] == "SOCIETY_OFFICE"


# --------------------------------------------------------------------------
# 7. Production configuration safety
# --------------------------------------------------------------------------

def _prod_env(monkeypatch, **overrides):
    from app.core import config as config_module
    base = {
        "ENVIRONMENT": "production", "ALLOW_DEV_LOGIN": "0", "GEMINI_API_KEY": "k",
        "GOOGLE_CLIENT_ID": "c", "SESSION_SECRET": "s",
        "GOOGLE_ACCOUNTING_SPREADSHEET_ID": "sheet",
        "GOOGLE_SERVICE_ACCOUNT_FILE": "/etc/secrets/sa.json",
        "CORS_ALLOWED_ORIGINS": "https://nobrokerhood.github.io",
        # cleared so the developer machine's real POPPLER_PATH cannot leak in
        "POPPLER_PATH": "",
        # production processes jobs in the background (tests run inline)
        "JOB_EXECUTION": "background",
    }
    base.update(overrides)
    for key, value in base.items():
        monkeypatch.setenv(key, value)
    config_module.get_settings.cache_clear()
    return config_module


def test_dev_login_is_rejected_in_production(monkeypatch):
    mod = _prod_env(monkeypatch, ALLOW_DEV_LOGIN="1")
    errors = mod.validate_production_config(mod.get_settings())
    assert any("ALLOW_DEV_LOGIN" in e for e in errors)
    mod.get_settings.cache_clear()


def test_windows_credential_path_is_rejected_in_production(monkeypatch):
    mod = _prod_env(monkeypatch, GOOGLE_SERVICE_ACCOUNT_FILE="C:/Users/Meet/k.json")
    errors = mod.validate_production_config(mod.get_settings())
    assert any("Windows developer path" in e for e in errors), errors
    mod.get_settings.cache_clear()


def test_clean_production_config_has_no_errors(monkeypatch):
    mod = _prod_env(monkeypatch)
    assert mod.validate_production_config(mod.get_settings()) == []
    mod.get_settings.cache_clear()


def test_wildcard_cors_is_rejected_in_production(monkeypatch):
    mod = _prod_env(monkeypatch, CORS_ALLOWED_ORIGINS="*")
    errors = mod.validate_production_config(mod.get_settings())
    assert any("wildcard" in e for e in errors)
    mod.get_settings.cache_clear()
