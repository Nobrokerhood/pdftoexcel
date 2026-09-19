"""Amount/date normalisation and the Gemini response contract."""

from datetime import date
from decimal import Decimal

import pytest

from app.accounting.dates import parse_date
from app.accounting.money import AMBIGUOUS, FOUND, INVALID, MISSING, parse_amount
from app.agents.gemini_contract import ExtractionContractError, normalize_response


@pytest.mark.parametrize("raw, value", [
    ("Rs.900", "900"), ("Rs. 1,789/-", "1789"), ("750/-", "750"), ("₹ 1,00,000.50", "100000.50"),
    ("INR 12,450", "12450"), ("12,45,000", "1245000"), ("1,08,800", "108800"), ("(1,200.00)", "-1200.00"),
    ("10.50", "10.50"), (900, "900"), ("200 Cr", "200"), ("5,000 only", "5000"),
])
def test_amounts_found(raw, value):
    reading = parse_amount(raw)
    assert reading.status == FOUND, (raw, reading.reason)
    assert reading.value == Decimal(value)


@pytest.mark.parametrize("raw, status", [
    ("2500 (105)", AMBIGUOUS), ("1.500,00", AMBIGUOUS), ("18001-", AMBIGUOUS), ("1.500", AMBIGUOUS),
    ("500 600", AMBIGUOUS), ("25O0", INVALID), ("1,2345", INVALID), ("98,6231,08,300", INVALID),
    ("-", MISSING), ("", MISSING), (None, MISSING), ("N/A", MISSING), ("unknown", MISSING), (True, INVALID),
])
def test_amounts_never_guessed(raw, status):
    reading = parse_amount(raw)
    assert reading.status == status, (raw, reading.status, reading.reason)
    assert reading.value is None


def test_ambiguous_amounts_carry_candidates_and_reason():
    reading = parse_amount("18001-")
    assert set(reading.candidates) == {"1800", "18001"}
    assert "/-" in reading.reason


@pytest.mark.parametrize("raw", ["04-07-25", "04/07/2025", "2025-07-04", "4 Jul 2025", "04.07.2025", "Jul 4, 2025"])
def test_dates_are_day_first(raw):
    assert parse_date(raw).value == date(2025, 7, 4)


@pytest.mark.parametrize("raw, status", [("31-02-2025", "INVALID"), ("86+0-25", "INVALID"), ("-", "MISSING"),
                                         ("next Tuesday", "INVALID"), ("04-07-1850", "INVALID")])
def test_invalid_dates_are_not_coerced(raw, status):
    reading = parse_date(raw)
    assert reading.status == status and reading.value is None


# ---- Gemini purpose contracts ------------------------------------------------------

def test_petty_cash_contract_accepts_rows_object():
    model, notes = normalize_response("PETTY_CASH_REGISTER", {"rows": [
        {"row_kind": "EXPENSE", "voucher_no": 266, "date": "04-07-25", "amount": "900", "source_line_ids": ["L3"]}]})
    assert model.rows[0].voucher_no == "266" and model.rows[0].source_line_ids == ["L3"] and notes == []


def test_list_response_is_normalized_explicitly_for_multi_row_purposes():
    model, notes = normalize_response("PETTY_CASH_REGISTER", [{"voucher_no": "1", "amount": "5"}])
    assert len(model.rows) == 1 and notes and "SHAPE_NORMALIZED" in notes[0]


def test_flat_object_is_a_contract_violation_for_a_register():
    """Live failure: a flat 12-column dict silently became zero rows."""
    flat = {"Payment Type*": "Cash", "Amount*": "900", "Cheque/Ref No*": "266"}
    with pytest.raises(ExtractionContractError) as err:
        normalize_response("PETTY_CASH_REGISTER", flat)
    assert err.value.shape == "object_without_rows"


def test_flat_object_is_a_single_receipt_for_member_receipt():
    model, notes = normalize_response("MEMBER_RECEIPT", {"Amount*": "5000", "Cheque/Ref No*": "UPI1"})
    assert len(model.rows) == 1 and "single" in notes[0]


@pytest.mark.parametrize("purpose, raw, shape", [
    ("PETTY_CASH_REGISTER", {"rows": "not a list"}, "rows_not_list"),
    ("PETTY_CASH_REGISTER", {"rows": ["x"]}, "row_not_object"),
    ("PETTY_CASH_REGISTER", {"rows": [{"amount": {"nested": 1}}]}, "schema_mismatch"),
    ("PETTY_CASH_REGISTER", {"rows": [{"row_kind": "TRANSFER"}]}, "schema_mismatch"),
    ("PETTY_CASH_REGISTER", "not json", "text"),
    ("PETTY_CASH_REGISTER", [1, 2], "list"),
    ("VENDOR_INVOICE", [{"amount": "1"}], "list"),
    ("VENDOR_INVOICE", {"bill_number": "1"}, "object_without_line_items"),
    ("MEMBER_RECEIPT", {"summary": "nothing"}, "object_without_rows"),
    ("MEMBER_RECEIPT", {"rows": [{"Amount*": ["a"]}]}, "bad_field"),
])
def test_invalid_shapes_raise_structured_errors(purpose, raw, shape):
    with pytest.raises(ExtractionContractError) as err:
        normalize_response(purpose, raw)
    assert err.value.shape == shape


def test_petty_cash_prompt_is_not_the_vendor_schema():
    from app.agents.gemini_contract import prompt_for
    prompt = prompt_for("PETTY_CASH_REGISTER")
    assert "voucher_no" in prompt and "INFLOW" in prompt
    assert "cgst_amount" not in prompt and "bill_number" not in prompt
