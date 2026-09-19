import io
import logging
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from PIL import Image

from app.agents.extractor import GeminiExtractionProvider
from app.agents.repair import GeminiRepairProvider
from app.agents.verifier import GeminiVerificationProvider
from app.app_factory import create_app
from app.services import gemini_client as gemini_module
from app.services.gemini_client import (
    GEMINI_AUTH_MESSAGE,
    GEMINI_QUOTA_EXHAUSTED_MESSAGE,
    GEMINI_SPEND_CAP_MESSAGE,
    GeminiAuthenticationError,
    GeminiDocumentClient,
    GeminiQuotaExhaustedError,
    GeminiSpendCapError,
    GeminiTemporarilyUnavailableError,
    is_authentication_error,
    is_daily_quota_error,
    is_retryable_error,
    is_spend_cap_error,
)
from tests.test_accounting_workflow import (
    MEMBER_DATA,
    FakeDriveService,
    FakeSheetsService,
    FakeVerifier,
    StaticExtractionProvider,
    StaticRepairProvider,
    StaticVerificationProvider,
    headers,
    records,
    settings,
    token,
)


API_KEY = "fake-gemini-key-must-not-leak"

# Body shapes as returned by the Gemini API and parsed by google-genai's APIError.
SPEND_CAP_BODY = {
    "error": {
        "code": 429,
        "message": (
            "Your project has exceeded its monthly spending cap. Please go to AI Studio "
            "at https://ai.studio/spend to manage your project spend cap."
        ),
        "status": "RESOURCE_EXHAUSTED",
    }
}
RATE_LIMIT_BODY = {
    "error": {
        "code": 429,
        "message": "Resource has been exhausted (e.g. check quota).",
        "status": "RESOURCE_EXHAUSTED",
    }
}


def spend_cap_error():
    return genai_errors.ClientError(429, SPEND_CAP_BODY)


def rate_limit_error():
    return genai_errors.ClientError(429, RATE_LIMIT_BODY)


def server_error():
    return genai_errors.ServerError(
        503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}
    )


def daily_quota_error():
    return genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota, please check your plan and billing details.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {
                                "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                                "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                "quotaValue": "20",
                            }
                        ],
                    },
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "59s"},
                ],
            }
        },
    )


def invalid_api_key_error():
    return genai_errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT",
                "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}],
            }
        },
    )


def invalid_argument_error():
    return genai_errors.ClientError(
        400, {"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}}
    )


class FakeResponse:
    def __init__(self, text):
        self.text = text


class ScriptedModels:
    """Raises or returns the scripted outcomes in order, counting every request."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        index = min(self.calls, len(self.outcomes) - 1)
        self.calls += 1
        outcome = self.outcomes[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome)


class ScriptedGenAIClient:
    def __init__(self, models):
        self.models = models


def gemini_settings():
    return replace(settings(), gemini_api_key=API_KEY)


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(gemini_module.time, "sleep", lambda seconds: calls.append(seconds))
    return calls


def scripted_client(monkeypatch, outcomes):
    models = ScriptedModels(outcomes)
    monkeypatch.setattr(
        gemini_module.genai, "Client", lambda api_key, **_kwargs: ScriptedGenAIClient(models)
    )
    return GeminiDocumentClient(gemini_settings()), models


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(buf, format="PNG")
    return buf.getvalue()


def test_installed_sdk_exposes_code_status_and_message_on_api_error():
    error = spend_cap_error()

    assert isinstance(error, genai_errors.APIError)
    assert error.code == 429
    assert error.status == "RESOURCE_EXHAUSTED"
    assert "monthly spending cap" in error.message


def test_classification_separates_spend_cap_from_transient_errors():
    assert is_spend_cap_error(spend_cap_error())
    assert not is_retryable_error(spend_cap_error())

    assert not is_spend_cap_error(rate_limit_error())
    assert is_retryable_error(rate_limit_error())
    assert is_retryable_error(server_error())
    assert is_retryable_error(ConnectionError("network down"))

    assert not is_retryable_error(invalid_argument_error())
    assert not is_spend_cap_error(ValueError("spending cap"))


def test_transient_rate_limit_is_retried_then_succeeds(monkeypatch, sleeps):
    client, models = scripted_client(monkeypatch, [rate_limit_error(), '{"ok": true}'])

    assert client.generate_json(["prompt"]) == {"ok": True}
    assert models.calls == 2
    assert len(sleeps) == 1


def test_transient_server_error_retries_up_to_limit(monkeypatch, sleeps):
    client, models = scripted_client(monkeypatch, [server_error()])

    with pytest.raises(GeminiTemporarilyUnavailableError) as raised:
        client.generate_content(["prompt"], retries=3)
    assert models.calls == 3
    assert len(sleeps) == 2
    assert isinstance(raised.value.__cause__, genai_errors.ServerError)
    assert "503 UNAVAILABLE" in str(raised.value)


def test_workflow_verification_overload_needs_review_instead_of_500(monkeypatch, sleeps):
    """A persistently overloaded verifier never yields PASSED; rows stay unverified for review."""
    gemini, models = scripted_client(monkeypatch, [server_error()])
    client, sheets = workflow_client(
        extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA}),
        verification_provider=GeminiVerificationProvider(gemini),
        repair_provider=StaticRepairProvider(MEMBER_DATA),
    )

    response = upload_png(client)

    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["verification_status"] == "NEEDS_REVIEW"
    assert any("temporarily unavailable" in note for note in job["verification_result"]["notes"])
    assert job["extracted_data"]["rows"][0]["_status"] == "UNVERIFIED"
    assert models.calls == 2  # bounded retries, then stop
    assert "AI_VERIFICATION_FAILED" in activity_actions(sheets)


def test_daily_quota_is_not_retried_and_has_its_own_message(monkeypatch, sleeps, caplog):
    assert is_daily_quota_error(daily_quota_error())
    assert not is_daily_quota_error(rate_limit_error())
    assert not is_daily_quota_error(spend_cap_error())
    client, models = scripted_client(monkeypatch, [daily_quota_error(), '{"ok": true}'])

    with caplog.at_level(logging.DEBUG, logger="app.services.gemini_client"):
        with pytest.raises(GeminiQuotaExhaustedError) as raised:
            client.generate_json(["prompt"], retries=5)

    assert models.calls == 1
    assert sleeps == []
    assert str(raised.value) == GEMINI_QUOTA_EXHAUSTED_MESSAGE
    assert API_KEY not in caplog.text


def test_invalid_api_key_is_not_retried_and_has_safe_message(monkeypatch, sleeps):
    assert is_authentication_error(invalid_api_key_error())
    assert not is_authentication_error(invalid_argument_error())
    client, models = scripted_client(monkeypatch, [invalid_api_key_error(), '{"ok": true}'])

    with pytest.raises(GeminiAuthenticationError) as raised:
        client.generate_json(["prompt"], retries=5)

    assert models.calls == 1
    assert sleeps == []
    assert str(raised.value) == GEMINI_AUTH_MESSAGE


def test_persistent_rate_limit_says_rate_limited(monkeypatch, sleeps):
    client, models = scripted_client(monkeypatch, [rate_limit_error()])

    with pytest.raises(GeminiTemporarilyUnavailableError) as raised:
        client.generate_json(["prompt"])

    assert models.calls == 2
    assert "temporarily rate limited" in str(raised.value)


def test_workflow_extraction_daily_quota_degrades_to_review_with_one_call(monkeypatch, sleeps):
    """Quota exhaustion: one call, no retry storm, no fabricated rows, NEEDS_REVIEW with the reason."""
    gemini, models = scripted_client(monkeypatch, [daily_quota_error()])
    client, sheets = workflow_client(
        extraction_provider=GeminiExtractionProvider(gemini),
        verification_provider=GeminiVerificationProvider(gemini),
        repair_provider=GeminiRepairProvider(gemini),
    )

    job = upload_png(client).json()

    assert models.calls == 1
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["extraction_provider"] == "LOCAL_OCR"
    assert GEMINI_QUOTA_EXHAUSTED_MESSAGE in job["extracted_data"]["provider_note"]
    assert job["extracted_data"]["rows"] == []                       # nothing invented
    assert job["validation_status"] == "BLOCKED"                     # zero rows cannot be approved
    stages = {p["stage"]: p["status"] for p in job["progress"]}
    assert stages["UPLOAD"] == "DONE"
    assert stages["OCR_AND_EXTRACTION"] == "DONE"
    assert stages["HUMAN_REVIEW"] == "NEEDS_ATTENTION"
    assert "EXPORT" not in stages                                    # never claimed


def test_non_retryable_client_error_is_not_retried(monkeypatch, sleeps):
    client, models = scripted_client(monkeypatch, [invalid_argument_error(), '{"ok": true}'])

    with pytest.raises(genai_errors.ClientError):
        client.generate_json(["prompt"])
    assert models.calls == 1
    assert sleeps == []


@pytest.mark.parametrize("method", ["generate_json", "generate_content"])
def test_spend_cap_is_not_retried_and_raises_actionable_error(monkeypatch, sleeps, caplog, method):
    client, models = scripted_client(monkeypatch, [spend_cap_error(), '{"ok": true}'])

    with caplog.at_level(logging.DEBUG, logger="app.services.gemini_client"):
        with pytest.raises(GeminiSpendCapError) as raised:
            getattr(client, method)(["prompt"], retries=5)

    assert models.calls == 1
    assert sleeps == []
    assert str(raised.value) == GEMINI_SPEND_CAP_MESSAGE
    assert API_KEY not in str(raised.value)
    assert API_KEY not in caplog.text
    assert "spending cap" in caplog.text


def workflow_client(extraction_provider=None, verification_provider=None, repair_provider=None):
    sheets = FakeSheetsService(records())
    app = create_app(
        settings=gemini_settings(),
        sheets_service=sheets,
        drive_service=FakeDriveService(),
        google_token_verifier=FakeVerifier(),
        extraction_provider=extraction_provider,
        verification_provider=verification_provider,
        repair_provider=repair_provider,
    )
    return TestClient(app), sheets


def upload_png(client):
    session_token = token(client)
    return client.post(
        "/processing/jobs",
        headers=headers(session_token),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": ("receipt.png", png_bytes(), "image/png")},
    )


def activity_actions(sheets):
    return [values[4] for sheet_id, values in sheets.appended if sheet_id == "activity"]


def assert_degraded_safely(response, sheets):
    """Gemini unavailable -> NEEDS_REVIEW with the reason; never PASSED, never fabricated."""
    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["human_status"] == "NEEDS_REVIEW"
    assert job["current_step"] == "HUMAN_REVIEW"
    assert job["verification_status"] != "PASSED"
    assert job["output_filename"] == ""
    assert API_KEY not in response.text
    return job


def test_workflow_extraction_spend_cap_degrades_safely_with_one_gemini_call(monkeypatch, sleeps):
    gemini, models = scripted_client(monkeypatch, [spend_cap_error()])
    client, sheets = workflow_client(
        extraction_provider=GeminiExtractionProvider(gemini),
        verification_provider=GeminiVerificationProvider(gemini),
        repair_provider=GeminiRepairProvider(gemini),
    )

    job = assert_degraded_safely(upload_png(client), sheets)

    assert models.calls == 1
    assert sleeps == []
    assert job["extraction_provider"] == "LOCAL_OCR"
    assert GEMINI_SPEND_CAP_MESSAGE in job["extracted_data"]["provider_note"]
    assert job["extracted_data"]["rows"] == []
    assert "EXTRACTION_COMPLETED" in activity_actions(sheets)


def test_workflow_verification_spend_cap_needs_review_without_repair(monkeypatch, sleeps):
    gemini, models = scripted_client(monkeypatch, [spend_cap_error()])
    repair = StaticRepairProvider(MEMBER_DATA)
    client, sheets = workflow_client(
        extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA}),
        verification_provider=GeminiVerificationProvider(gemini),
        repair_provider=repair,
    )

    job = assert_degraded_safely(upload_png(client), sheets)

    assert models.calls == 1
    assert repair.calls == 0
    assert job["verification_status"] == "NEEDS_REVIEW"
    assert any(GEMINI_SPEND_CAP_MESSAGE in n for n in job["verification_result"]["notes"])
    assert "AI_VERIFICATION_FAILED" in activity_actions(sheets)


def test_workflow_repair_spend_cap_is_bounded_and_goes_to_review(monkeypatch, sleeps):
    gemini, models = scripted_client(monkeypatch, [spend_cap_error()])
    verifier = StaticVerificationProvider([{"overall_status": "FAILED", "fields": []}])
    missing_amount = {**MEMBER_DATA, "amount": None, "reference_number": "UPI123456"}
    client, sheets = workflow_client(
        extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": missing_amount}),
        verification_provider=verifier,
        repair_provider=GeminiRepairProvider(gemini),
    )

    job = assert_degraded_safely(upload_png(client), sheets)

    assert models.calls == 1          # one repair attempt, spend cap not retried
    assert verifier.calls == 2        # verify, repair, re-verify once; no loop
    repair_stage = [p for p in job["progress"] if p["stage"] == "REPAIR"]
    assert repair_stage and repair_stage[0]["status"] == "SKIPPED"


def test_legacy_converter_returns_503_on_spend_cap_without_retrying_pages(monkeypatch, sleeps):
    gemini, models = scripted_client(monkeypatch, [spend_cap_error()])
    client, _ = workflow_client()
    client.app.state.gemini_client = gemini

    response = client.post(
        "/export-to-excel/",
        headers=headers(token(client)),  # legacy tools now require a session
        files={"file": ("table.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == GEMINI_SPEND_CAP_MESSAGE
    assert models.calls == 1
    assert API_KEY not in response.text
