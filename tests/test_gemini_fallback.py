import json
import logging
from dataclasses import replace

import pytest
from google.genai import errors as genai_errors

from app.agents.extractor import GeminiExtractionProvider
from app.agents.repair import GeminiRepairProvider
from app.agents.verifier import GeminiVerificationProvider
from app.core.config import get_settings
from app.core.errors import ExternalServiceUnavailableError
from app.services import gemini_client as gemini_module
from app.services.gemini_client import (
    GEMINI_SPEND_CAP_MESSAGE,
    PRIMARY_CAP_COOLDOWN_SECONDS,
    GeminiDocumentClient,
    GeminiFallbackUnavailableError,
    GeminiSpendCapError,
    GeminiTemporarilyUnavailableError,
)
from tests.test_accounting_workflow import MEMBER_DATA, settings
from tests.test_gemini_errors import (
    ScriptedGenAIClient,
    ScriptedModels,
    activity_actions,
    invalid_argument_error,
    rate_limit_error,
    sleeps,  # noqa: F401  (pytest fixture)
    spend_cap_error,
    upload_png,
    workflow_client,
)


PRIMARY_KEY = "primary-key-value-must-not-leak"
FALLBACK_KEY = "fallback-key-value-must-not-leak"
OK_JSON = '{"ok": true}'


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def fallback_settings(enabled=True, fallback_key=FALLBACK_KEY):
    return replace(
        settings(),
        gemini_api_key=PRIMARY_KEY,
        gemini_api_key_fallback=fallback_key,
        gemini_fallback_enabled=enabled,
    )


def keyed_clients(monkeypatch, primary_outcomes, fallback_outcomes=(OK_JSON,)):
    """Give each API key its own scripted models and record every client created."""
    models = {
        PRIMARY_KEY: ScriptedModels(primary_outcomes),
        FALLBACK_KEY: ScriptedModels(fallback_outcomes),
    }
    created = []

    def factory(api_key, **_kwargs):
        created.append(api_key)
        return ScriptedGenAIClient(models[api_key])

    monkeypatch.setattr(gemini_module.genai, "Client", factory)
    return models[PRIMARY_KEY], models[FALLBACK_KEY], created


def assert_no_key_values(*texts):
    for text in texts:
        assert PRIMARY_KEY not in text
        assert FALLBACK_KEY not in text


def test_primary_success_never_touches_fallback(monkeypatch, sleeps):
    primary, fallback, created = keyed_clients(monkeypatch, [OK_JSON])
    client = GeminiDocumentClient(fallback_settings())

    assert client.generate_json(["prompt"]) == {"ok": True}
    assert primary.calls == 1
    assert fallback.calls == 0
    assert created == [PRIMARY_KEY]


def test_primary_spend_cap_uses_separate_fallback_client(monkeypatch, sleeps, caplog):
    primary, fallback, created = keyed_clients(monkeypatch, [spend_cap_error()], [OK_JSON])
    client = GeminiDocumentClient(fallback_settings())

    with caplog.at_level(logging.DEBUG, logger="app.services.gemini_client"):
        assert client.generate_json(["prompt"]) == {"ok": True}

    assert primary.calls == 1
    assert fallback.calls == 1
    assert created == [PRIMARY_KEY, FALLBACK_KEY]
    assert client._client is not client._fallback_client
    assert sleeps == []
    assert "fallback key used" in caplog.text
    assert_no_key_values(caplog.text)


def test_generate_content_also_falls_back(monkeypatch, sleeps):
    primary, fallback, _ = keyed_clients(monkeypatch, [spend_cap_error()], ["plain text"])
    client = GeminiDocumentClient(fallback_settings())

    assert client.generate_content(["prompt"]).text == "plain text"
    assert (primary.calls, fallback.calls) == (1, 1)


def test_both_keys_capped_raises_safe_error_after_one_attempt_each(monkeypatch, sleeps, caplog):
    primary, fallback, _ = keyed_clients(monkeypatch, [spend_cap_error()], [spend_cap_error()])
    client = GeminiDocumentClient(fallback_settings())

    with caplog.at_level(logging.DEBUG, logger="app.services.gemini_client"):
        with pytest.raises(GeminiFallbackUnavailableError) as raised:
            client.generate_json(["prompt"], retries=5)

    assert (primary.calls, fallback.calls) == (1, 1)
    assert sleeps == []
    assert isinstance(raised.value, GeminiSpendCapError)
    assert isinstance(raised.value, ExternalServiceUnavailableError)
    assert str(raised.value).startswith(GEMINI_SPEND_CAP_MESSAGE)
    assert "fallback Gemini API key also failed" in str(raised.value)
    assert_no_key_values(str(raised.value), caplog.text)


def test_fallback_rate_limited_is_bounded_and_never_returns_to_primary(monkeypatch, sleeps):
    primary, fallback, _ = keyed_clients(monkeypatch, [spend_cap_error()], [rate_limit_error()])
    client = GeminiDocumentClient(fallback_settings())

    with pytest.raises(GeminiFallbackUnavailableError) as raised:
        client.generate_json(["prompt"], retries=3)

    assert primary.calls == 1
    assert fallback.calls == 3
    assert "429 RESOURCE_EXHAUSTED" in str(raised.value)


def test_fallback_invalid_key_is_reported_without_retry(monkeypatch, sleeps):
    primary, fallback, _ = keyed_clients(monkeypatch, [spend_cap_error()], [invalid_argument_error()])
    client = GeminiDocumentClient(fallback_settings())

    with pytest.raises(GeminiFallbackUnavailableError) as raised:
        client.generate_json(["prompt"])

    assert (primary.calls, fallback.calls) == (1, 1)
    assert "400 INVALID_ARGUMENT" in str(raised.value)


def test_primary_transient_429_retries_primary_without_fallback(monkeypatch, sleeps):
    primary, fallback, created = keyed_clients(monkeypatch, [rate_limit_error(), OK_JSON])
    client = GeminiDocumentClient(fallback_settings())

    assert client.generate_json(["prompt"]) == {"ok": True}
    assert (primary.calls, fallback.calls) == (2, 0)
    assert len(sleeps) == 1
    assert created == [PRIMARY_KEY]


def test_primary_persistent_429_raises_without_fallback(monkeypatch, sleeps):
    primary, fallback, _ = keyed_clients(monkeypatch, [rate_limit_error()])
    client = GeminiDocumentClient(fallback_settings())

    with pytest.raises(GeminiTemporarilyUnavailableError) as raised:
        client.generate_json(["prompt"])
    assert (primary.calls, fallback.calls) == (2, 0)
    assert isinstance(raised.value.__cause__, genai_errors.ClientError)


def test_invalid_primary_key_is_surfaced_not_hidden_by_fallback(monkeypatch, sleeps):
    primary, fallback, created = keyed_clients(monkeypatch, [invalid_argument_error()])
    client = GeminiDocumentClient(fallback_settings())

    with pytest.raises(genai_errors.ClientError) as raised:
        client.generate_json(["prompt"])

    assert (primary.calls, fallback.calls) == (1, 0)
    assert created == [PRIMARY_KEY]
    assert_no_key_values(str(raised.value))


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"fallback_key": None},
        {"fallback_key": "   "},
        {"fallback_key": PRIMARY_KEY},
    ],
    ids=["flag-off", "no-key", "blank-key", "same-as-primary"],
)
def test_fallback_not_used_unless_enabled_with_distinct_key(monkeypatch, sleeps, overrides):
    primary, fallback, created = keyed_clients(monkeypatch, [spend_cap_error()])
    client = GeminiDocumentClient(fallback_settings(**overrides))

    with pytest.raises(GeminiSpendCapError) as raised:
        client.generate_json(["prompt"])

    assert not isinstance(raised.value, GeminiFallbackUnavailableError)
    assert str(raised.value) == GEMINI_SPEND_CAP_MESSAGE
    assert (primary.calls, fallback.calls) == (1, 0)
    assert created == [PRIMARY_KEY]


def test_capped_primary_is_skipped_during_cooldown_then_rechecked(monkeypatch, sleeps):
    primary, fallback, _ = keyed_clients(
        monkeypatch, [spend_cap_error(), spend_cap_error(), OK_JSON], [OK_JSON]
    )
    clock = FakeClock()
    client = GeminiDocumentClient(fallback_settings(), clock=clock)

    client.generate_json(["first"])
    client.generate_json(["second"])
    assert (primary.calls, fallback.calls) == (1, 2)

    clock.now += PRIMARY_CAP_COOLDOWN_SECONDS + 1
    client.generate_json(["after cooldown, still capped"])
    assert (primary.calls, fallback.calls) == (2, 3)

    clock.now += PRIMARY_CAP_COOLDOWN_SECONDS + 1
    client.generate_json(["after cooldown, billing fixed"])
    assert (primary.calls, fallback.calls) == (3, 3)


def test_settings_read_fallback_variables_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY_FALLBACK", FALLBACK_KEY)
    monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "true")
    get_settings.cache_clear()
    try:
        loaded = get_settings()
        assert loaded.gemini_api_key_fallback == FALLBACK_KEY
        assert loaded.gemini_fallback_enabled is True

        monkeypatch.setenv("GEMINI_FALLBACK_ENABLED", "false")
        get_settings.cache_clear()
        assert get_settings().gemini_fallback_enabled is False
    finally:
        get_settings.cache_clear()


def gemini_workflow(monkeypatch, primary_outcomes, fallback_outcomes):
    primary, fallback, _ = keyed_clients(monkeypatch, primary_outcomes, fallback_outcomes)
    gemini = GeminiDocumentClient(fallback_settings())
    client, sheets = workflow_client(
        extraction_provider=GeminiExtractionProvider(gemini),
        verification_provider=GeminiVerificationProvider(gemini),
        repair_provider=GeminiRepairProvider(gemini),
    )
    return client, sheets, primary, fallback


def recorded_google_writes(sheets):
    return json.dumps([sheets.appended, sheets.updated], default=str)


def test_workflow_continues_on_fallback_when_primary_capped(monkeypatch, sleeps, caplog):
    client, sheets, primary, fallback = gemini_workflow(
        monkeypatch, [spend_cap_error()], [json.dumps(MEMBER_DATA)]
    )

    with caplog.at_level(logging.DEBUG):
        response = upload_png(client)

    assert response.status_code == 200
    job = response.json()
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["extraction_status"] == "COMPLETED"
    assert job["extraction_provider"] == "GEMINI"
    assert job["extracted_data"]["rows"][0]["Cheque/Ref No*"] == MEMBER_DATA["reference_number"]
    # The source image has no OCR text, so the model's row cannot be corroborated:
    # it goes to review instead of being verified by the same model.
    assert job["verification_status"] == "NEEDS_REVIEW"
    assert job["last_error"] == ""
    # Primary tried once; the fallback served the extraction.
    assert (primary.calls, fallback.calls) == (1, 1)
    assert "fallback" not in response.text.lower()
    assert_no_key_values(response.text, recorded_google_writes(sheets), caplog.text)


def test_workflow_fails_safely_when_both_keys_capped(monkeypatch, sleeps, caplog):
    client, sheets, primary, fallback = gemini_workflow(
        monkeypatch, [spend_cap_error()], [spend_cap_error()]
    )

    with caplog.at_level(logging.DEBUG):
        response = upload_png(client)

    assert response.status_code == 200
    job = response.json()
    # Degrades to OCR evidence and human review; never fabricated, never PASSED.
    assert job["overall_status"] == "NEEDS_REVIEW"
    assert job["current_step"] == "HUMAN_REVIEW"
    assert job["extracted_data"]["rows"] == []
    assert job["extraction_provider"] == "LOCAL_OCR"
    assert GEMINI_SPEND_CAP_MESSAGE in job["extracted_data"]["provider_note"]
    assert job["verification_status"] != "PASSED"
    assert (primary.calls, fallback.calls) == (1, 1)
    assert_no_key_values(response.text, job["extracted_data"]["provider_note"], recorded_google_writes(sheets),
                         caplog.text)
