import os
import pytest
from starlette.testclient import TestClient

from app.app_factory import create_app
from app.auth.user_master import AuthorizationError, UserMasterService
from app.core.config import Settings, get_config_diagnostic, validate_production_config
from tests.test_accounting_workflow import (
    FakeDriveService,
    FakeSheetsService,
    FakeVerifier,
    MEMBER_DATA,
    StaticExtractionProvider,
    StaticRepairProvider,
    StaticVerificationProvider,
    records,
)


def _base_test_settings(**kwargs) -> Settings:
    defaults = {
        "google_client_id": "test-client-id.apps.googleusercontent.com",
        "allowed_email_domain": "nobroker.in",
        "allow_domain_wide_access": False,
        "session_inactivity_seconds": 1200,
        "session_heartbeat_grace_seconds": 120,
        "ai_verification_max_retries": 2,
        "allow_dev_login": False,
        "gemini_api_key": "dummy-gemini-key",
        "gemini_model": "gemini-2.5-flash",
        "max_file_size_mb": 10,
        "google_service_account_json": None,
        "google_service_account_file": None,
        "google_accounting_spreadsheet_id": "dummy-sheet-id",
        "google_user_master_sheet_id": "users",
        "google_login_audit_sheet_id": "login",
        "google_api_usage_sheet_id": "usage",
        "google_session_log_sheet_id": "session",
        "google_activity_log_sheet_id": "activity",
        "google_processing_log_sheet_id": "processing",
        "google_template_master_sheet_id": "template",
        "google_folder_config_sheet_id": "folder",
        "google_mapping_master_sheet_id": "mapping",
        "google_drive_root_folder_id": None,
        "google_login_audit_sheet_name": "Login",
        "google_api_usage_sheet_name": "Usage",
        "cors_allowed_origins": ("https://nobrokerhood.github.io",),
        "environment": "development",
        "session_secret": "dummy-session-secret",
    }
    defaults.update(kwargs)
    return Settings(**defaults)


from app.auth.google_auth import AuthError, GoogleTokenVerifier

def _test_client(settings: Settings, verifier=None) -> TestClient:
    app = create_app(
        settings=settings,
        sheets_service=FakeSheetsService(records()),
        drive_service=FakeDriveService(),
        google_token_verifier=verifier or FakeVerifier(),
        extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA}),
        verification_provider=StaticVerificationProvider(),
        repair_provider=StaticRepairProvider(MEMBER_DATA),
    )
    return TestClient(app)


def test_public_config_contains_google_client_id():
    settings = _base_test_settings(google_client_id="123456789.apps.googleusercontent.com")
    client = _test_client(settings)
    res = client.get("/config/public")
    assert res.status_code == 200
    data = res.json()
    assert data["google_client_id"] == "123456789.apps.googleusercontent.com"
    assert data["application_name"] == "Accounting AI"
    assert data["allowed_email_domain"] == "nobroker.in"


def test_public_config_fails_safely_when_client_id_missing():
    settings = _base_test_settings(google_client_id=None)
    client = _test_client(settings)
    res = client.get("/config/public")
    assert res.status_code == 503
    data = res.json()
    assert "detail" in data
    assert "Google OAuth client is not configured on the server." in data["detail"]


def test_public_config_never_returns_google_client_secret():
    os.environ["GOOGLE_CLIENT_SECRET"] = "super-secret-client-secret-xyz"
    try:
        settings = _base_test_settings()
        client = _test_client(settings)
        res = client.get("/config/public")
        raw_text = res.text
        assert "super-secret-client-secret-xyz" not in raw_text
        assert "GOOGLE_CLIENT_SECRET" not in raw_text
    finally:
        os.environ.pop("GOOGLE_CLIENT_SECRET", None)


def test_public_config_never_returns_gemini_api_key():
    settings = _base_test_settings(gemini_api_key="confidential-gemini-api-key-999")
    client = _test_client(settings)
    res = client.get("/config/public")
    assert "confidential-gemini-api-key-999" not in res.text
    assert "gemini" not in res.text.lower()


def test_public_config_never_returns_session_secret():
    settings = _base_test_settings(session_secret="confidential-session-secret-abc")
    client = _test_client(settings)
    res = client.get("/config/public")
    assert "confidential-session-secret-abc" not in res.text
    assert "session_secret" not in res.text


def test_public_config_never_returns_service_account_private_key():
    settings = _base_test_settings(
        google_service_account_json='{"type": "service_account", "private_key": "-----BEGIN SENSITIVE KEY-----"}'
    )
    client = _test_client(settings)
    res = client.get("/config/public")
    assert "-----BEGIN SENSITIVE KEY-----" not in res.text
    assert "private_key" not in res.text


def test_production_config_requires_google_client_id():
    settings = _base_test_settings(
        environment="production",
        google_client_id=None,
        session_secret="test-secret",
        google_service_account_json='{"type": "service_account"}',
        gemini_api_key="test-key",
        google_accounting_spreadsheet_id="test-sheet",
        allow_dev_login=False,
    )
    errors = validate_production_config(settings)
    assert any("GOOGLE_CLIENT_ID is required in production." in e for e in errors)


def test_production_config_requires_session_secret():
    settings = _base_test_settings(
        environment="production",
        google_client_id="test.apps.googleusercontent.com",
        session_secret=None,
        google_service_account_json='{"type": "service_account"}',
        gemini_api_key="test-key",
        google_accounting_spreadsheet_id="test-sheet",
        allow_dev_login=False,
    )
    errors = validate_production_config(settings)
    assert any("SESSION_SECRET is required in production." in e for e in errors)


def test_production_authentication_rejects_invalid_google_identity():
    settings = _base_test_settings()
    client = _test_client(settings, verifier=GoogleTokenVerifier(settings))
    res = client.post("/auth/google-login", json={"credential": "invalid-token-string"})
    assert res.status_code == 401
    assert "Invalid Google ID token." in res.json()["detail"]


def test_domain_restriction_works_server_side():
    settings = _base_test_settings(
        allowed_email_domain="nobroker.in",
        allow_domain_wide_access=True,
    )
    sheets_service = FakeSheetsService(records())
    user_master = UserMasterService(settings, sheets_service)

    # Valid domain succeeds
    valid_user = user_master.authorize("john.doe@nobroker.in", "John Doe")
    assert valid_user.email == "john.doe@nobroker.in"
    assert valid_user.role == "USER"

    # Invalid domain fails
    with pytest.raises(AuthorizationError) as exc_info:
        user_master.authorize("attacker@external-domain.com", "Attacker")
    assert "User is not authorized." in str(exc_info.value)


def test_production_config_diagnostic_safe_booleans_only():
    settings = _base_test_settings(
        google_client_id="12345.apps.googleusercontent.com",
        session_secret="secret",
        gemini_api_key="gemini-key",
    )
    diag = get_config_diagnostic(settings)
    # Check all keys are booleans
    for key, value in diag.items():
        assert isinstance(value, bool), f"Key {key} must be a boolean, got {type(value)}"
    assert diag["google_client_id_configured"] is True
    assert diag["google_client_id_format_valid"] is True
    assert diag["gemini_configured"] is True
    assert diag["session_secret_configured"] is True


def test_auth_endpoints_return_401_for_missing_or_invalid_session():
    settings = _base_test_settings()
    client = _test_client(settings)

    # Missing token
    res = client.get("/auth/me")
    assert res.status_code == 401

    res = client.post("/auth/heartbeat", json={"user_active": True, "page_visible": True})
    assert res.status_code == 401

    # Invalid token
    res = client.get("/auth/me", headers={"Authorization": "Bearer bad_token_xyz"})
    assert res.status_code == 401


def test_auth_endpoints_return_403_for_dev_login_when_disabled():
    settings = _base_test_settings(
        allow_dev_login=False,
    )
    client = _test_client(settings)
    res = client.post("/auth/dev-login", json={"email": "dev@nobroker.in"})
    assert res.status_code == 403
    assert res.json()["detail"] == "DEV_LOGIN_DISABLED"



def test_production_gemini_and_ocr_config_invariants():
    settings = _base_test_settings(
        gemini_model="gemini-2.5-flash",
    )
    assert settings.gemini_model == "gemini-2.5-flash"

    client = _test_client(settings)
    res = client.get("/config/capabilities")
    data = res.json()
    # RapidOCR availability is proven by a real inference probe on the
    # pipeline's own engine instance (it used to be a hardcoded True).
    assert data["capabilities"]["rapidocr"]["status"] == "READY"
    assert data["capabilities"]["rapidocr"]["detail"] == "inference probe passed"

