from dataclasses import replace

from app.core.config import Settings
from app.services.gemini_client import GeminiDocumentClient
from tools import bootstrap_google_resources as bootstrap


def settings() -> Settings:
    return Settings(
        google_client_id="client-id",
        frontend_google_client_id="client-id",
        allowed_email_domain="nobroker.in",
        allow_domain_wide_access=False,
        session_inactivity_seconds=1200,
        session_heartbeat_grace_seconds=120,
        ai_verification_max_retries=2,
        allow_dev_login=False,
        gemini_api_key="gemini-secret",
        gemini_model="gemini-2.5-flash",
        max_file_size_mb=10,
        google_service_account_json=None,
        google_service_account_file=None,
        google_accounting_spreadsheet_id="shared-sheet",
        google_user_master_sheet_id=None,
        google_login_audit_sheet_id=None,
        google_api_usage_sheet_id=None,
        google_session_log_sheet_id=None,
        google_activity_log_sheet_id=None,
        google_processing_log_sheet_id=None,
        google_template_master_sheet_id=None,
        google_folder_config_sheet_id=None,
        google_mapping_master_sheet_id=None,
        google_drive_root_folder_id=None,
        google_login_audit_sheet_name="Accounting_AI_Login_Audit",
        google_api_usage_sheet_name="API_Usage_Report",
        cors_allowed_origins=("http://localhost:5000",),
    )


class FakeResponse:
    text = '{"ok": true}'


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeGenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.models = FakeModels()


def test_gemini_client_uses_google_genai_schema_json(monkeypatch):
    created = []

    def fake_client(api_key):
        client = FakeGenAIClient(api_key)
        created.append(client)
        return client

    monkeypatch.setattr("app.services.gemini_client.genai.Client", fake_client)
    client = GeminiDocumentClient(settings())

    result = client.generate_json(["prompt"])

    assert result == {"ok": True}
    assert created[0].api_key == "gemini-secret"
    call = created[0].models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == ["prompt"]
    assert call["config"].response_mime_type == "application/json"


def test_bootstrap_mock_check_reports_missing_without_secrets(monkeypatch, capsys):
    local_settings = replace(
        settings(),
        google_accounting_spreadsheet_id=None,
        gemini_api_key="do-not-print",
    )
    monkeypatch.setattr(bootstrap, "get_settings", lambda: local_settings)

    exit_code = bootstrap.main(["--check", "--mock"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "MISSING" in output
    assert "User_Master" in output
    assert "do-not-print" not in output
    assert "gemini-secret" not in output


def test_bootstrap_mock_check_passes_with_shared_spreadsheet(monkeypatch):
    monkeypatch.setattr(bootstrap, "get_settings", settings)

    results = bootstrap.check_resources(bootstrap=False, create_folders=False, mock=True)

    assert any(result.resource == "Job_State" and result.status == "READY" for result in results)
    assert all("shared-sheet" not in bootstrap.status_line(result) for result in results)
