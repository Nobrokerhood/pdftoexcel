import json
from dataclasses import replace
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.accounting.folders import FolderConfigurationError, FolderRouterService
from app.accounting.templates import TemplateConfigurationError
from app.app_factory import create_app
from app.auth.google_auth import AuthError, VerifiedGoogleUser
from app.auth.sessions import SessionService, utc_now
from app.auth.user_master import AuthorizedUser, AuthorizationError, UserMasterService
from app.core.config import Settings
from app.google.drive_service import GoogleDriveError, GoogleDriveService
from app.google.sheets_service import GoogleSheetsService


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
        gemini_api_key=None,
        gemini_model="gemini-2.5-flash",
        max_file_size_mb=10,
        google_service_account_json=None,
        google_service_account_file=None,
        google_accounting_spreadsheet_id=None,
        google_user_master_sheet_id="users",
        google_login_audit_sheet_id="login",
        google_api_usage_sheet_id="usage",
        google_session_log_sheet_id="sessions",
        google_activity_log_sheet_id="activity",
        google_processing_log_sheet_id="processing",
        google_template_master_sheet_id="templates",
        google_folder_config_sheet_id="folders",
        google_mapping_master_sheet_id="mapping",
        google_drive_root_folder_id="root",
        google_login_audit_sheet_name="Accounting_AI_Login_Audit",
        google_api_usage_sheet_name="API_Usage_Report",
        cors_allowed_origins=("http://localhost:5000",),
    )


class FakeSheetsService:
    def __init__(self, records=None):
        self.records = records or {}
        self.appended = []
        self.updated = []

    def read_records(self, spreadsheet_id, worksheet_name="Sheet1", use_cache=True):
        return [row.copy() for row in self.records.get(spreadsheet_id, [])]

    def read_table(self, table_key, use_cache=True):
        table_map = {
            "user_master": "users",
            "folders": "folders",
            "folder_config": "folders",
            "template_master": "templates",
            "mapping_master": "mapping",
            "login_audit": "login",
            "activity_log": "activity",
            "session_log": "sessions",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.read_records(table_map.get(table_key, table_key))

    def append_row(self, spreadsheet_id, values, worksheet_name="Sheet1"):
        self.appended.append((spreadsheet_id, values))
        return True

    def append_table_row(self, table_key, values):
        table_map = {
            "login_audit": "login",
            "activity_log": "activity",
            "session_log": "sessions",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.append_row(table_map.get(table_key, table_key), values)

    def lookup_row_by_key(self, spreadsheet_id, key_column, key_value, worksheet_name="Sheet1"):
        for record in self.read_records(spreadsheet_id, worksheet_name):
            if str(record.get(key_column, "")).strip().lower() == key_value.lower():
                return record
        return None

    def lookup_table_row_by_key(self, table_key, key_column, key_value):
        for record in self.read_table(table_key):
            if str(record.get(key_column, "")).strip().lower() == key_value.lower():
                return record
        return None

    def update_row_by_key(
        self,
        spreadsheet_id,
        key_column,
        key_value,
        updates,
        worksheet_name="Sheet1",
    ):
        self.updated.append((spreadsheet_id, key_column, key_value, updates))
        return True

    def update_table_row_by_key(self, table_key, key_column, key_value, updates):
        table_map = {
            "login_audit": "login",
            "processing_log": "processing",
            "job_state": "job_state",
        }
        return self.update_row_by_key(table_map.get(table_key, table_key), key_column, key_value, updates)


class FakeVerifier:
    def __init__(self, user=None, error=None):
        self.user = user or VerifiedGoogleUser("active@nobroker.in", "Active User")
        self.error = error

    def verify(self, credential):
        if self.error:
            raise self.error
        return self.user


def app_client(records=None, verifier=None):
    app = create_app(
        settings=settings(),
        sheets_service=FakeSheetsService(records),
        google_token_verifier=verifier or FakeVerifier(),
    )
    return TestClient(app), app


def user_records(active=True):
    return {
        "users": [
            {
                "Email": "active@nobroker.in",
                "Name": "Active User",
                "Role": "ADMIN",
                "Active": "true" if active else "false",
            }
        ],
        "folders": [
            {
                "Purpose": "MEMBER_RECEIPT",
                "Incoming Folder ID": "member-in",
                "Review Folder ID": "member-review",
                "Completed Folder ID": "member-done",
                "Output Folder ID": "member-out",
                "Active": "true",
            },
            {
                "Purpose": "VENDOR_INVOICE",
                "Incoming Folder ID": "vendor-in",
                "Review Folder ID": "vendor-review",
                "Completed Folder ID": "vendor-done",
                "Output Folder ID": "vendor-out",
                "Active": "true",
            },
        ],
    }


def login(client):
    response = client.post("/auth/google-login", json={"credential": "token"})
    assert response.status_code == 200
    return response.json()["session_token"]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def test_google_auth_verified_active_user_allowed():
    client, app = app_client(user_records())

    response = client.post("/auth/google-login", json={"credential": "token"})

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "active@nobroker.in"
    assert body["role"] == "ADMIN"
    assert body["session_token"]
    assert any(row[0] == "login" for row in app.state.sheets_service.appended)


def test_google_auth_inactive_user_denied():
    client, _ = app_client(user_records(active=False))

    response = client.post("/auth/google-login", json={"credential": "token"})

    assert response.status_code == 403
    assert response.json()["detail"] == "User is inactive."


def test_google_auth_unknown_user_denied():
    client, _ = app_client({"users": []})

    response = client.post("/auth/google-login", json={"credential": "token"})

    assert response.status_code == 403
    assert response.json()["detail"] == "User is not authorized."


def test_google_auth_invalid_token_denied():
    client, _ = app_client(user_records(), FakeVerifier(error=AuthError("bad token")))

    response = client.post("/auth/google-login", json={"credential": "token"})

    assert response.status_code == 401
    assert response.json()["detail"] == "bad token"


def test_session_created_heartbeat_and_logout():
    client, _ = app_client(user_records())
    token = login(client)

    heartbeat = client.post(
        "/auth/heartbeat",
        json={"user_active": True, "page_visible": True},
        headers=auth_headers(token),
    )
    assert heartbeat.status_code == 200
    assert heartbeat.json()["status"] == "ACTIVE"

    logout = client.post("/auth/logout", headers=auth_headers(token))
    assert logout.status_code == 200
    assert logout.json()["status"] == "LOGGED_OUT"

    after_logout = client.get("/auth/me", headers=auth_headers(token))
    assert after_logout.status_code == 401


def test_session_active_time_uses_heartbeat_grace():
    svc = SessionService(settings())
    session = svc.create_session(
        AuthorizedUser("active@nobroker.in", "Active User", "ADMIN")
    )
    session.last_seen_at = utc_now() - timedelta(seconds=500)

    svc.heartbeat(session.token, user_active=True, page_visible=True)

    assert session.active_duration_seconds == 120


def test_user_master_lookup_active_and_inactive():
    svc = UserMasterService(settings(), FakeSheetsService(user_records()))

    user = svc.authorize("active@nobroker.in")

    assert user.role == "ADMIN"

    inactive = UserMasterService(settings(), FakeSheetsService(user_records(active=False)))
    with pytest.raises(AuthorizationError):
        inactive.authorize("active@nobroker.in")


def test_activity_and_login_appends():
    client, app = app_client(user_records())

    login(client)

    sheet_ids = [sheet_id for sheet_id, _ in app.state.sheets_service.appended]
    assert "login" in sheet_ids
    assert "activity" in sheet_ids
    assert "sessions" in sheet_ids


def test_folder_router_member_and_vendor_routes():
    client, _ = app_client(user_records())
    token = login(client)

    member = client.get(
        "/config/folder-route?purpose=MEMBER_RECEIPT&status=incoming",
        headers=auth_headers(token),
    )
    vendor = client.get(
        "/config/folder-route?purpose=VENDOR_INVOICE&status=output",
        headers=auth_headers(token),
    )

    assert member.json()["folder_id"] == "member-in"
    assert vendor.json()["folder_id"] == "vendor-out"


def test_folder_router_missing_config_error():
    client, _ = app_client({"users": user_records()["users"], "folders": []})
    token = login(client)

    response = client.get(
        "/config/folder-route?purpose=MEMBER_RECEIPT&status=incoming",
        headers=auth_headers(token),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "FOLDER_CONFIGURATION_MISSING"


def test_template_registry_member_vendor_and_missing():
    client, _ = app_client(user_records())
    token = login(client)

    member = client.get("/config/template/MEMBER_RECEIPT", headers=auth_headers(token))
    vendor = client.get("/config/template/VENDOR_INVOICE", headers=auth_headers(token))
    missing = client.get("/config/template/MEMBER_PAYMENT", headers=auth_headers(token))

    assert member.json()["template_code"] == "NBH_MEMBER_RECEIPT_V1"
    assert vendor.json()["template_code"] == "NBH_VENDOR_BILL_V1"
    assert vendor.json()["supports_multiple_expense_entries"] is True
    assert missing.status_code == 404


def test_processing_config_validation():
    client, _ = app_client(user_records())
    token = login(client)

    response = client.post(
        "/processing/validate-config",
        json={"purpose": "MEMBER_RECEIPT", "destination_status": "incoming"},
        headers=auth_headers(token),
    )

    assert response.status_code == 200
    assert response.json()["template"]["template_code"] == "NBH_MEMBER_RECEIPT_V1"
    assert response.json()["destination"]["folder_id"] == "member-in"


class ExecResult:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class FakeDriveFiles:
    def __init__(self, folder_metadata):
        self.folder_metadata = folder_metadata
        self.created_body = None

    def get(self, fileId, fields):
        return ExecResult(self.folder_metadata)

    def create(self, body, media_body, fields):
        self.created_body = body
        return ExecResult({"id": "uploaded-file-id"})


class FakeDrive:
    def __init__(self, files):
        self._files = files

    def files(self):
        return self._files


def test_drive_upload_calls_correct_folder_and_sanitizes_name():
    svc = GoogleDriveService(settings())
    files = FakeDriveFiles(
        {
            "id": "folder-1",
            "mimeType": "application/vnd.google-apps.folder",
            "trashed": False,
        }
    )
    svc._service = FakeDrive(files)

    file_id = svc.upload_file('bad:name?.pdf', b"data", "folder-1", "application/pdf")

    assert file_id == "uploaded-file-id"
    assert files.created_body["parents"] == ["folder-1"]
    assert files.created_body["name"] == "bad_name_.pdf"


def test_drive_invalid_folder_rejected():
    svc = GoogleDriveService(settings())
    svc._service = FakeDrive(
        FakeDriveFiles({"id": "x", "mimeType": "text/plain", "trashed": False})
    )

    with pytest.raises(GoogleDriveError):
        svc.validate_folder_id("x")


def test_folder_router_service_raises_missing_config():
    class MissingConfig:
        def get_config(self, purpose):
            raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING")

    router = FolderRouterService(MissingConfig())

    with pytest.raises(FolderConfigurationError):
        router.route("MEMBER_RECEIPT", "incoming")


def test_template_registry_missing_purpose_error():
    client, _ = app_client(user_records())
    token = login(client)

    response = client.get("/config/template/UNKNOWN", headers=auth_headers(token))

    assert response.status_code == 404
    assert response.json()["detail"] == "TEMPLATE_CONFIGURATION_MISSING"


def test_shared_spreadsheet_table_resolution_prefers_legacy_id():
    shared = replace(
        settings(),
        google_accounting_spreadsheet_id="shared",
        google_user_master_sheet_id=None,
        google_folder_config_sheet_id=None,
    )
    svc = GoogleSheetsService(shared)

    assert svc.resolve_table("folder_config") == ("shared", "Folder_Config")

    legacy = replace(shared, google_user_master_sheet_id="legacy-users")
    assert GoogleSheetsService(legacy).resolve_table("user_master") == (
        "legacy-users",
        "Sheet1",
    )


def test_public_config_does_not_expose_secrets_or_resource_ids():
    client, _ = app_client(user_records())

    response = client.get("/config/public")

    assert response.status_code == 200
    body = response.json()
    raw = json.dumps(body)
    assert body["google_client_id"] == "client-id"
    assert "google_service_account" not in raw.lower()
    assert "users" not in raw
    assert "root" not in raw


def test_config_health_is_admin_only_and_reports_safe_statuses():
    client, _ = app_client(user_records())
    session_token = login(client)

    response = client.get("/config/health", headers=auth_headers(session_token))

    assert response.status_code == 200
    body = response.json()["status"]
    assert body["google_auth"] == "READY"
    assert body["user_master"] == "READY"
    assert body["folder_config"] == "READY"
    assert body["mapping_master"] == "READY"
    assert body["gemini"] == "MISSING"
    assert set(body.values()) <= {"READY", "MISSING", "NO_ACCESS", "CONFIGURED"}

    non_admin_records = user_records()
    non_admin_records["users"][0]["Role"] = "USER"
    non_admin, _ = app_client(non_admin_records)
    user_token = login(non_admin)
    forbidden = non_admin.get("/config/health", headers=auth_headers(user_token))
    assert forbidden.status_code == 403
