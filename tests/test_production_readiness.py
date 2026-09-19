from tests.conftest import png_bytes
import io
import pytest
from starlette.testclient import TestClient

from app.app_factory import create_app
from app.auth.user_master import AuthorizedUser
from app.core.config import Settings, validate_production_config
from tests.test_accounting_workflow import (
    MEMBER_DATA,
    FakeDriveService,
    FakeSheetsService,
    FakeVerifier,
    client_for,
    headers,
    records,
    start_job,
    token,
)


def test_dev_login_disabled_when_flag_is_false():
    client, _, _, _ = client_for()
    # allow_dev_login is False in default test settings
    res = client.post("/auth/dev-login", json={"email": "attacker@nobroker.in"})
    assert res.status_code == 403
    assert res.json()["detail"] == "DEV_LOGIN_DISABLED"


def test_dev_login_disabled_when_environment_is_production():
    import dataclasses
    from tests.test_accounting_workflow import (
        FakeDriveService,
        FakeSheetsService,
        FakeVerifier,
        StaticExtractionProvider,
        StaticRepairProvider,
        StaticVerificationProvider,
        MEMBER_DATA,
        VENDOR_DATA,
        records,
        settings as base_settings,
    )
    # If allow_dev_login is set to True in production, create_app must fail fast
    s = dataclasses.replace(
        base_settings(),
        environment="production",
        allow_dev_login=True,
        gemini_api_key="key",
        google_client_id="client",
        session_secret="test-session-secret",
        google_service_account_json='{"client_email": "test@nobroker.in"}',
        google_accounting_spreadsheet_id="sheet-id",
    )
    with pytest.raises(RuntimeError) as exc_info:
        create_app(
            settings=s,
            sheets_service=FakeSheetsService(records()),
            drive_service=FakeDriveService(),
            google_token_verifier=FakeVerifier(),
            extraction_provider=StaticExtractionProvider({"MEMBER_RECEIPT": MEMBER_DATA, "VENDOR_INVOICE": VENDOR_DATA}),
            verification_provider=StaticVerificationProvider(),
            repair_provider=StaticRepairProvider(MEMBER_DATA),
        )
    assert "ALLOW_DEV_LOGIN must be false in production" in str(exc_info.value)


def test_file_upload_validation_rejects_empty_file():
    client, _, _, _ = client_for()
    session_token = token(client)
    res = client.post(
        "/processing/jobs",
        headers=headers(session_token),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "File is empty."


def test_file_upload_validation_rejects_invalid_extension():
    client, _, _, _ = client_for()
    session_token = token(client)
    res = client.post(
        "/processing/jobs",
        headers=headers(session_token),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": ("malicious.exe", b"%PDF-test", "application/pdf")},
    )
    assert res.status_code == 400
    assert "Invalid file extension" in res.json()["detail"]


def test_file_upload_validation_rejects_mismatched_magic_bytes():
    client, _, _, _ = client_for()
    session_token = token(client)
    res = client.post(
        "/processing/jobs",
        headers=headers(session_token),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": ("fake.pdf", b"NOT_A_PDF_CONTENT", "application/pdf")},
    )
    assert res.status_code == 400
    assert "Invalid PDF file signature" in res.json()["detail"]


def test_file_upload_sanitizes_filename_and_prevents_path_traversal():
    client, _, _, _ = client_for()
    session_token = token(client)
    res = client.post(
        "/processing/jobs",
        headers=headers(session_token),
        data={"purpose": "MEMBER_RECEIPT"},
        files={"file": ("../../../../etc/passwd.png", png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    job = res.json()
    assert ".." not in job["source_filename"]
    assert "/" not in job["source_filename"]
    assert "\\" not in job["source_filename"]
    assert job["source_filename"] == "passwd.png"


def test_synthetic_signature_bypass_is_gone():
    """The upload validator used to accept b"SYNTHETIC" as a PDF/JPEG/PNG signature."""
    client, _, _, _ = client_for()
    session_token = token(client)
    for name, ctype in (("a.pdf", "application/pdf"), ("a.jpg", "image/jpeg"), ("a.png", "image/png")):
        res = client.post("/processing/jobs", headers=headers(session_token), data={"purpose": "MEMBER_RECEIPT"},
                          files={"file": (name, b"SYNTHETIC TEST DATA", ctype)})
        assert res.status_code == 400, name


def test_image_with_valid_magic_but_undecodable_body_is_rejected():
    client, _, _, _ = client_for()
    res = client.post("/processing/jobs", headers=headers(token(client)), data={"purpose": "MEMBER_RECEIPT"},
                      files={"file": ("a.png", b"\x89PNG\r\n\x1a\n garbage", "image/png")})
    assert res.status_code == 400


def test_job_access_isolation_between_users():
    sheet_data = records()
    sheet_data["users"].append({"Email": "other@nobroker.in", "Name": "Other User", "Role": "USER", "Active": "true"})

    client, app, _, _ = client_for(sheet_records=sheet_data)

    # Login as User A
    token_a = token(client)
    res_job = start_job(client, token_a, "MEMBER_RECEIPT")
    assert res_job.status_code == 200
    job = res_job.json()
    job_id = job["job_id"]

    # Login as User B
    session_b = app.state.session_service.create_session(
        AuthorizedUser(email="other@nobroker.in", name="Other User", role="USER")
    )
    auth_b = headers(session_b.token)

    # 1. User B cannot GET User A's job
    res_get = client.get(f"/processing/jobs/{job_id}", headers=auth_b)
    assert res_get.status_code == 403
    assert res_get.json()["detail"] == "JOB_ACCESS_DENIED"

    # 2. User B cannot apply corrections to User A's job
    res_corr = client.post(
        f"/processing/jobs/{job_id}/corrections",
        headers=auth_b,
        json={"corrections": {"amount": "9999"}},
    )
    assert res_corr.status_code == 403
    assert res_corr.json()["detail"] == "JOB_ACCESS_DENIED"

    # 3. User B cannot approve User A's job
    res_appr = client.post(f"/processing/jobs/{job_id}/approve", headers=auth_b)
    assert res_appr.status_code == 403
    assert res_appr.json()["detail"] == "JOB_ACCESS_DENIED"

    # 4. User B cannot reject User A's job
    res_rej = client.post(f"/processing/jobs/{job_id}/reject", headers=auth_b, json={"reason": "OTHER"})
    assert res_rej.status_code == 403
    assert res_rej.json()["detail"] == "JOB_ACCESS_DENIED"

    # 5. User B cannot download User A's job
    res_down = client.get(f"/processing/jobs/{job_id}/download", headers=auth_b)
    assert res_down.status_code == 403
    assert res_down.json()["detail"] == "JOB_ACCESS_DENIED"

    # 6. User B cannot view User A's source document
    res_src = client.get(f"/processing/jobs/{job_id}/source", headers=auth_b)
    assert res_src.status_code == 403
    assert res_src.json()["detail"] == "JOB_ACCESS_DENIED"


def test_past_work_lists_only_own_jobs():
    sheet_data = records()
    sheet_data["users"].append({"Email": "user_b@nobroker.in", "Name": "User B", "Role": "USER", "Active": "true"})
    client, app, _, _ = client_for(sheet_records=sheet_data)

    token_a = token(client)
    res_job = start_job(client, token_a, "MEMBER_RECEIPT")
    assert res_job.status_code == 200

    session_b = app.state.session_service.create_session(
        AuthorizedUser(email="user_b@nobroker.in", name="User B", role="USER")
    )
    res_b = client.get("/processing/jobs", headers=headers(session_b.token))
    assert res_b.status_code == 200
    # User B should see 0 jobs
    assert len(res_b.json()["jobs"]) == 0


def test_source_document_endpoint_streams_for_owner():
    client, _, _, _ = client_for()
    token_a = token(client)
    res_job = start_job(client, token_a, "MEMBER_RECEIPT")
    assert res_job.status_code == 200
    job = res_job.json()

    res = client.get(f"/processing/jobs/{job['job_id']}/source", headers=headers(token_a))
    assert res.status_code == 200
    assert res.content == png_bytes()


def test_health_and_readiness_endpoints():
    client, _, _, _ = client_for()

    # Health probe is public and fast
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Readiness reflects real capability probes (it used to hardcode rapidocr_ready=True)
    res_ready = client.get("/readiness")
    body = res_ready.json()
    assert res_ready.status_code in (200, 503)
    assert body["overall"] in ("READY", "DEGRADED", "NOT_READY")
    assert (res_ready.status_code == 503) == (body["overall"] == "NOT_READY")


def test_production_config_validation_fails_fast_on_missing_keys():
    # In development mode, validation returns no errors
    dev_settings = Settings(
        google_client_id=None,
        allowed_email_domain=None,
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
        google_login_audit_sheet_name="Login",
        google_api_usage_sheet_name="Usage",
        cors_allowed_origins=("http://localhost:5000",),
        environment="development",
    )
    assert validate_production_config(dev_settings) == []

    # In production mode, missing mandatory keys produce explicit errors
    prod_settings = Settings(
        google_client_id=None,
        allowed_email_domain=None,
        allow_domain_wide_access=False,
        session_inactivity_seconds=1200,
        session_heartbeat_grace_seconds=120,
        ai_verification_max_retries=2,
        allow_dev_login=True,  # Disallowed in prod!
        gemini_api_key=None,
        gemini_model="gemini-2.5-flash",
        max_file_size_mb=10,
        google_service_account_json=None,
        google_service_account_file=None,
        google_accounting_spreadsheet_id=None,
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
        google_login_audit_sheet_name="Login",
        google_api_usage_sheet_name="Usage",
        cors_allowed_origins=("https://accounting.nobrokerhood.com",),
        environment="production",
        session_secret=None,
    )
    errors = validate_production_config(prod_settings)
    assert len(errors) > 0
    assert any("GEMINI_API_KEY" in e for e in errors)
    assert any("GOOGLE_CLIENT_ID" in e for e in errors)
    assert any("SESSION_SECRET" in e for e in errors)
    assert any("ALLOW_DEV_LOGIN" in e for e in errors)
