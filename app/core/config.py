import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    google_client_id: str | None
    frontend_google_client_id: str | None
    allowed_email_domain: str | None
    allow_domain_wide_access: bool
    session_inactivity_seconds: int
    session_heartbeat_grace_seconds: int
    ai_verification_max_retries: int
    allow_dev_login: bool
    gemini_api_key: str | None
    gemini_model: str
    max_file_size_mb: int
    google_service_account_json: str | None
    google_service_account_file: str | None
    google_accounting_spreadsheet_id: str | None
    google_user_master_sheet_id: str | None
    google_login_audit_sheet_id: str | None
    google_api_usage_sheet_id: str | None
    google_session_log_sheet_id: str | None
    google_activity_log_sheet_id: str | None
    google_processing_log_sheet_id: str | None
    google_template_master_sheet_id: str | None
    google_folder_config_sheet_id: str | None
    google_mapping_master_sheet_id: str | None
    google_drive_root_folder_id: str | None
    google_login_audit_sheet_name: str
    google_api_usage_sheet_name: str
    cors_allowed_origins: tuple[str, ...]


def _flag(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


@lru_cache
def get_settings() -> Settings:
    default_origins = (
        "https://nobrokerhood.github.io",
        "https://nobrokerhood.github.io/pdftoexcel",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    )
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS")

    return Settings(
        google_client_id=os.getenv("GOOGLE_CLIENT_ID"),
        frontend_google_client_id=os.getenv("VITE_GOOGLE_CLIENT_ID")
        or os.getenv("GOOGLE_CLIENT_ID")
        or "414963441128-69gsdlfdfn8hrf7ovgc9mfh10spnc5nq.apps.googleusercontent.com",
        allowed_email_domain=os.getenv("ALLOWED_EMAIL_DOMAIN", "nobroker.in"),
        allow_domain_wide_access=_flag(os.getenv("ALLOW_DOMAIN_WIDE_ACCESS")),
        session_inactivity_seconds=int(os.getenv("SESSION_INACTIVITY_SECONDS", "1200")),
        session_heartbeat_grace_seconds=int(
            os.getenv("SESSION_HEARTBEAT_GRACE_SECONDS", "120")
        ),
        ai_verification_max_retries=int(os.getenv("AI_VERIFICATION_MAX_RETRIES", "2")),
        allow_dev_login=_flag(os.getenv("ALLOW_DEV_LOGIN")),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "10")),
        google_service_account_json=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
        google_service_account_file=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE"),
        google_accounting_spreadsheet_id=os.getenv("GOOGLE_ACCOUNTING_SPREADSHEET_ID"),
        google_user_master_sheet_id=os.getenv("GOOGLE_USER_MASTER_SHEET_ID"),
        google_login_audit_sheet_id=os.getenv("GOOGLE_LOGIN_AUDIT_SHEET_ID"),
        google_api_usage_sheet_id=os.getenv("GOOGLE_API_USAGE_SHEET_ID"),
        google_session_log_sheet_id=os.getenv("GOOGLE_SESSION_LOG_SHEET_ID"),
        google_activity_log_sheet_id=os.getenv("GOOGLE_ACTIVITY_LOG_SHEET_ID"),
        google_processing_log_sheet_id=os.getenv("GOOGLE_PROCESSING_LOG_SHEET_ID"),
        google_template_master_sheet_id=os.getenv("GOOGLE_TEMPLATE_MASTER_SHEET_ID"),
        google_folder_config_sheet_id=os.getenv("GOOGLE_FOLDER_CONFIG_SHEET_ID"),
        google_mapping_master_sheet_id=os.getenv("GOOGLE_MAPPING_MASTER_SHEET_ID"),
        google_drive_root_folder_id=os.getenv("GOOGLE_DRIVE_ROOT_FOLDER_ID"),
        google_login_audit_sheet_name=os.getenv(
            "GOOGLE_LOGIN_AUDIT_SHEET_NAME", "Accounting_AI_Login_Audit"
        ),
        google_api_usage_sheet_name=os.getenv(
            "GOOGLE_API_USAGE_SHEET_NAME", "API_Usage_Report"
        ),
        cors_allowed_origins=(
            _split_csv(configured_origins) if configured_origins else default_origins
        ),
    )
