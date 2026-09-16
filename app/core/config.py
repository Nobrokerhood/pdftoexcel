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
    # Temporary second key, used only when the primary project hits its spending cap
    # and GEMINI_FALLBACK_ENABLED is true.
    gemini_api_key_fallback: str | None = None
    gemini_fallback_enabled: bool = False
    # Folder containing Poppler's pdfinfo/pdftoppm; when unset they must be on PATH.
    poppler_path: str | None = None
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    enable_docs: bool = True
    google_shared_drive_id: str | None = None
    session_secret: str | None = None


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
        google_client_id=(os.getenv("GOOGLE_CLIENT_ID") or "").strip() or None,
        allowed_email_domain=os.getenv("ALLOWED_EMAIL_DOMAIN", "nobroker.in"),
        allow_domain_wide_access=_flag(os.getenv("ALLOW_DOMAIN_WIDE_ACCESS", "true")),
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
        gemini_api_key_fallback=os.getenv("GEMINI_API_KEY_FALLBACK"),
        gemini_fallback_enabled=_flag(os.getenv("GEMINI_FALLBACK_ENABLED")),
        poppler_path=(
            os.getenv("POPPLER_PATH").strip()
            if os.getenv("POPPLER_PATH") and os.path.exists(os.getenv("POPPLER_PATH").strip())
            else None
        ),
        environment=os.getenv("ENVIRONMENT", "development").strip().lower(),
        debug=_flag(os.getenv("DEBUG", "false")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        enable_docs=_flag(os.getenv("ENABLE_DOCS", "false" if os.getenv("ENVIRONMENT") == "production" else "true")),
        google_shared_drive_id=(os.getenv("GOOGLE_SHARED_DRIVE_ID") or "").strip() or None,
        session_secret=(os.getenv("SESSION_SECRET") or "").strip() or None,
    )


def validate_production_config(settings: Settings) -> list[str]:
    """
    Validates mandatory settings when running in production mode.
    Does NOT require GOOGLE_DRIVE_ROOT_FOLDER_ID (preserves purpose-specific folders in Shared Drive).
    """
    if settings.environment != "production":
        return []

    errors: list[str] = []
    if not settings.gemini_api_key:
        errors.append("GEMINI_API_KEY is required in production.")
    if not settings.google_client_id:
        errors.append("GOOGLE_CLIENT_ID is required in production.")
    if not (settings.google_service_account_json or settings.google_service_account_file):
        errors.append("Google Service Account credentials (GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE) are required in production.")
    if not settings.google_accounting_spreadsheet_id:
        errors.append("GOOGLE_ACCOUNTING_SPREADSHEET_ID is required in production.")
    if not settings.session_secret:
        errors.append("SESSION_SECRET is required in production.")
    if settings.allow_dev_login:
        errors.append("ALLOW_DEV_LOGIN must be false in production.")
    if any(origin.strip() == "*" for origin in settings.cors_allowed_origins):
        errors.append("CORS_ALLOWED_ORIGINS cannot contain '*' wildcard in production.")

    return errors


def get_config_diagnostic(settings: Settings) -> dict[str, bool]:
    """
    Returns safe boolean configuration flags for production diagnostics.
    Never exposes credentials or secrets.
    """
    client_id = (settings.google_client_id or "").strip()
    return {
        "google_client_id_configured": bool(client_id),
        "google_client_id_format_valid": bool(client_id.endswith(".apps.googleusercontent.com")),
        "google_client_secret_configured": bool(os.getenv("GOOGLE_CLIENT_SECRET")),
        "gemini_configured": bool(settings.gemini_api_key),
        "service_account_configured": bool(
            settings.google_service_account_json or settings.google_service_account_file
        ),
        "shared_drive_configured": bool(
            settings.google_shared_drive_id or settings.google_drive_root_folder_id
        ),
        "sheets_configured": bool(
            settings.google_accounting_spreadsheet_id or settings.google_user_master_sheet_id
        ),
        "session_secret_configured": bool(settings.session_secret),
    }

