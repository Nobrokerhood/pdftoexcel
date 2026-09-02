import logging
from datetime import datetime

from app.core.config import Settings
from app.google.sheets_service import (
    GoogleSheetsNotConfiguredError,
    GoogleSheetsService,
)


logger = logging.getLogger(__name__)


class GoogleSheetsAuditClient:
    def __init__(
        self,
        settings: Settings,
        sheets_service: GoogleSheetsService | None = None,
    ):
        self.settings = settings
        self.sheets_service = sheets_service or GoogleSheetsService(settings)

    def append_login(
        self,
        email: str,
        name: str | None,
        login_time: str,
        ip: str,
        user_agent: str,
    ) -> bool:
        try:
            return self.sheets_service.append_table_row(
                "login_audit",
                [
                    "",
                    email,
                    name or "",
                    login_time,
                    "",
                    "SUCCESS",
                    ip,
                    user_agent,
                ],
            )
        except GoogleSheetsNotConfiguredError:
            logger.info("Login audit skipped because Google Sheets is not configured.")
            return False
        except Exception as exc:
            logger.error("Login logging failed: %s", exc)
            return False

    def append_usage(
        self,
        email: str,
        method: str,
        path: str,
        status: str,
        process_time: float,
        ip: str,
        user_agent: str,
    ) -> bool:
        try:
            if self.settings.google_api_usage_sheet_id:
                return self.sheets_service.append_row(
                    self.settings.google_api_usage_sheet_id,
                    [
                        email,
                        method,
                        path,
                        status,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        process_time,
                        ip,
                        user_agent,
                    ],
                )
            return False
        except GoogleSheetsNotConfiguredError:
            return False
        except Exception as exc:
            logger.error("Usage logging failed: %s", exc)
            return False
