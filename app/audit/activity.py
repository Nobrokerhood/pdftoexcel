import logging
from datetime import datetime
from typing import Any

from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService


logger = logging.getLogger(__name__)


class AuditLogService:
    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def _append(self, table_key: str, values: list[Any]) -> bool:
        try:
            return self.sheets_service.append_table_row(table_key, values)
        except GoogleSheetsNotConfiguredError:
            logger.info("Audit append skipped because sheet is not configured.")
            return False
        except Exception as exc:
            logger.error("Audit append failed: %s", exc)
            return False

    def login(self, session_id: str, email: str, name: str, ip: str, user_agent: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._append(
            "login_audit",
            [session_id, email, name, now, "", "SUCCESS", ip, user_agent],
        )

    def logout(self, session_id: str, email: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.activity(session_id, email, "", "LOGOUT", "auth", "", "", "OK", "")
        try:
            return self.sheets_service.update_table_row_by_key(
                "login_audit",
                "Session ID",
                session_id,
                {"Logout Time": now},
            )
        except GoogleSheetsNotConfiguredError:
            logger.info("Logout audit update skipped because sheet is not configured.")
            return False
        except Exception as exc:
            logger.error("Logout audit update failed: %s", exc)
            return False

    def session_snapshot(self, session) -> bool:
        return self._append(
            "session_log",
            [
                session.session_id,
                session.email,
                session.login_at.isoformat(),
                session.last_seen_at.isoformat(),
                session.logout_at.isoformat() if session.logout_at else "",
                int((session.last_seen_at - session.login_at).total_seconds()),
                session.active_duration_seconds,
                session.status,
            ],
        )

    def activity(
        self,
        session_id: str,
        user_email: str,
        job_id: str,
        action: str,
        purpose: str,
        source_file_id: str,
        output_file_id: str,
        status: str,
        details: str,
    ) -> bool:
        return self._append(
            "activity_log",
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                session_id,
                user_email,
                job_id,
                action,
                purpose,
                source_file_id,
                output_file_id,
                status,
                details,
            ],
        )


class ProcessingLogService:
    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def append_started(
        self,
        job_id: str,
        session_id: str,
        user_email: str,
        purpose: str,
        template_code: str,
        source_filename: str,
        source_drive_file_id: str = "",
        source_folder_id: str = "",
        overall_status: str = "STARTED",
    ) -> bool:
        try:
            return self.sheets_service.append_table_row(
                "processing_log",
                [
                    job_id,
                    session_id,
                    user_email,
                    purpose,
                    template_code,
                    source_filename,
                    source_drive_file_id,
                    source_folder_id,
                    "NOT_STARTED",
                    "NOT_STARTED",
                    "NOT_STARTED",
                    "NOT_STARTED",
                    "NOT_STARTED",
                    "",
                    "",
                    overall_status,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "",
                ],
            )
        except GoogleSheetsNotConfiguredError:
            logger.info("Processing log skipped because sheet is not configured.")
            return False

    def update_job(self, job) -> bool:
        updates = {
            "Source Drive File ID": job.source_drive_file_id,
            "Source Folder ID": job.source_folder_id,
            "Extraction Status": job.extraction_status,
            "Verification Status": job.verification_status,
            "Mapping Status": job.mapping_status,
            "Validation Status": job.validation_status,
            "Human Status": job.human_status,
            "Output Filename": job.output_filename,
            "Output Drive File ID": job.output_drive_file_id,
            "Overall Status": job.overall_status,
            "Completed At": job.completed_at,
        }
        try:
            return self.sheets_service.update_table_row_by_key(
                "processing_log",
                "Job ID",
                job.job_id,
                updates,
            )
        except GoogleSheetsNotConfiguredError:
            logger.info("Processing log update skipped because sheet is not configured.")
            return False
        except Exception as exc:
            logger.error("Processing log update failed: %s", exc)
            return False
