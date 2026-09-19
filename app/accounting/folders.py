from dataclasses import dataclass
import logging
from typing import Any

from app.accounting.purposes import supported_purpose_codes
from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService

logger = logging.getLogger(__name__)


class FolderConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class FolderConfig:
    purpose: str
    incoming_folder_id: str
    review_folder_id: str
    completed_folder_id: str
    output_folder_id: str
    active: bool = True

    def folder_for_status(self, status: str) -> str:
        status = status.strip().lower()
        mapping = {
            "incoming": self.incoming_folder_id,
            "review": self.review_folder_id,
            "completed": self.completed_folder_id,
            "output": self.output_folder_id,
        }
        folder_id = mapping.get(status)
        if not folder_id:
            raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING")
        return folder_id


def _active(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "active"}


class FolderConfigService:
    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def get_config(self, purpose: str) -> FolderConfig:
        purpose = purpose.strip().upper()
        if purpose not in supported_purpose_codes():
            raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING")

        import os

        # Check environment variable overrides if configured
        inc = os.getenv(f"{purpose}_INCOMING_FOLDER_ID")
        rev = os.getenv(f"{purpose}_REVIEW_FOLDER_ID")
        comp = os.getenv(f"{purpose}_COMPLETED_FOLDER_ID")
        out = os.getenv(f"{purpose}_OUTPUT_FOLDER_ID")
        if inc and rev and comp and out:
            return FolderConfig(
                purpose=purpose,
                incoming_folder_id=inc.strip(),
                review_folder_id=rev.strip(),
                completed_folder_id=comp.strip(),
                output_folder_id=out.strip(),
                active=True,
            )

        try:
            records = self.sheets_service.read_table("folder_config")
        except GoogleSheetsNotConfiguredError as exc:
            raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING") from exc
        except Exception as exc:
            logger.warning("Could not read folder_config from Google Sheets: %s", exc)
            raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING") from exc

        for record in records:
            if (
                str(record.get("Purpose", "")).strip().upper() == purpose
                and _active(record.get("Active", False))
            ):
                return FolderConfig(
                    purpose=purpose,
                    incoming_folder_id=str(record.get("Incoming Folder ID", "")).strip(),
                    review_folder_id=str(record.get("Review Folder ID", "")).strip(),
                    completed_folder_id=str(record.get("Completed Folder ID", "")).strip(),
                    output_folder_id=str(record.get("Output Folder ID", "")).strip(),
                    active=True,
                )

        raise FolderConfigurationError("FOLDER_CONFIGURATION_MISSING")


class FolderRouterService:
    def __init__(self, folder_config_service: FolderConfigService):
        self.folder_config_service = folder_config_service

    def route(self, purpose: str, workflow_status: str) -> str:
        config = self.folder_config_service.get_config(purpose)
        return config.folder_for_status(workflow_status)
