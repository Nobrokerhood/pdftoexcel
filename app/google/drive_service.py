import io
import logging
import re
from typing import BinaryIO

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from app.core.config import Settings
from app.google.sheets_service import SCOPES


logger = logging.getLogger(__name__)


class GoogleDriveError(RuntimeError):
    pass


class GoogleDriveNotConfiguredError(GoogleDriveError):
    pass


class GoogleDriveService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._service = None
        self._disabled_reason: str | None = None

    def _credentials(self):
        if self.settings.google_service_account_json:
            import json

            info = json.loads(self.settings.google_service_account_json)
            return Credentials.from_service_account_info(info, scopes=SCOPES)
        if self.settings.google_service_account_file:
            return Credentials.from_service_account_file(
                self.settings.google_service_account_file, scopes=SCOPES
            )
        raise GoogleDriveNotConfiguredError("Google Drive credentials not configured.")

    def _drive(self):
        if self._service is not None:
            return self._service
        if self._disabled_reason:
            raise GoogleDriveNotConfiguredError(self._disabled_reason)

        try:
            self._service = build("drive", "v3", credentials=self._credentials())
            return self._service
        except GoogleDriveNotConfiguredError:
            raise
        except Exception as exc:
            self._disabled_reason = "Google Drive authorization failed."
            logger.warning("%s %s", self._disabled_reason, exc)
            raise GoogleDriveError(self._disabled_reason) from exc

    @staticmethod
    def safe_filename(filename: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:180] or "upload"

    def validate_folder_id(self, folder_id: str) -> bool:
        if not folder_id:
            raise GoogleDriveError("Folder ID is required.")

        metadata = (
            self._drive()
            .files()
            .get(fileId=folder_id, fields="id,mimeType,trashed")
            .execute()
        )
        if metadata.get("trashed"):
            raise GoogleDriveError("Folder is trashed.")
        if metadata.get("mimeType") != "application/vnd.google-apps.folder":
            raise GoogleDriveError("Drive ID is not a folder.")
        return True

    def upload_file(
        self,
        filename: str,
        content: bytes | BinaryIO,
        folder_id: str,
        mime_type: str = "application/octet-stream",
    ) -> str:
        self.validate_folder_id(folder_id)
        body = {"name": self.safe_filename(filename), "parents": [folder_id]}
        stream = content if hasattr(content, "read") else io.BytesIO(content)
        media = MediaIoBaseUpload(stream, mimetype=mime_type, resumable=False)
        result = (
            self._drive()
            .files()
            .create(body=body, media_body=media, fields="id")
            .execute()
        )
        return result["id"]

    def move_file(self, file_id: str, folder_id: str) -> bool:
        self.validate_folder_id(folder_id)
        file_metadata = (
            self._drive().files().get(fileId=file_id, fields="parents").execute()
        )
        previous_parents = ",".join(file_metadata.get("parents", []))
        self._drive().files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id,parents",
        ).execute()
        return True

    def get_file_metadata(self, file_id: str) -> dict:
        return (
            self._drive()
            .files()
            .get(fileId=file_id, fields="id,name,mimeType,parents,trashed")
            .execute()
        )

    def download_file(self, file_id: str) -> bytes:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._drive().files().get_media(fileId=file_id)
        stream = io.BytesIO()
        downloader = MediaIoBaseDownload(stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return stream.getvalue()

    def create_folder(self, name: str, parent_id: str | None = None) -> str:
        body = {
            "name": self.safe_filename(name),
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            self.validate_folder_id(parent_id)
            body["parents"] = [parent_id]
        result = self._drive().files().create(body=body, fields="id").execute()
        return result["id"]
