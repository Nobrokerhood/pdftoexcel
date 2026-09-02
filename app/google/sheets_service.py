import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2.service_account import Credentials

from app.core.config import Settings
from app.google.sheet_schemas import schema_for


logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class GoogleSheetsError(RuntimeError):
    pass


class GoogleSheetsNotConfiguredError(GoogleSheetsError):
    pass


@dataclass
class CachedRecords:
    loaded_at: float
    records: list[dict[str, Any]]


class GoogleSheetsService:
    def __init__(self, settings: Settings, cache_ttl_seconds: int = 60):
        self.settings = settings
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = None
        self._cache: dict[tuple[str, str], CachedRecords] = {}
        self._disabled_reason: str | None = None

    def _authorize(self):
        if self._client is not None:
            return self._client
        if self._disabled_reason:
            raise GoogleSheetsNotConfiguredError(self._disabled_reason)

        try:
            if self.settings.google_service_account_json:
                info = json.loads(self.settings.google_service_account_json)
                credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
            elif self.settings.google_service_account_file:
                credentials = Credentials.from_service_account_file(
                    self.settings.google_service_account_file, scopes=SCOPES
                )
            else:
                self._disabled_reason = "Google Sheets credentials not configured."
                raise GoogleSheetsNotConfiguredError(self._disabled_reason)

            self._client = gspread.authorize(credentials)
            return self._client
        except GoogleSheetsNotConfiguredError:
            raise
        except Exception as exc:
            self._disabled_reason = "Google Sheets authorization failed."
            logger.warning("%s %s", self._disabled_reason, exc)
            raise GoogleSheetsError(self._disabled_reason) from exc

    def _worksheet(self, spreadsheet_id: str | None, worksheet_name: str = "Sheet1"):
        if not spreadsheet_id:
            raise GoogleSheetsNotConfiguredError("Google Sheet ID is not configured.")

        client = self._authorize()
        spreadsheet = client.open_by_key(spreadsheet_id)
        return spreadsheet.worksheet(worksheet_name)

    def spreadsheet(self, spreadsheet_id: str | None = None):
        spreadsheet_id = spreadsheet_id or self.settings.google_accounting_spreadsheet_id
        if not spreadsheet_id:
            raise GoogleSheetsNotConfiguredError("Google Spreadsheet ID is not configured.")
        return self._authorize().open_by_key(spreadsheet_id)

    def resolve_table(self, table_key: str) -> tuple[str | None, str]:
        schema = schema_for(table_key)
        legacy_id = (
            getattr(self.settings, schema.legacy_setting_name)
            if schema.legacy_setting_name
            else None
        )
        if legacy_id:
            return legacy_id, "Sheet1"
        return self.settings.google_accounting_spreadsheet_id, schema.tab_name

    def read_records(
        self,
        spreadsheet_id: str | None,
        worksheet_name: str = "Sheet1",
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        key = (spreadsheet_id or "", worksheet_name)
        cached = self._cache.get(key)
        if use_cache and cached and time.time() - cached.loaded_at < self.cache_ttl_seconds:
            return [record.copy() for record in cached.records]

        worksheet = self._worksheet(spreadsheet_id, worksheet_name)
        records = worksheet.get_all_records()
        self._cache[key] = CachedRecords(time.time(), records)
        return [record.copy() for record in records]

    def read_table(
        self,
        table_key: str,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        return self.read_records(spreadsheet_id, worksheet_name, use_cache)

    def table_headers(self, table_key: str) -> list[str]:
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        return self._worksheet(spreadsheet_id, worksheet_name).row_values(1)

    def ensure_table(self, table_key: str) -> tuple[bool, str]:
        schema = schema_for(table_key)
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        if not spreadsheet_id:
            raise GoogleSheetsNotConfiguredError("Google Spreadsheet ID is not configured.")

        spreadsheet = self.spreadsheet(spreadsheet_id)
        created = False
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(
                title=worksheet_name,
                rows=max(100, len(schema.headers) + 10),
                cols=max(26, len(schema.headers)),
            )
            created = True

        headers = worksheet.row_values(1)
        if not headers:
            worksheet.append_row(list(schema.headers))
            self._cache.pop((spreadsheet_id or "", worksheet_name), None)
            return True, "CREATED" if created else "HEADER_ADDED"
        if headers != list(schema.headers):
            return False, "MISMATCH"
        return False, "READY"

    def append_row(
        self,
        spreadsheet_id: str | None,
        values: list[Any],
        worksheet_name: str = "Sheet1",
    ) -> bool:
        worksheet = self._worksheet(spreadsheet_id, worksheet_name)
        worksheet.append_row(values)
        self._cache.pop((spreadsheet_id or "", worksheet_name), None)
        return True

    def append_table_row(self, table_key: str, values: list[Any]) -> bool:
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        return self.append_row(spreadsheet_id, values, worksheet_name)

    def lookup_row_by_key(
        self,
        spreadsheet_id: str | None,
        key_column: str,
        key_value: str,
        worksheet_name: str = "Sheet1",
    ) -> dict[str, Any] | None:
        for record in self.read_records(spreadsheet_id, worksheet_name):
            if str(record.get(key_column, "")).strip().lower() == key_value.lower():
                return record
        return None

    def lookup_table_row_by_key(
        self,
        table_key: str,
        key_column: str,
        key_value: str,
    ) -> dict[str, Any] | None:
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        return self.lookup_row_by_key(
            spreadsheet_id, key_column, key_value, worksheet_name
        )

    def update_row_by_key(
        self,
        spreadsheet_id: str | None,
        key_column: str,
        key_value: str,
        updates: dict[str, Any],
        worksheet_name: str = "Sheet1",
    ) -> bool:
        worksheet = self._worksheet(spreadsheet_id, worksheet_name)
        rows = worksheet.get_all_values()
        if not rows:
            return False

        headers = rows[0]
        try:
            key_index = headers.index(key_column)
        except ValueError:
            return False

        for row_number, row in enumerate(rows[1:], start=2):
            if key_index < len(row) and row[key_index].strip().lower() == key_value.lower():
                for column_name, value in updates.items():
                    if column_name in headers:
                        worksheet.update_cell(
                            row_number, headers.index(column_name) + 1, value
                        )
                self._cache.pop((spreadsheet_id or "", worksheet_name), None)
                return True
        return False

    def update_table_row_by_key(
        self,
        table_key: str,
        key_column: str,
        key_value: str,
        updates: dict[str, Any],
    ) -> bool:
        spreadsheet_id, worksheet_name = self.resolve_table(table_key)
        return self.update_row_by_key(
            spreadsheet_id, key_column, key_value, updates, worksheet_name
        )
