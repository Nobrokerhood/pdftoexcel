import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import gspread
from gspread.exceptions import WorksheetNotFound
from gspread.utils import ValueInputOption, rowcol_to_a1
from google.oauth2.service_account import Credentials

from app.core.config import Settings
from app.google.sheet_schemas import schema_for


logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MAX_CELL_CHARS = 45000


def safe_cell_value(val: Any) -> Any:
    if isinstance(val, str) and len(val) > MAX_CELL_CHARS:
        return val[:MAX_CELL_CHARS] + "...[TRUNCATED]"
    return val


class GoogleSheetsError(RuntimeError):
    pass


class GoogleSheetsNotConfiguredError(GoogleSheetsError):
    pass


@dataclass
class CachedRecords:
    loaded_at: float
    records: list[dict[str, Any]]


class SheetsWriteQueue:
    """Ordered background writer for audit / log / state rows.

    * Sheets quota (429) is retried with backoff instead of failing a request.
    * Writes sharing a coalesce key (e.g. one job's state row) collapse to the
      latest, so a burst of status changes costs one write.
    * A write that finally fails is logged; the in-memory job stays authoritative
      and the next update rewrites the row, so nothing is corrupted.
    * `synchronous=True` (tests / inline mode) executes immediately.
    """

    BACKOFF_SECONDS = (5, 15, 30, 60, 90)

    def __init__(self, synchronous: bool = False):
        import threading
        from collections import OrderedDict
        self.synchronous = synchronous
        self._pending: "OrderedDict[str, object]" = OrderedDict()
        self._cond = threading.Condition()
        self._seq = 0
        self.failures = 0
        if not synchronous:
            threading.Thread(target=self._run, name="sheets-writer", daemon=True).start()

    def submit(self, fn, coalesce_key: str | None = None) -> bool:
        if self.synchronous:
            return self._execute(fn, retry=False)
        with self._cond:
            if coalesce_key is None:
                self._seq += 1
                coalesce_key = f"__seq_{self._seq}"
            self._pending[coalesce_key] = fn
            self._pending.move_to_end(coalesce_key)
            self._cond.notify()
        return True

    def pending(self) -> int:
        with self._cond:
            return len(self._pending)

    def _run(self):
        while True:
            with self._cond:
                while not self._pending:
                    self._cond.wait()
                _, fn = self._pending.popitem(last=False)
            self._execute(fn, retry=True)

    def _execute(self, fn, retry: bool) -> bool:
        attempts = self.BACKOFF_SECONDS if retry else ()
        for wait in (0, *attempts):
            if wait:
                time.sleep(wait)
            try:
                result = fn()
                return bool(result) if result is not None else True
            except GoogleSheetsNotConfiguredError:
                return False
            except Exception as exc:
                quota = "429" in str(exc) or "Quota exceeded" in str(exc)
                if not (retry and quota):
                    self.failures += 1
                    logger.warning("Sheets write failed (%s); in-memory state kept.", type(exc).__name__)
                    return False
                logger.info("Sheets quota hit; retrying write after backoff.")
        self.failures += 1
        logger.warning("Sheets write abandoned after quota backoff; in-memory state kept.")
        return False


def submit_write(sheets_service, fn, coalesce_key: str | None = None) -> bool:
    queue = getattr(sheets_service, "write_queue", None)
    if queue is None:
        try:
            result = fn()
            return bool(result) if result is not None else True
        except GoogleSheetsNotConfiguredError:
            return False
        except Exception as exc:
            logger.warning("Sheets write failed (%s).", type(exc).__name__)
            return False
    return queue.submit(fn, coalesce_key)


class GoogleSheetsService:
    def __init__(self, settings: Settings, cache_ttl_seconds: int = 180):
        self.settings = settings
        self.cache_ttl_seconds = cache_ttl_seconds
        self._client = None
        self._cache: dict[tuple[str, str], CachedRecords] = {}
        self._disabled_reason: str | None = None
        # Each open_by_key()/worksheet() is a quota-counted READ; handles, headers
        # and row positions are cached so steady-state updates cost no reads.
        self._worksheets: dict[tuple[str, str], Any] = {}
        self._headers: dict[tuple[str, str], list[str]] = {}
        self._row_index: dict[tuple[str, str, str, str], int] = {}
        self.write_queue: SheetsWriteQueue | None = None

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

        key = (spreadsheet_id, worksheet_name)
        cached = self._worksheets.get(key)
        if cached is not None:
            return cached
        client = self._authorize()
        worksheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)
        self._worksheets[key] = worksheet
        return worksheet

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

        try:
            worksheet = self._worksheet(spreadsheet_id, worksheet_name)
            records = worksheet.get_all_records()
            self._cache[key] = CachedRecords(time.time(), records)
            return [record.copy() for record in records]
        except Exception as exc:
            if cached:
                logger.warning("Google Sheets read failed (%s); using cached records.", exc)
                return [record.copy() for record in cached.records]
            raise

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
        safe_values = [safe_cell_value(v) for v in values]
        worksheet.append_row(safe_values)
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
        sheet_key = (spreadsheet_id or "", worksheet_name)
        row_key = (spreadsheet_id or "", worksheet_name, key_column, key_value.lower())
        headers = self._headers.get(sheet_key)
        cached_row = self._row_index.get(row_key)
        if headers and cached_row:
            cells = [{"range": rowcol_to_a1(cached_row, headers.index(c) + 1), "values": [[safe_cell_value(v)]]}
                     for c, v in updates.items() if c in headers]
            if cells:
                worksheet.batch_update(cells, value_input_option=ValueInputOption.raw)
            self._cache.pop((spreadsheet_id or "", worksheet_name), None)
            return True

        rows = worksheet.get_all_values()
        if not rows:
            return False

        headers = rows[0]
        self._headers[sheet_key] = headers
        try:
            key_index = headers.index(key_column)
        except ValueError:
            return False

        for row_number, row in enumerate(rows[1:], start=2):
            if key_index < len(row) and row[key_index].strip().lower() == key_value.lower():
                self._row_index[row_key] = row_number
                # One batch request per row update: per-cell writes exceed the
                # Sheets "write requests per minute" quota within a single job.
                cells = [
                    {
                        "range": rowcol_to_a1(row_number, headers.index(column_name) + 1),
                        "values": [[safe_cell_value(value)]],
                    }
                    for column_name, value in updates.items()
                    if column_name in headers
                ]
                if cells:
                    # RAW: user-controlled text (filenames, errors) must never be
                    # interpreted as a Sheets formula.
                    worksheet.batch_update(cells, value_input_option=ValueInputOption.raw)
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
