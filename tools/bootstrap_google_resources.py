import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.google.drive_service import GoogleDriveService
from app.google.sheet_schemas import SHEET_SCHEMAS
from app.google.sheets_service import (
    GoogleSheetsError,
    GoogleSheetsNotConfiguredError,
    GoogleSheetsService,
)


@dataclass
class CheckResult:
    resource: str
    status: str
    detail: str = ""


def status_line(result: CheckResult) -> str:
    detail = f" - {result.detail}" if result.detail else ""
    return f"{result.status:<10} {result.resource}{detail}"


def _has_service_account(settings) -> bool:
    return bool(settings.google_service_account_json or settings.google_service_account_file)


def mock_results(settings) -> list[CheckResult]:
    results = [
        CheckResult("Service Account", "READY" if _has_service_account(settings) else "MISSING"),
        CheckResult(
            "Spreadsheet",
            "READY" if settings.google_accounting_spreadsheet_id else "MISSING",
            "GOOGLE_ACCOUNTING_SPREADSHEET_ID",
        ),
    ]
    for schema in SHEET_SCHEMAS.values():
        spreadsheet_id, tab_name = GoogleSheetsService(settings).resolve_table(schema.key)
        status = "READY" if spreadsheet_id else "MISSING"
        results.append(CheckResult(tab_name, status, schema.key))
    if settings.google_drive_root_folder_id:
        results.append(CheckResult("Drive Root", "READY"))
    else:
        results.append(CheckResult("Drive Root", "MISSING", "GOOGLE_DRIVE_ROOT_FOLDER_ID"))
    return results


def check_or_bootstrap_tables(sheets: GoogleSheetsService, bootstrap: bool) -> list[CheckResult]:
    results = []
    for schema in SHEET_SCHEMAS.values():
        try:
            if bootstrap:
                changed, status = sheets.ensure_table(schema.key)
                results.append(
                    CheckResult(schema.tab_name, "PASS" if status != "MISMATCH" else "MISMATCH", status)
                )
                continue

            headers = sheets.table_headers(schema.key)
            if not headers:
                results.append(CheckResult(schema.tab_name, "MISSING", "Header row missing"))
            elif headers == list(schema.headers):
                results.append(CheckResult(schema.tab_name, "PASS"))
            else:
                results.append(CheckResult(schema.tab_name, "MISMATCH", "Header row differs from schema"))
        except GoogleSheetsNotConfiguredError:
            results.append(CheckResult(schema.tab_name, "MISSING", "Spreadsheet ID not configured"))
        except WorksheetNotFound:
            results.append(CheckResult(schema.tab_name, "MISSING", "Tab missing"))
        except (SpreadsheetNotFound, APIError, GoogleSheetsError) as exc:
            results.append(CheckResult(schema.tab_name, "NO_ACCESS", exc.__class__.__name__))
    return results


def check_resources(bootstrap: bool, create_folders: bool, mock: bool) -> list[CheckResult]:
    settings = get_settings()
    if mock:
        return mock_results(settings)

    sheets = GoogleSheetsService(settings, cache_ttl_seconds=0)
    drive = GoogleDriveService(settings)
    results = [
        CheckResult("Service Account", "READY" if _has_service_account(settings) else "MISSING"),
    ]

    try:
        sheets.spreadsheet()
        results.append(CheckResult("Spreadsheet", "PASS"))
    except GoogleSheetsNotConfiguredError:
        results.append(CheckResult("Spreadsheet", "MISSING", "GOOGLE_ACCOUNTING_SPREADSHEET_ID"))
    except (SpreadsheetNotFound, APIError, GoogleSheetsError) as exc:
        results.append(CheckResult("Spreadsheet", "NO_ACCESS", exc.__class__.__name__))

    results.extend(check_or_bootstrap_tables(sheets, bootstrap))

    if settings.google_drive_root_folder_id:
        try:
            drive.validate_folder_id(settings.google_drive_root_folder_id)
            results.append(CheckResult("Drive Root", "PASS"))
        except Exception as exc:
            results.append(CheckResult("Drive Root", "NO_ACCESS", exc.__class__.__name__))
    elif bootstrap and create_folders:
        try:
            drive.create_folder("Accounting AI")
            results.append(CheckResult("Drive Root", "PASS", "Created Accounting AI folder"))
        except Exception as exc:
            results.append(CheckResult("Drive Root", "NO_ACCESS", exc.__class__.__name__))
    else:
        results.append(CheckResult("Drive Root", "MISSING", "GOOGLE_DRIVE_ROOT_FOLDER_ID"))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or bootstrap Accounting AI Google resources.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check resources without modifying Google.")
    mode.add_argument("--bootstrap", action="store_true", help="Create missing tabs/header rows only.")
    parser.add_argument("--create-folders", action="store_true", help="Create the Drive root folder when bootstrapping.")
    parser.add_argument("--mock", action="store_true", help="Run checks without contacting Google.")
    args = parser.parse_args(argv)

    bootstrap = args.bootstrap
    results = check_resources(bootstrap, args.create_folders, args.mock)
    for result in results:
        print(status_line(result))

    blocking = {"MISSING", "MISMATCH", "NO_ACCESS"}
    return 1 if any(result.status in blocking for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
