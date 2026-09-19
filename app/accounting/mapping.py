import re
from typing import Any

from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.schemas import MappingMissingItem, MappingResult
from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _is_empty_or_dash(val: Any) -> bool:
    if val is None:
        return True
    text = str(val).strip()
    return text in {"", "-", "null", "none", "n/a", "unknown"}


class MappingMasterService:
    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def _records(self) -> list[dict[str, Any]]:
        try:
            return self.sheets_service.read_table("mapping_master")
        except GoogleSheetsNotConfiguredError:
            return []

    def lookup(self, purpose: str, mapping_type: str, source_value: str | None) -> str | None:
        if _is_empty_or_dash(source_value):
            return None
        purpose = purpose.upper()
        mapping_type = mapping_type.upper()
        source_norm = normalize(source_value)

        exact_match = None
        alias_match = None
        for record in self._records():
            if str(record.get("Purpose", "")).strip().upper() != purpose:
                continue
            record_type = record.get("Mapping Type", record.get("Type", ""))
            if str(record_type).strip().upper() != mapping_type:
                continue
            if str(record.get("Active", "true")).strip().lower() not in {"true", "1", "yes", "active"}:
                continue

            configured_source = str(record.get("Source Value", "")).strip()
            target = str(
                record.get("Canonical Code", record.get("Target Value", ""))
            ).strip()
            aliases = [
                item.strip()
                for item in str(record.get("Approved Alias", "") or record.get("Alias", "")).split("|")
                if item.strip()
            ]

            if configured_source == source_value and target:
                exact_match = target
                break
            if normalize(configured_source) == source_norm and target:
                exact_match = target
            if any(normalize(alias) == source_norm for alias in aliases) and target:
                alias_match = target

        return exact_match or alias_match

    def map_data(self, purpose: str, extracted_data: dict[str, Any]) -> MappingResult:
        """Map bank / bill head (and vendor) values to NBH codes via Mapping_Master.

        * Rows are addressed by _row_id; reviewer-edited fields are never changed.
        * An unmapped value is reported (mapping_missing) and the SOURCE value is
          exported. A missing mapping never blocks approval.
        """
        import copy

        purpose = purpose.upper()
        mapped = copy.deepcopy(extracted_data)
        missing: list[MappingMissingItem] = []

        def note_missing(kind: str, value: str, row_id: str | None):
            for item in missing:
                if item.type == kind and item.source_value == value:
                    if row_id:
                        item.rows.append(len(item.rows) + 1)
                    return
            missing.append(MappingMissingItem(type=kind, source_value=value, rows=[1] if row_id else [],
                                              reason="no active Mapping_Master entry; source value exported"))

        for row in mapped.get("rows") or []:
            if not isinstance(row, dict):
                continue
            edited = set(row.get("_edited_fields") or [])
            for column, kind in (("Society Bank Name/Bank code(Given to you by nobrokerhood)*", "BANK"),
                                 ("Bill Head*", "BILL_HEAD")):
                if column in edited:
                    continue
                value = row.get(column)
                if _is_empty_or_dash(value):
                    continue
                code = self.lookup(purpose, kind, value)
                if code:
                    row[column] = code
                else:
                    note_missing(kind, str(value), row.get("_row_id"))

        if purpose == VENDOR_INVOICE:
            detail = mapped.get("vendor_detail") or {}
            name = detail.get("vendor_name") or mapped.get("vendor_name")
            if not _is_empty_or_dash(name):
                code = self.lookup(purpose, "VENDOR", name)
                if code:
                    detail["vendor_code"] = code
                    mapped["vendor_detail"] = detail
                else:
                    note_missing("VENDOR", str(name), None)

        mapped["mapping_missing"] = [m.model_dump() for m in missing]
        return MappingResult(status="NEEDS_MAPPING" if missing else "MAPPED", mapped_data=mapped, missing=missing)
