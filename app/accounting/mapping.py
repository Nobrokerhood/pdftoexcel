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
        purpose = purpose.upper()
        mapped = dict(extracted_data)
        missing: list[MappingMissingItem] = []

        if purpose in {MEMBER_RECEIPT, PETTY_CASH_REGISTER, "BANK_STATEMENT", "SOCIETY_MEMBER_LEDGER"}:
            # Handle multi-row mapping
            rows = mapped.get("rows")
            if isinstance(rows, list) and len(rows) > 0:
                mapped_rows = []
                for idx, row in enumerate(rows, start=1):
                    row_copy = dict(row)
                    bank = row_copy.get("Society Bank Name/Bank code(Given to you by nobrokerhood)*")
                    if not _is_empty_or_dash(bank):
                        bank_code = self.lookup(purpose, "BANK", bank)
                        if bank_code:
                            row_copy["Society Bank Name/Bank code(Given to you by nobrokerhood)*"] = bank_code
                        elif not any(m.type == "BANK" and m.source_value == str(bank) for m in missing):
                            missing.append(MappingMissingItem(type="BANK", source_value=str(bank), rows=[idx]))

                    bill_head = row_copy.get("Bill Head*")
                    if not _is_empty_or_dash(bill_head):
                        bill_head_code = self.lookup(purpose, "BILL_HEAD", bill_head)
                        if bill_head_code:
                            row_copy["Bill Head*"] = bill_head_code
                        elif not any(m.type == "BILL_HEAD" and m.source_value == str(bill_head) for m in missing):
                            missing.append(MappingMissingItem(type="BILL_HEAD", source_value=str(bill_head), rows=[idx]))

                    mapped_rows.append(row_copy)
                mapped["rows"] = mapped_rows

            # Also check top-level fields for single-record receipts
            bank = extracted_data.get("bank_name_or_code")
            if not _is_empty_or_dash(bank):
                bank_code = self.lookup(purpose, "BANK", bank)
                if bank_code:
                    mapped["bank_name_or_code"] = bank_code
                elif not any(m.type == "BANK" and m.source_value == str(bank) for m in missing):
                    missing.append(MappingMissingItem(type="BANK", source_value=str(bank)))

            bill_head = extracted_data.get("bill_head")
            if not _is_empty_or_dash(bill_head):
                bill_head_code = self.lookup(purpose, "BILL_HEAD", bill_head)
                if bill_head_code:
                    mapped["bill_head"] = bill_head_code
                elif not any(m.type == "BILL_HEAD" and m.source_value == str(bill_head) for m in missing):
                    missing.append(MappingMissingItem(type="BILL_HEAD", source_value=str(bill_head)))

            tower = str(extracted_data.get("tower") or "").strip()
            flat = str(extracted_data.get("flat") or "").strip()
            tower_flat = f"{tower} {flat}".strip()
            if not _is_empty_or_dash(tower_flat) and tower_flat != "-":
                flat_code = self.lookup(purpose, "TOWER_FLAT", tower_flat)
                if flat_code:
                    mapped["flat"] = flat_code

        elif purpose == VENDOR_INVOICE:
            vendor_code = extracted_data.get("vendor_code")
            vendor_name = extracted_data.get("vendor_name")
            if not _is_empty_or_dash(vendor_name) and _is_empty_or_dash(vendor_code):
                v_code = self.lookup(purpose, "VENDOR", vendor_name)
                if v_code:
                    mapped["vendor_code"] = v_code
                else:
                    missing.append(MappingMissingItem(type="VENDOR", source_value=str(vendor_name)))

            mapped_expenses = []
            for expense in extracted_data.get("expenses", []):
                expense = dict(expense)
                desc = expense.get("expense_description")
                if _is_empty_or_dash(expense.get("expense_code")) and not _is_empty_or_dash(desc):
                    code = self.lookup(purpose, "EXPENSE", str(desc))
                    if code:
                        expense["expense_code"] = code
                    else:
                        missing.append(
                            MappingMissingItem(
                                type="EXPENSE",
                                source_value=str(desc),
                            )
                        )
                mapped_expenses.append(expense)
            mapped["expenses"] = mapped_expenses

        return MappingResult(
            status="NEEDS_MAPPING" if missing else "MAPPED",
            mapped_data=mapped,
            missing=missing,
        )
