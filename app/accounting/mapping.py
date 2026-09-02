import re
from typing import Any

from app.accounting.purposes import MEMBER_RECEIPT, VENDOR_INVOICE
from app.accounting.schemas import MappingMissingItem, MappingResult
from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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
        if not source_value:
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

        if purpose == MEMBER_RECEIPT:
            bank = extracted_data.get("bank_name_or_code")
            bank_code = self.lookup(purpose, "BANK", bank)
            if bank and bank_code:
                mapped["bank_name_or_code"] = bank_code
            elif bank:
                missing.append(MappingMissingItem(type="BANK", source_value=str(bank)))

            bill_head = extracted_data.get("bill_head")
            bill_head_code = self.lookup(purpose, "BILL_HEAD", bill_head)
            if bill_head and bill_head_code:
                mapped["bill_head"] = bill_head_code
            elif bill_head:
                missing.append(MappingMissingItem(type="BILL_HEAD", source_value=str(bill_head)))

            tower_flat = " ".join(
                item for item in [str(extracted_data.get("tower") or ""), str(extracted_data.get("flat") or "")]
                if item
            )
            flat_code = self.lookup(purpose, "TOWER_FLAT", tower_flat)
            if tower_flat and flat_code:
                mapped["flat"] = flat_code

        if purpose == VENDOR_INVOICE:
            vendor_code = extracted_data.get("vendor_code")
            vendor_name = extracted_data.get("vendor_name")
            if not vendor_code and vendor_name:
                vendor_code = self.lookup(purpose, "VENDOR", vendor_name)
                if vendor_code:
                    mapped["vendor_code"] = vendor_code
                else:
                    missing.append(MappingMissingItem(type="VENDOR", source_value=str(vendor_name)))

            mapped_expenses = []
            for expense in extracted_data.get("expenses", []):
                expense = dict(expense)
                if not expense.get("expense_code") and expense.get("expense_description"):
                    code = self.lookup(
                        purpose, "EXPENSE", str(expense["expense_description"])
                    )
                    if code:
                        expense["expense_code"] = code
                    else:
                        missing.append(
                            MappingMissingItem(
                                type="EXPENSE",
                                source_value=str(expense["expense_description"]),
                            )
                        )
                mapped_expenses.append(expense)
            mapped["expenses"] = mapped_expenses

        return MappingResult(
            status="NEEDS_MAPPING" if missing else "MAPPED",
            mapped_data=mapped,
            missing=missing,
        )
