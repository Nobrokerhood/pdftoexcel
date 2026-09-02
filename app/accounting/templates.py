from dataclasses import dataclass
from typing import Any

from app.accounting.purposes import MEMBER_RECEIPT, VENDOR_INVOICE
from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService


class TemplateConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TemplateDefinition:
    purpose: str
    template_code: str
    template_name: str
    version: str
    output_format: str
    fields: tuple[str, ...]
    supports_multiple_expense_entries: bool = False
    active: bool = True

    def public_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "template_code": self.template_code,
            "template_name": self.template_name,
            "version": self.version,
            "output_format": self.output_format,
            "fields": list(self.fields),
            "supports_multiple_expense_entries": self.supports_multiple_expense_entries,
            "active": self.active,
        }


MEMBER_RECEIPT_TEMPLATE = TemplateDefinition(
    purpose=MEMBER_RECEIPT,
    template_code="NBH_MEMBER_RECEIPT_V1",
    template_name="NBH Member Receipt Import v1",
    version="1",
    output_format="CSV",
    fields=(
        "Payment Type",
        "Society Bank Name/Bank code",
        "Cheque/Ref No",
        "Tower No",
        "Flat No",
        "Bill Head",
        "Amount",
        "Transaction Date",
        "Comments",
        "Meter No",
        "Cheque Issuer Bank",
        "Cheque Date",
    ),
)

VENDOR_INVOICE_TEMPLATE = TemplateDefinition(
    purpose=VENDOR_INVOICE,
    template_code="NBH_VENDOR_BILL_V1",
    template_name="NBH Vendor Bill v1",
    version="1",
    output_format="XLSX",
    fields=(
        "Bill Number",
        "Bill Date",
        "Vendor Code",
        "Due Date",
        "Narration",
        "CGST Amount",
        "SGST Amount",
        "IGST Amount",
        "TDS Amount",
        "Expense Code",
        "Expense Amount",
    ),
    supports_multiple_expense_entries=True,
)


BUILT_IN_TEMPLATES = {
    MEMBER_RECEIPT: MEMBER_RECEIPT_TEMPLATE,
    VENDOR_INVOICE: VENDOR_INVOICE_TEMPLATE,
}


def _active(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "active"}


class TemplateRegistryService:
    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def _sheet_template(self, purpose: str) -> TemplateDefinition | None:
        try:
            records = self.sheets_service.read_table("template_master")
        except GoogleSheetsNotConfiguredError:
            return None

        for record in records:
            if (
                str(record.get("Purpose", "")).strip().upper() == purpose
                and _active(record.get("Active", False))
            ):
                built_in = BUILT_IN_TEMPLATES.get(purpose)
                fields = built_in.fields if built_in else ()
                return TemplateDefinition(
                    purpose=purpose,
                    template_code=str(record.get("Template Code", "")).strip(),
                    template_name=str(record.get("Template Name", "")).strip(),
                    version=str(record.get("Version", "")).strip(),
                    output_format=str(record.get("Output Format", "")).strip(),
                    fields=fields,
                    supports_multiple_expense_entries=(
                        built_in.supports_multiple_expense_entries if built_in else False
                    ),
                    active=True,
                )
        return None

    def get_active_template(self, purpose: str) -> TemplateDefinition:
        purpose = purpose.strip().upper()
        configured = self._sheet_template(purpose)
        if configured:
            return configured

        built_in = BUILT_IN_TEMPLATES.get(purpose)
        if built_in and built_in.active:
            return built_in

        raise TemplateConfigurationError("TEMPLATE_CONFIGURATION_MISSING")
