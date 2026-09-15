import io
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.templates import NBH_IMPORT_COLUMNS, TemplateDefinition


class OutputGenerationError(RuntimeError):
    pass


# Day-first formats, as written on Indian receipts and invoices.
DATE_FORMATS = (
    "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y", "%d-%m-%Y", "%d/%m/%Y",
    "%d.%m.%Y", "%Y-%m-%d", "%d-%b-%y", "%d/%m/%y",
)

NBH_FIELD_SYNONYMS = {
    "Payment Type*": ("Payment Type*", "payment_type", "Payment Type", "payment_mode"),
    "Society Bank Name/Bank code(Given to you by nobrokerhood)*": (
        "Society Bank Name/Bank code(Given to you by nobrokerhood)*",
        "bank_name_or_code",
        "Society Bank Name/Bank code",
        "bank_code",
        "bank_name",
    ),
    "Cheque/Ref No*": ("Cheque/Ref No*", "reference_number", "Cheque/Ref No", "ref_no", "cheque_no", "voucher_no"),
    "Tower No*": ("Tower No*", "tower", "Tower No", "wing", "building"),
    "Flat No*": ("Flat No*", "flat", "Flat No", "unit_no", "unit", "flat_no"),
    "Bill Head*": ("Bill Head*", "bill_head", "Bill Head", "category", "charge_head"),
    "Amount*": ("Amount*", "amount", "Amount", "expense_amount"),
    "Transaction Date*": ("Transaction Date*", "transaction_date", "Transaction Date", "date", "bill_date"),
    "Comments": ("Comments", "comments", "narration", "remarks", "description"),
    "Meter No": ("Meter No", "meter_number", "meter_no"),
    "Cheque Issuer Bank": ("Cheque Issuer Bank", "cheque_issuer_bank", "issuer_bank"),
    "Cheque Date": ("Cheque Date", "cheque_date"),
}


def _clean_str(value) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none", "n/a", "unknown"}:
        return "-"
    return text


def _amount(value):
    # If missing or dash, return "-"
    if value is None or str(value).strip() in {"", "-", "null", "None", "N/A", "UNKNOWN"}:
        return "-"
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    cleaned = re.sub(r"[^\d.\-]", "", str(value))
    try:
        return float(Decimal(cleaned))
    except (InvalidOperation, ValueError):
        return str(value)


def _date(value):
    if value is None or str(value).strip() in {"", "-", "null", "None", "N/A", "UNKNOWN"}:
        return "-"
    if isinstance(value, (date, datetime)):
        return value
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return text


def _extract_row_field(row_dict: dict, col_name: str):
    synonyms = NBH_FIELD_SYNONYMS.get(col_name, (col_name,))
    for key in synonyms:
        if key in row_dict and row_dict[key] is not None:
            val = row_dict[key]
            if str(val).strip() not in {"", "null", "None", "N/A", "UNKNOWN"}:
                return val
    return "-"


class TemplateOutputGenerator:
    def generate_xlsx(
        self,
        purpose: str,
        template: TemplateDefinition,
        data: dict,
        job_id: str,
    ) -> tuple[str, bytes]:
        purpose = purpose.upper()
        workbook = Workbook()
        
        # Primary / Active sheet must be the NBH Import sheet
        sheet = workbook.active
        sheet.title = "NBH Accounting"[:31]

        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="0F766E", end_color="0F766E", fill_type="solid")

        if purpose in {MEMBER_RECEIPT, PETTY_CASH_REGISTER, "BANK_STATEMENT", "SOCIETY_MEMBER_LEDGER"}:
            sheet.append(list(NBH_IMPORT_COLUMNS))
            for cell in sheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center")

            rows = data.get("rows")
            if isinstance(rows, list) and len(rows) > 0:
                for row_dict in rows:
                    if not isinstance(row_dict, dict):
                        continue
                    row_cells = []
                    for col in NBH_IMPORT_COLUMNS:
                        val = _extract_row_field(row_dict, col)
                        if "amount" in col.lower():
                            row_cells.append(_amount(val))
                        elif "date" in col.lower():
                            row_cells.append(_date(val))
                        else:
                            row_cells.append(_clean_str(val))
                    sheet.append(row_cells)
            else:
                # Single receipt payload
                row_cells = []
                for col in NBH_IMPORT_COLUMNS:
                    val = _extract_row_field(data, col)
                    if "amount" in col.lower():
                        row_cells.append(_amount(val))
                    elif "date" in col.lower():
                        row_cells.append(_date(val))
                    else:
                        row_cells.append(_clean_str(val))
                sheet.append(row_cells)

        elif purpose == VENDOR_INVOICE:
            sheet.append(list(template.fields))
            for cell in sheet[1]:
                cell.font = header_font
                cell.fill = header_fill

            expenses = data.get("expenses") or [{}]
            for expense in expenses:
                sheet.append(
                    [
                        _clean_str(data.get("bill_number")),
                        _date(data.get("bill_date")),
                        _clean_str(data.get("vendor_code")),
                        _date(data.get("due_date")),
                        _clean_str(data.get("narration")),
                        _amount(data.get("cgst_amount") or 0),
                        _amount(data.get("sgst_amount") or 0),
                        _amount(data.get("igst_amount") or 0),
                        _amount(data.get("tds_amount") or 0),
                        _clean_str(expense.get("expense_code")),
                        _amount(expense.get("expense_amount")),
                    ]
                )
        else:
            # Generic fallback
            fields = list(template.fields) or list(NBH_IMPORT_COLUMNS)
            sheet.append(fields)
            rows = data.get("rows") or [data]
            for row_dict in rows:
                if isinstance(row_dict, dict):
                    sheet.append([_clean_str(row_dict.get(k, "-")) for k in fields])

        # Apply column formats
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                header_val = str(sheet.cell(row=1, column=cell.column).value).lower()
                if "date" in header_val and isinstance(cell.value, (date, datetime)):
                    cell.number_format = "dd-mmm-yyyy"
                elif "amount" in header_val and isinstance(cell.value, (int, float)):
                    cell.number_format = "#,##0.00"

        # Auto-adjust column widths
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            sheet.column_dimensions[col_letter].width = max(max_len + 4, 14)

        # Supporting sheet: Summary
        summary_sheet = workbook.create_sheet(title="Summary")
        summary_sheet.append(["NoBrokerHood Accounting AI — Execution Summary"])
        summary_sheet.append(["Job ID", job_id])
        summary_sheet.append(["Purpose", purpose])
        summary_sheet.append(["Template", template.template_name])
        summary_sheet.append(["Document Type", data.get("document_type", purpose)])
        summary_sheet.append(["Description", data.get("summary", "-")])
        total_rows = len(data.get("rows", [])) if isinstance(data.get("rows"), list) else 1
        summary_sheet.append(["Total Transactions", total_rows])
        summary_sheet.append(["Generated At", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        stream = io.BytesIO()
        workbook.save(stream)
        filename = f"{template.template_code}_{job_id}.xlsx"
        return filename, stream.getvalue()
