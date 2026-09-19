"""XLSX generation.

Primary sheet: EXACTLY the 12 NBH import columns, in order, for every purpose.
Output policy:
* Amount*: a number, only when `parse_amount` reads it unambiguously.
* Transaction Date* / Cheque Date: a real date formatted DD-MM-YYYY, or "-".
* Every other cell: text; a missing value is "-".
* Text cells are always stored as strings, so source text beginning with
  = + - @ can never execute as a formula (formula injection).
* Rows come only from `rows`. There is no path that builds a transaction from
  document-level fields; zero rows cannot reach here (approval blocks them).
Supporting sheets carry evidence, reconciliation, the candidate ledger and the
review audit trail.
"""

import io
from datetime import datetime

from openpyxl import Workbook
from openpyxl.cell.cell import TYPE_STRING
from openpyxl.styles import Alignment, Font, PatternFill

from app.accounting.dates import EXCEL_DATE_FORMAT, parse_date
from app.accounting.money import parse_amount
from app.accounting.templates import NBH_IMPORT_COLUMNS, TemplateDefinition

AMOUNT_COLUMNS = {"Amount*"}
DATE_COLUMNS = {"Transaction Date*", "Cheque Date"}


class OutputGenerationError(RuntimeError):
    pass


def _text(value) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text and text.lower() not in {"null", "none", "n/a", "unknown", "undefined", "nan"} else "-"


def _set_text(cell, value):
    cell.value = _text(value)
    cell.data_type = TYPE_STRING  # never a formula, whatever the text starts with


def _write_row(sheet, values: list):
    sheet.append([None] * len(values))
    r = sheet.max_row
    for c, value in enumerate(values, start=1):
        cell = sheet.cell(row=r, column=c)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cell.value = value
        else:
            _set_text(cell, value)


def nbh_cells(row: dict, row_label: str) -> list:
    """The 12 exported cells for one row, or OutputGenerationError."""
    cells = []
    for col in NBH_IMPORT_COLUMNS:
        value = row.get(col)
        if col in AMOUNT_COLUMNS:
            reading = parse_amount(value)
            if not reading.found:
                raise OutputGenerationError(f"{row_label}: amount '{value}' is not a readable amount ({reading.reason}).")
            cells.append(("amount", float(reading.value)))
        elif col in DATE_COLUMNS:
            reading = parse_date(value)
            if reading.found:
                cells.append(("date", reading.value))
            elif reading.status == "MISSING" and col != "Transaction Date*":
                cells.append(("text", "-"))
            else:
                raise OutputGenerationError(f"{row_label}: {col} '{value}' is not a valid date ({reading.reason}).")
        else:
            cells.append(("text", _text(value)))
    return cells


class TemplateOutputGenerator:
    def generate_xlsx(self, purpose: str, template: TemplateDefinition, data: dict, job_id: str,
                      audit: list[dict] | None = None) -> tuple[str, bytes]:
        rows = [r for r in (data.get("rows") or []) if isinstance(r, dict)]
        if not rows:
            raise OutputGenerationError("There are no transaction rows to export.")

        wb = Workbook()
        sheet = wb.active
        sheet.title = "NBH Accounting"
        sheet.append(list(NBH_IMPORT_COLUMNS))
        header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="0F766E", end_color="0F766E", fill_type="solid")
        for cell in sheet[1]:
            cell.font, cell.fill = header_font, header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for index, row in enumerate(rows, start=1):
            cells = nbh_cells(row, row.get("_row_id") or f"row {index}")
            sheet.append([None] * len(cells))
            r = sheet.max_row
            for c, (kind, value) in enumerate(cells, start=1):
                cell = sheet.cell(row=r, column=c)
                if kind == "amount":
                    cell.value = value
                    cell.number_format = "0.00"
                elif kind == "date":
                    cell.value = value
                    cell.number_format = EXCEL_DATE_FORMAT
                else:
                    _set_text(cell, value)
        for col in sheet.columns:
            width = max(len(str(c.value or "")) for c in col)
            sheet.column_dimensions[col[0].column_letter].width = min(max(width + 4, 14), 60)

        self._evidence_sheet(wb, data, rows)
        self._reconciliation_sheet(wb, data)
        self._ledger_sheet(wb, data)
        self._audit_sheet(wb, audit or [])
        if data.get("vendor_detail"):
            self._vendor_sheet(wb, data["vendor_detail"])
        self._summary_sheet(wb, purpose, template, data, job_id, rows)

        stream = io.BytesIO()
        wb.save(stream)
        return f"{template.template_code}_{job_id}.xlsx", stream.getvalue()

    # -- supporting sheets ------------------------------------------------------
    def _evidence_sheet(self, wb, data, rows):
        ws = wb.create_sheet("Evidence")
        ws.append(["Row ID", "Row status", "Page", "Column", "Final value", "Field status", "Reason", "Sources"])
        evidence = data.get("row_evidence") or {}
        for row in rows:
            rid = row.get("_row_id", "")
            ev = evidence.get(rid) or {}
            fields = ev.get("fields") or {}
            if not fields:
                _write_row(ws, [rid, row.get("_status", ""), ev.get("page") or "-", "-", "-", "-",
                                "; ".join(ev.get("reasons") or []) or "no field evidence", "-"])
            for col, d in fields.items():
                sources = "; ".join(f"{v.get('source')}={v.get('raw')}" for v in d.get("votes") or [])
                _write_row(ws, [rid, row.get("_status", ""), ev.get("page") or "-", col, d.get("value", "-"),
                                d.get("status", "-"), d.get("reason", "-"), sources or "-"])

    def _reconciliation_sheet(self, wb, data):
        ws = wb.create_sheet("Reconciliation")
        ws.append(["Check", "Source-written value", "Calculated value", "Difference", "Status", "Note"])
        for check in (data.get("reconciliation") or {}).get("checks") or []:
            _write_row(ws, [check.get("label"), check.get("source_value"), check.get("calculated_value"),
                            check.get("difference"), check.get("status"), check.get("note") or "-"])

    def _ledger_sheet(self, wb, data):
        ws = wb.create_sheet("Source Candidates")
        ledger = data.get("candidate_ledger") or {}
        _write_row(ws, ["Equation", ledger.get("equation", "-")])
        _write_row(ws, ["Balanced", str(ledger.get("balanced", "-"))])
        ws.append(["Candidate", "Page", "Classification", "Status", "Reason", "Row ID", "Text"])
        for cand in ledger.get("candidates") or []:
            text = " | ".join(f.get("text", "") for items in (cand.get("fields") or {}).values() for f in items[:1])
            _write_row(ws, [cand.get("candidate_id"), cand.get("page"), cand.get("classification"), cand.get("status"),
                            cand.get("status_reason"), cand.get("row_id") or "-", text[:300] or "-"])

    def _audit_sheet(self, wb, audit):
        ws = wb.create_sheet("Review Audit")
        ws.append(["Time", "User", "Action", "Row ID", "Field", "Before", "After", "Issue", "Reason"])
        for entry in audit:
            _write_row(ws, [entry.get("timestamp"), entry.get("user_email"), entry.get("action"), entry.get("row_id"),
                            entry.get("field"), str(entry.get("old_value")), str(entry.get("new_value")),
                            entry.get("issue"), entry.get("reason")])

    def _vendor_sheet(self, wb, detail):
        ws = wb.create_sheet("Vendor Detail")
        for key, value in detail.items():
            if key == "line_items":
                continue
            _write_row(ws, [key, value])
        ws.append([])
        ws.append(["Description", "Quantity", "Rate", "Amount"])
        for item in detail.get("line_items") or []:
            _write_row(ws, [item.get("description"), item.get("quantity"), item.get("rate"), item.get("amount")])

    def _summary_sheet(self, wb, purpose, template, data, job_id, rows):
        ws = wb.create_sheet("Summary")
        for label, value in (
            ("Job ID", job_id), ("Purpose", purpose), ("Template", template.template_name),
            ("Document type", data.get("document_type", purpose)),
            ("Extraction provider", data.get("_extraction_provider", "-")),
            ("Transactions exported", len(rows)),
            ("Reconciliation", (data.get("reconciliation") or {}).get("overall_status", "-")),
            ("Generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ):
            _write_row(ws, [label, value])
