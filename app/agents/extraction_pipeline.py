"""Document extraction: OCR evidence -> source candidates -> Gemini (purpose
contract) -> fusion -> focused visual arbitration -> decided rows + ledger.

Gemini is shown the SAME canonical page images the OCR engines read, together
with every primary OCR line (alias id, full box, page size, engine,
confidence), so its row citations map exactly onto OCR geometry.

Failure behaviour (never fabricated, never silently empty):
* Gemini unavailable (quota, timeout, 5xx, auth): rows are built from OCR
  candidates, all NEEDS_REVIEW; provider recorded as LOCAL_OCR with the reason.
* Gemini response breaks the purpose contract: same, with the contract problem
  recorded as EXTRACTION_CONTRACT_VIOLATION.
* Arbitration failure: decisions stay as they were (unresolved stays NEEDS_REVIEW).
"""

import io
import json
import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from app.accounting.fusion import AMOUNT_COL, DATE_COL, REF_COL, DecidedRow
from app.accounting.purposes import MEMBER_RECEIPT, PETTY_CASH_REGISTER, VENDOR_INVOICE
from app.accounting.row_assembly import _blank_values, _clean, assemble_document
from app.accounting.source_candidates import build_candidates
from app.agents.gemini_contract import ExtractionContractError, normalize_response, prompt_for
from app.documents.ingestion import build_manifest
from app.documents.ocr import DocumentOcrService, DocumentRepresentation

logger = logging.getLogger(__name__)

ARBITRATION_BATCH = 10
MAX_ARBITRATION_CALLS = 4
MAX_OCR_LINES_IN_PROMPT = 1500
# Reasoning budget for the full-page read; with 0 the handwritten page read was row-shifted in live runs.
EXTRACTION_THINKING_BUDGET = int(__import__('os').getenv('GEMINI_EXTRACTION_THINKING_BUDGET', '4096'))


@dataclass
class CallTrace:
    calls: list[dict] = field(default_factory=list)

    def record(self, purpose: str, started: float, status: str, detail: str = "", images: int = 0):
        self.calls.append({"purpose": purpose, "status": status, "duration_ms": int((time.perf_counter() - started) * 1000),
                           "images": images, "detail": detail[:200]})


def ocr_prompt_block(rep: DocumentRepresentation) -> tuple[str, dict[str, str]]:
    """Primary OCR lines with short aliases, full boxes and page sizes."""
    alias_to_line: dict[str, str] = {}
    parts = []
    n = 0
    for page in rep.pages:
        parts.append(f"--- PAGE {page.page_number} (image {page.width}x{page.height} px, source {page.engine}, "
                     f"geometry {page.geometry}) ---")
        for line in sorted(page.lines, key=lambda l: (l.bbox[1], l.bbox[0])):
            n += 1
            if n > MAX_OCR_LINES_IN_PROMPT:
                break
            alias = f"L{n}"
            alias_to_line[alias] = line.line_id
            x0, y0, x1, y1 = line.bbox
            parts.append(f"{alias} [x={x0}-{x1} y={y0}-{y1} conf={line.confidence:.2f}] {line.text}")
    return "\n".join(parts), alias_to_line


class GeminiArbiter:
    """Focused visual arbitration on row crops. Bounded calls; ACCEPT/CORRECT/ABSTAIN."""

    def __init__(self, gemini_client, rep: DocumentRepresentation, trace: CallTrace, columns=None, layouts=None):
        self.client = gemini_client
        self.rep = rep
        self.trace = trace
        self.columns = columns or (REF_COL, DATE_COL, AMOUNT_COL)
        self.layouts = {l.page: l for l in (layouts or [])}

    def _cell_span(self, row: DecidedRow, column: str) -> tuple[float, float] | None:
        layout = self.layouts.get(row.page)
        if layout is None or not layout.tabular:
            return None
        if column == REF_COL:
            return layout.ref_col
        if column == DATE_COL:
            return layout.date_col
        if column == AMOUNT_COL and layout.amount_cols:
            idx = layout.receipt_col_index if row.kind == "INFLOW" else layout.payment_col_index
            if idx is None:
                return None
            return layout.amount_cols[idx]
        return None

    def _cell(self, row: DecidedRow, column: str) -> Image.Image | None:
        """One cell of the row (column span x band), upscaled: reading a single
        cell removes neighbouring-column confusion (serial fused into voucher)."""
        span = self._cell_span(row, column)
        if span is None or not row.bbox or not row.page:
            return None
        page = self.rep.page_image(row.page)
        if page is None:
            return None
        x0, x1 = span
        pad_x = max(14, (x1 - x0) * 0.12)
        y0, y1 = row.bbox[1], row.bbox[3]
        pad_y = max(8, int((y1 - y0) * 0.15))
        cell = page.crop((max(0, int(x0 - pad_x)), max(0, y0 - pad_y), min(page.width, int(x1 + pad_x)),
                          min(page.height, y1 + pad_y)))
        scale = 2 if cell.height < 200 else 1
        if scale > 1:
            cell = cell.resize((cell.width * scale, cell.height * scale), Image.Resampling.LANCZOS)
        return cell

    def cell_reads(self, rows: list[DecidedRow]) -> tuple[dict, list[str]]:
        """Blind per-cell reads for tabular rows: {row_id: {column: text|None}}."""
        items = []
        for row in rows:
            for column in self.columns:
                img = self._cell(row, column)
                if img is not None:
                    items.append((row.row_id, column, img))
        reads: dict = {}
        failures: list[str] = []
        per_call = 24
        chunks = [items[i:i + per_call] for i in range(0, len(items), per_call)]
        for call_no, chunk in enumerate(chunks[:MAX_ARBITRATION_CALLS + 2]):
            entries = [{"image": i + 1, "id": f"{rid}|{self.KEYS[col]}", "field": self.KEYS[col]}
                       for i, (rid, col, _) in enumerate(chunk)]
            prompt = (
                "Each image is ONE cell cut from a handwritten or printed accounting register.\n"
                "field tells you what the cell holds: ref = a voucher/reference number (digits only);\n"
                "date = a date, written day-month-year; amount = a money amount.\n"
                "Transcribe exactly what is written in the cell. A crossed 7 is 7, not 1. '/-' after an amount\n"
                "is not a digit. Ignore ruled lines and anything cut off at the image edge. Do not guess: if the\n"
                "cell is not legible with certainty, return null for it.\n"
                'Return compact JSON only: {"cells":[{"id":"...","text":"..."}]}\n'
                f"CELLS (image numbers are 1-based, in order): {json.dumps(entries)}"
            )
            started = time.perf_counter()
            try:
                raw = self.client.generate_json([prompt, *[img for _, _, img in chunk]], retries=3,
                                                purpose="arbitration")
                self.trace.record("cell_arbitration", started, "SUCCESS", f"call {call_no + 1}", len(chunk))
            except Exception as exc:
                self.trace.record("cell_arbitration", started, "FAILED", f"call {call_no + 1}: {type(exc).__name__}",
                                  len(chunk))
                failures.append(f"cell call {call_no + 1}: {type(exc).__name__}")
                continue
            valid = {e["id"]: (rid, col) for e, (rid, col, _) in zip(entries, chunk)}
            cells = raw.get("cells") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
            for item in cells or []:
                if not isinstance(item, dict) or item.get("id") not in valid:
                    continue
                rid, col = valid[item["id"]]
                text = item.get("text")
                reads.setdefault(rid, {})[col] = None if isinstance(text, (dict, list)) else text
        return reads, failures

    def _crop(self, row: DecidedRow) -> Image.Image | None:
        if not row.bbox or not row.page:
            return None
        page = self.rep.page_image(row.page)
        if page is None:
            return None
        x0, y0, x1, y1 = row.bbox
        pad = max(12, int((y1 - y0) * 0.35))
        crop = page.crop((0, max(0, y0 - pad), page.width, min(page.height, y1 + pad)))
        if crop.width > 1800:
            scale = 1800 / crop.width
            crop = crop.resize((1800, max(1, int(crop.height * scale))), Image.Resampling.LANCZOS)
        return crop

    # Short JSON keys keep responses compact (long responses were the source of
    # truncated / invalid JSON in live runs).
    KEYS = {REF_COL: "ref", DATE_COL: "date", AMOUNT_COL: "amount"}

    def __call__(self, rows: list[DecidedRow], context: dict) -> dict:
        """Blind reading: the model is NOT shown any proposed value, so its read is
        independent evidence. The decision (ACCEPT/CORRECT/ABSTAIN) is derived by
        comparing that read with the current proposal."""
        verdicts: dict = {}
        failures: list[str] = []
        from app.accounting.fusion import _canon

        # Tabular rows: per-cell blind reads. Everything else: whole-row crops.
        cell_rows = [r for r in rows if any(self._cell_span(r, c) for c in self.columns)]
        if cell_rows:
            reads, cell_failures = self.cell_reads(cell_rows)
            failures += cell_failures
            for row in cell_rows:
                got = reads.get(row.row_id)
                if not got:
                    continue
                out = {}
                for column, text in got.items():
                    if text in (None, "", "-", "null"):
                        out[column] = ("ABSTAIN", None)
                        continue
                    current = row.decisions.get(column)
                    same = current is not None and _canon(column, text) == current.value
                    out[column] = ("ACCEPT" if same else "CORRECT", str(text))
                verdicts[row.row_id] = out
            rows = [r for r in rows if r not in cell_rows]

        batches = [rows[i:i + ARBITRATION_BATCH] for i in range(0, len(rows), ARBITRATION_BATCH)]
        if len(batches) > MAX_ARBITRATION_CALLS:
            failures.append(f"{len(batches) - MAX_ARBITRATION_CALLS} batch(es) skipped: call budget reached")
        wanted = [self.KEYS[c] for c in self.columns]
        for call_no, batch in enumerate(batches[:MAX_ARBITRATION_CALLS]):
            images, entries = [], []
            for row in batch:
                crop = self._crop(row)
                if crop is None:
                    continue
                images.append(crop)
                entries.append({"image": len(images), "row_id": row.row_id,
                                "kind": "cash received" if row.kind == "INFLOW" else "transaction"})
            if not entries:
                continue
            prompt = (
                "Each image is ONE row cut from a financial document (register, statement or invoice).\n"
                f"For each row read these fields directly from the image: {', '.join(wanted)}.\n"
                "ref = the voucher / reference / cheque number of the row; date = the row's date (day first);\n"
                "amount = the row's money amount (for 'cash received' rows, the received amount).\n"
                "Rules: read only what is written; a crossed 7 is 7, not 1; read every digit up to the column\n"
                "border; '/-' after an amount is not a digit; a note in parentheses is not part of an amount;\n"
                "never use arithmetic or totals. If a field is not legible with certainty, use null.\n"
                'Return compact JSON only: {"rows":[{"row_id":"...","ref":...,"date":...,"amount":...}]}\n'
                f"ROWS (image numbers are 1-based, in order): {json.dumps(entries, ensure_ascii=False)}"
            )
            started = time.perf_counter()
            try:
                raw = self.client.generate_json([prompt, *images], retries=3, purpose="arbitration")
                self.trace.record("arbitration", started, "SUCCESS", f"batch {call_no + 1}", len(images))
            except Exception as exc:
                # One failed batch must not discard the others' verdicts.
                self.trace.record("arbitration", started, "FAILED", f"batch {call_no + 1}: {type(exc).__name__}", len(images))
                failures.append(f"batch {call_no + 1}: {type(exc).__name__}")
                continue
            by_id = {r.row_id: r for r in batch}
            items = raw.get("rows") if isinstance(raw, dict) else raw if isinstance(raw, list) else None
            for item in items or []:
                if not isinstance(item, dict) or item.get("row_id") not in by_id:
                    continue
                row = by_id[item["row_id"]]
                out = {}
                for column in self.columns:
                    value = item.get(self.KEYS[column])
                    if isinstance(value, (dict, list)):
                        continue
                    if value in (None, "", "-", "null"):
                        out[column] = ("ABSTAIN", None)
                        continue
                    current = row.decisions.get(column)
                    from app.accounting.fusion import _canon
                    same = current is not None and _canon(column, value) == current.value
                    out[column] = ("ACCEPT" if same else "CORRECT", str(value))
                verdicts[row.row_id] = out
        if failures and not verdicts:
            raise RuntimeError("; ".join(failures))
        if failures:
            verdicts["__failures__"] = failures
        return verdicts


def _legacy_local_rows(purpose: str, rep: DocumentRepresentation) -> list[dict]:
    """Model-like rows from the deterministic local extractors (non-tabular documents)."""
    try:
        from app.intelligence.extraction import extractor_for
        from app.intelligence.knowledge import KnowledgeBase
        if purpose not in (MEMBER_RECEIPT, VENDOR_INVOICE):
            return []
        data, _ = extractor_for(purpose, KnowledgeBase()).extract(rep)
    except Exception as exc:
        logger.warning("Local extractor failed: %s", type(exc).__name__)
        return []
    rows = []
    if purpose == VENDOR_INVOICE:
        for e in data.get("expenses") or []:
            values = _blank_values()
            values[REF_COL] = _clean(data.get("bill_number"))
            values[DATE_COL] = _clean(data.get("bill_date"))
            values[AMOUNT_COL] = _clean(e.get("expense_amount"))
            values["Comments"] = _clean(e.get("expense_description"))
            rows.append({"values": values, "kind": "LINE_ITEM", "source_line_ids": []})
    else:
        values = _blank_values()
        mapping = {"payment_type": "Payment Type*", "bank_name_or_code": "Society Bank Name/Bank code(Given to you by nobrokerhood)*",
                   "reference_number": REF_COL, "tower": "Tower No*", "flat": "Flat No*", "bill_head": "Bill Head*",
                   "amount": AMOUNT_COL, "transaction_date": DATE_COL, "comments": "Comments"}
        for key, col in mapping.items():
            values[col] = _clean(data.get(key))
        if values[AMOUNT_COL] != "-" or values[REF_COL] != "-":
            rows.append({"values": values, "kind": "TRANSACTION", "source_line_ids": []})
    return rows


class DocumentExtractionPipeline:
    def __init__(self, gemini_client, ocr_service: DocumentOcrService):
        self.gemini_client = gemini_client
        self.ocr_service = ocr_service

    @property
    def settings(self):
        return self.gemini_client.settings

    def represent(self, source_bytes: bytes) -> DocumentRepresentation:
        manifest = build_manifest(source_bytes, "source", "application/octet-stream", self.settings.poppler_path)
        return self.ocr_service.represent(source_bytes, manifest)

    def extract(self, source_bytes: bytes, purpose: str, template=None) -> dict:
        started = time.perf_counter()
        trace = CallTrace()
        rep = self.represent(source_bytes)
        ocr_ms = int((time.perf_counter() - started) * 1000)
        candidates, layouts = build_candidates(rep)
        block, alias_to_line = ocr_prompt_block(rep)

        model, shape_notes, contract_error, provider_note = None, [], None, ""
        provider = "GEMINI"
        gemini_ok = bool(self.settings.gemini_api_key)
        if not gemini_ok:
            provider, provider_note = "LOCAL_OCR", "GEMINI_API_KEY is not configured"
        else:
            images = rep.page_images()
            prompt = f"{prompt_for(purpose)}\nOCR LINES (hints; ids for source_line_ids):\n{block}"
            t = time.perf_counter()
            try:
                raw = self.gemini_client.generate_json([prompt, *images], purpose="extraction",
                                                      thinking_budget=EXTRACTION_THINKING_BUDGET)
                trace.record("extraction", t, "SUCCESS", images=len(images))
                try:
                    model, shape_notes = normalize_response(purpose, raw)
                except ExtractionContractError as exc:
                    contract_error = exc.problem
                    provider = "LOCAL_OCR"
                    provider_note = f"Gemini response violated the {purpose} contract: {exc.problem}"
                    logger.warning("EXTRACTION_CONTRACT_VIOLATION purpose=%s shape=%s problem=%s",
                                   purpose, exc.shape, exc.problem)
            except Exception as exc:
                trace.record("extraction", t, "FAILED", type(exc).__name__, len(images))
                provider = "LOCAL_OCR"
                provider_note = f"Gemini unavailable ({type(exc).__name__}): {str(exc)[:500]}"
                gemini_ok = False
                logger.warning("Gemini extraction failed; using OCR evidence only: %s", type(exc).__name__)

        local_rows = _legacy_local_rows(purpose, rep) if model is None and not any(l.tabular for l in layouts) else None
        arbiter = None
        if gemini_ok:
            columns = (AMOUNT_COL,) if purpose == VENDOR_INVOICE else (REF_COL, DATE_COL, AMOUNT_COL)
            arbiter = GeminiArbiter(self.gemini_client, rep, trace, columns, layouts)

        assembled = assemble_document(
            purpose, rep, candidates, layouts, model=model, shape_notes=shape_notes,
            contract_error=contract_error, alias_to_line=alias_to_line, arbiter=arbiter, local_rows=local_rows,
        )

        result: dict = {
            "purpose": purpose,
            "document_type": (getattr(model, "document_type", None) or purpose) if model else purpose,
            "_extraction_provider": provider,
            "provider_note": provider_note,
            **assembled,
            "ocr": rep.summary(),
            "gemini_calls": trace.calls,
            "timing_ms": {"ocr": ocr_ms, "total": int((time.perf_counter() - started) * 1000)},
        }
        if model is not None and purpose in (PETTY_CASH_REGISTER, MEMBER_RECEIPT):
            result["balance_summary"] = model.balance_summary.model_dump()
            result["period"] = getattr(model, "period", None)
        if model is not None and purpose == VENDOR_INVOICE:
            result["vendor_detail"] = model.model_dump(exclude={"line_items"})
            result["vendor_detail"]["line_items"] = [li.model_dump(exclude={"source_line_ids"}) for li in model.line_items]
        return result
