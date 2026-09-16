# Accounting AI — Architecture Audit

Audit date: 2026-09-15. Scope: `D:\Gen Ai\pdftoexcel` as it exists today, before the local-first rebuild.

## 1. Current architecture

```
accounting.html ──POST /processing/jobs──▶ app/api/processing.py
                                             │  PDF pre-flight (Poppler page count)
                                             │  Template_Master + Folder_Config lookup (Sheets)
                                             │  upload source → Shared Drive <PURPOSE>/Incoming
                                             ▼
                               app/workflows/accounting_graph.py (LangGraph)
     prepare → extract → verify ─┬─ PASSED ─▶ map ─┬─ MAPPED ─▶ validate ─▶ human_review
                                 ├─ FAILED ─▶ repair ─▶ verify (bounded by AI_VERIFICATION_MAX_RETRIES)
                                 └─ overall FAILED ─▶ human_review
     approve (API) → re-validate → XLSX (openpyxl) → Drive Output, source → Completed
```

| Concern | Where | How it works today |
|---|---|---|
| Extraction | `app/agents/extractor.py` | `GeminiExtractionProvider`: page images + prompt → one JSON object. |
| Verification | `app/agents/verifier.py` | `GeminiVerificationProvider`: second Gemini call over the same images. |
| Repair | `app/agents/repair.py` | `GeminiRepairProvider`: third Gemini call, returns the whole object again. |
| Schemas | `app/accounting/schemas.py` | `MemberReceiptExtraction`, `VendorInvoiceExtraction` (pydantic). Only two purposes. |
| Mapping | `app/accounting/mapping.py` | `Mapping_Master` sheet lookup (BANK, BILL_HEAD, TOWER_FLAT, VENDOR, EXPENSE). Missing → `NEEDS_MAPPING`. Sheet is currently empty. |
| Validation | `app/accounting/validation.py` | Required-field presence and amount > 0. No arithmetic, GST, TDS, balance or duplicate checks. |
| Output | `app/accounting/output.py` | One sheet with template columns; numeric amounts and real dates. |
| Job state | `app/processing/stores.py` | Whole job serialised into one `Job_State` cell (`State JSON`). |
| Progress | `app/api/processing.py::progress` | Derived from per-stage status fields; FAILED jobs show `NOT_REACHED`. |
| Drive | `app/google/drive_service.py` | Shared Drive, `supportsAllDrives=True` on every call. |

## 2. Current problems

1. **Every intelligence step is Gemini.** Extraction, verification and repair are all Gemini calls. With the configured key at its monthly spending cap, Accounting AI cannot process any document.
2. **"Verification" is not independent.** It is the same model reading the same image again; there is no deterministic check of the values.
3. **"Repair" re-extracts the whole record** rather than fixing the fields that failed.
4. **No document classification.** Whatever purpose the user selects is assumed. The July 2025 handwritten petty-cash register would be forced into `MEMBER_RECEIPT` fields.
5. **Only single-record schemas.** A ledger/cash book with 29 rows cannot be represented.
6. **Validation is presence-only.** No GST/CGST/SGST/IGST, TDS, invoice-total, balance, duplicate or date-validity rules.
7. **No source evidence or confidence.** Values carry no page, text snippet, coordinates or confidence.
8. **Progress has no IN_PROGRESS/SKIPPED** and no stage for classification or OCR.
9. **Job state size.** `State JSON` lives in one Google Sheets cell (50,000-character limit). Full OCR output (words + boxes) per page will not fit and must be stored elsewhere (Drive) with only a summary in the cell.

## 3. Legacy Converter — what it really does

`ocr.html` calls `/export-to-excel/`, `/process-document/` and `/split-pdf/` (`app/api/legacy_tools.py` → `app/documents/conversion.py`, `pdf_splitter.py`).

| Step | Implementation | Reusable without Gemini? |
|---|---|---|
| Accept upload, type and size checks | `get_images_from_upload` | Yes |
| PDF detection, page count, Poppler errors | `app/documents/pdf_images.py` (`is_pdf`, `pdf_page_count`, `pdf_to_images`) | Yes — already shared with Accounting AI |
| Multi-page rendering | `pdf2image.convert_from_bytes` (all pages) | Yes |
| **OCR, table/row/column detection, handwriting** | **None locally. Each page image is sent to Gemini with a "extract the table as JSON" prompt.** | **No** |
| Column unification across pages | `export_as_excel` (first page's keys) | Idea reusable |
| Excel writing | `pandas.DataFrame.to_excel` | Yes |
| PDF split | `PyPDF2` | Not needed by Accounting AI |

**Consequence:** there is no proven local OCR engine to reuse. The reusable, proven parts are ingestion, Poppler rendering, multi-page handling and Excel writing. A local OCR engine has to be added; the Legacy Converter itself stays on Gemini (it is not the Accounting AI path) and must not regress.

## 4. Gemini dependencies

| Component | Gemini use | Accounting AI path? |
|---|---|---|
| `app/agents/extractor.py`, `verifier.py`, `repair.py` | Direct | **Yes — to be replaced** |
| `app/app_factory.py` | Builds Gemini providers as defaults | **Yes — defaults change to local providers** |
| `app/services/gemini_client.py` | Client, retries, fallback, error classes | Kept for Legacy Converter |
| `app/documents/conversion.py`, `app/api/legacy_tools.py` | Direct | No (Legacy Converter) |
| `kb/kb_service.py` | RAG answer generation (falls back to excerpts) | No (Knowledge Bot) |

Pinecone/sentence-transformers/chromadb are listed in `requirements.txt` but **not installed**; the Knowledge Bot's retrieval is not usable offline today. Accounting AI will not depend on it.

## 5. Machine capability (for local AI)

| Item | Finding |
|---|---|
| Python | 3.13.9 (venv) |
| RAM | 7.7 GB total, ~0.3 GB free at audit time |
| GPU | Intel Iris Xe (integrated) — no CUDA |
| Disk | C: 12.9 GB free, D: 12.6 GB free |
| OCR engines | None (no Tesseract, EasyOCR, PaddleOCR, RapidOCR, docTR) |
| ML runtimes | numpy, scipy, scikit-learn present; no torch, transformers, onnxruntime |
| Local LLM | None (no Ollama, llama.cpp, GGUF files, HF cache) |

A local LLM is not realistic on this machine without a multi-GB download and more free memory. Per the requirements it is optional and must not be downloaded without approval. **A local OCR engine is required** for any no-Gemini processing; the candidates are small (RapidOCR ≈ 15 MB of ONNX models via pip; Tesseract ≈ 50–70 MB installer).

## 6. Proposed architecture

```
UPLOAD ─▶ Intake agent ─▶ OCR agent ─▶ Classification agent ─▶ Extraction agent
         (manifest,       (Poppler      (rules over OCR text     (type-specific:
          sha256, pages,   render +      and layout; evidence     receipt / invoice /
          duplicate)       OCR provider  + confidence)            register rows)
                           per page)            │
                                                ├─ selected ≠ detected ─▶ DOCUMENT_PURPOSE_MISMATCH ─▶ human review
                                                ▼
        Verification agent (deterministic re-read of evidence + rule engine) ─▶ Repair agent (targeted)
                                                ▼
        Mapping agent (Mapping_Master) ─▶ Validation/reconciliation (rule engine) ─▶ Human review gate
                                                ▼
        Approve ─▶ Excel (Summary, Transactions, Validation, Exceptions, Source Evidence, Mapping, Audit) ─▶ Drive Output
```

- `app/documents/ocr.py` — `OcrProvider` protocol, local engine implementation, `DocumentRepresentation` (pages, lines, words, boxes, confidence).
- `app/intelligence/` — `classification.py`, `extraction/` (per document type), `rules/` (rule engine + rules), `knowledge/` (versioned YAML/JSON knowledge base + loader/retrieval), `confidence.py`, `verification.py`, `repair.py`, `providers.py` (`AccountingIntelligenceProvider`, `DeterministicAccountingProvider`, optional `LocalLLMAccountingProvider` stub gated on an existing runtime).
- Workflow keeps LangGraph; adds intake/OCR/classify nodes and a `stages` map with `NOT_STARTED | IN_PROGRESS | COMPLETED | FAILED | NEEDS_ATTENTION | NOT_REACHED | SKIPPED`.
- Heavy OCR output is saved as JSON beside the source in Drive (Review folder); `Job_State` keeps a compact summary.

## 7. Files to modify

`app/app_factory.py`, `app/workflows/accounting_graph.py`, `app/workflows/state.py`, `app/workflows/routing.py`, `app/processing/jobs.py`, `app/processing/stores.py`, `app/api/processing.py`, `app/accounting/schemas.py`, `app/accounting/mapping.py`, `app/accounting/validation.py`, `app/accounting/output.py`, `app/accounting/purposes.py`, `accounting.html`, `requirements.txt`, tests that construct Gemini providers as defaults.

## 8. Files to create

`app/documents/ocr.py`, `app/intelligence/**`, `knowledge_base/accounting/**` (versioned rules and policies), `tests/test_rules_*.py`, `tests/test_classification.py`, `tests/test_local_workflow.py`, `tests/test_no_gemini_dependency.py`, `tests/fixtures/` (synthetic documents with ground truth).

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Handwriting.** Small local OCR engines read printed text well and cursive handwriting poorly. | The July 2025 register will produce partial, low-confidence rows. | Classify from layout + whatever text is read; mark rows `AMBIGUOUS`/LOW; mandatory human review; never guess digits. |
| No local LLM | No semantic interpretation of free text | Deterministic rules + knowledge base; provider interface ready for a future local model. |
| Low free RAM | OCR on large pages may be slow or fail | Render at moderate DPI, process page by page, release images. |
| Sheets 50k cell limit | Job state write fails for big documents | Store OCR representation in Drive; compact state. |
| Mapping_Master empty | No job can reach approval | Correct `NEEDS_MAPPING` with exact missing items; admin mapping endpoint. No fabricated codes. |
| Company policy unknown (e.g. GST ledger codes, TDS sections applied) | Rules can't decide | `POLICY_REQUIRED` entries in the knowledge base, surfaced as review issues. |
| Existing 107 tests assume Gemini providers | Refactor could break them | Keep Gemini providers importable; tests that inject providers keep working; add new tests rather than weaken old ones. |
