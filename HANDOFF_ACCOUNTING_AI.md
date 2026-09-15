# HANDOFF PROMPT — Continue building the local (no-Gemini) Accounting AI

Copy everything below into your AI coding assistant.

---

You are continuing work in `D:\Gen Ai\pdftoexcel` (Windows, Python 3.13.9, venv at `.venv`). A previous assistant started the work and ran out of time. Read this whole document, then read `ACCOUNTING_AI_ARCHITECTURE_AUDIT.md`, then continue from **Section 6 (Next steps)**. Do not redo the finished work. Check the state of each file before you change it.

## 1. Goal

Turn the "Accounting AI" page (`accounting.html`, API `/processing/*`) into a real multi-agent accounting workflow that does **not** depend on Gemini or any paid external LLM.

- **Pipeline:** intake → OCR → document classification → type-specific accounting extraction → independent verification → targeted repair → mapping (Mapping_Master) → validation/reconciliation → mandatory human review → Excel → Drive Output.
- **External services that stay:** Google Drive (Shared Drive), Google Sheets (config and audit) and Google Login.
- **Legacy Converter:** `ocr.html`, `/export-to-excel/`, `/process-document/` and `/split-pdf/` stay on Gemini and must not regress.

## 2. Hard rules (from the project owner)

1. **Secrets.** Never print, log or commit secrets: `.env` values, `GEMINI_API_KEY`, `service-account.json`, OAuth secrets, tokens. Do not create another `.env` and do not overwrite it. Do not commit.
2. **No fabrication.** Never invent amounts, GST, TDS, ledger codes, flat numbers, invoice numbers, vendors, payment modes or categories. Use these statuses instead:
   - `MISSING` when a value isn't in the source
   - `AMBIGUOUS` when the reading is uncertain (with candidates)
   - `POLICY_REQUIRED` when a company policy is undefined
   - `NEEDS_MAPPING` when Mapping_Master has no entry
   - `DOCUMENT_PURPOSE_MISMATCH` when the document doesn't fit the selected purpose
   - `UNKNOWN` when the type can't be determined
3. **Mapping_Master is empty.** Do not create fake mappings. Jobs must correctly stop at `NEEDS_MAPPING`.
4. **Human review is a gate.** Never auto-approve, and never generate Excel from unverified data.
5. **Tests.** Do not weaken or delete existing tests. Unit tests must never call Gemini.
6. **Models.** Do not download large models without explicit approval. No local LLM exists on this machine (details in section 4).
7. **Live writes.** Keep Drive and Sheets writes controlled, and report exactly what was written.
8. **Chain-of-thought.** Show structured evidence and reasons in the UI, not chain-of-thought.

## 3. Environment facts (verified 2026-09-15)

- **Servers:**
  - Backend: `.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8030` (log in `backend.log`).
  - Static site: `tools/no_cache_static_server.py` on port 5000.
  - Demo: `demo_server.py` on port 8031.
- **Poppler 25.07.0:** installed; `POPPLER_PATH` is set in `.env`.
- **OCR engines (installed with the owner's approval):**
  - `rapidocr 3.9.2` + `onnxruntime 1.30.0` (pip, in the venv). Models are already downloaded into `.venv\Lib\site-packages\rapidocr\models`, so it works offline.
  - Tesseract 5.4.0 at `C:\Program Files\Tesseract-OCR\tesseract.exe` + `pytesseract 0.3.13`.
  - Also pulled in: `opencv_python 5.0.0.93`, `Shapely`, `pyclipper`, `colorlog`, `flatbuffers`.
  - **Not yet in `requirements.txt`: add `rapidocr`, `onnxruntime`, `pytesseract`.**
- **Machine:** 7.7 GB RAM (often under 1 GB free), Intel Iris Xe (no CUDA), about 13 GB free disk on C: and D:.
  - No Ollama, llama.cpp, GGUF, torch or transformers.
  - scikit-learn, scipy and numpy are installed.
- **Gemini:** the configured key's project hit its monthly spending cap (429). Irrelevant for the new path, but it's why the local path matters.
- **Google Drive:** Shared Drive "NoBrokerHood Accounting AI" contains `MEMBER_RECEIPT/` and `VENDOR_INVOICE/` folders, each with Incoming, Review, Completed and Output.
  - Folder_Config (Sheets) points to them, and every Drive call uses `supportsAllDrives=True`.
  - `GOOGLE_DRIVE_ROOT_FOLDER_ID` in `.env` still points to an old My Drive folder. Only the health check uses it; the owner should update it manually.
- **Job state:** stored as one JSON cell in the Sheets tab `Job_State` (limit about 50,000 characters). Large OCR data must not go there.

### Measured OCR results

- **Synthetic member receipt** (`tests/fixtures/test_member_receipt.pdf`): both engines read every field correctly. RapidOCR took 4.7 s and Tesseract 2 s.
- **Real handwritten July-2025 petty cash register:** `C:\Users\Meet\OneDrive\Documents\NBH\PDF\WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf`, one A4 page containing real staff names. **Do not copy it into the repo or tests.**
  - RapidOCR at 200 dpi takes 15.6 s and returns 180 lines. Voucher numbers are mostly correct, dates are mostly readable (7 is often read as 4), and particulars are partially readable.
  - Amounts appear with the handwritten `/-` suffix misread as `1`: `9001-` = 900/-, `8,4001` = 8,400/-, `100001-` = 10,000/-.
  - The page is a slightly skewed photo. The serial-number column sits 10–60 px lower than the date column in the same row.

## 4. What is already done

### 4.1 Earlier work (all tests passing)

- **Gemini client:** error classes for spend cap, daily quota (not retried), invalid key (safe message) and rate limit; an optional fallback key; bounded retries.
- **Stage progress:** `app/api/processing.py::progress()` reports `DONE / FAILED / NEEDS_ATTENTION / NOT_REACHED / PENDING`. Failed jobs show later stages as `NOT_REACHED`.
- **Frontend:** `accounting.html` renders those stage states, escapes HTML (`escapeHtml`), and shows review-block messages (`REVIEW_BLOCK_MESSAGES`).
- **Workflow guards:** approval requires `NEEDS_REVIEW` + verification `PASSED` + mapping `MAPPED`; reject is allowed only from `NEEDS_REVIEW` or `FAILED`.
- **Output:** the Excel writer produces numeric amounts and real dates (`app/accounting/output.py`), and Sheets row updates are batched.
- **Test suite:** `.venv\Scripts\python.exe -m pytest -q` → **107 passed** (re-run after the new work below, still 107 passed).

### 4.2 This session — FILES CREATED

| File | Status | Purpose |
|---|---|---|
| `ACCOUNTING_AI_ARCHITECTURE_AUDIT.md` | Done | Phase 1 audit: current architecture, problems, Gemini dependencies, what the Legacy Converter really does (its "OCR" is Gemini; the reusable parts are ingestion, Poppler rendering and Excel writing), machine capability, proposed architecture, risks. |
| `app/documents/ingestion.py` | Done, used | Shared intake: `SUPPORTED_DOCUMENT_TYPES`, `DocumentManifest` (filename, content_type, detected_format, size, sha256, page_count), `detect_format`, `load_page_images(bytes, poppler_path, dpi)` (all PDF pages via Poppler, or one image), `build_manifest`, `InvalidImageError`. |
| `app/documents/ocr.py` | Done, manually tested | `OcrLine` (text, confidence, bbox, page, engine), `OcrPage` (lines, preprocessing, `script` = PRINTED/HANDWRITTEN_LIKELY heuristic), `DocumentRepresentation` (manifest, pages, `to_dict/from_dict`, compact `summary()`), `group_rows(lines, overlap)`, `RapidOcrProvider` (primary, lazy singleton + lock), `TesseractOcrProvider` (`read`, and `read_region(image, bbox, numeric)` for independent crop re-reads), `find_tesseract()`, `DocumentOcrService(poppler_path, primary, secondary, dpi=200)` (renders every page, re-OCRs low-confidence pages with autocontrast+sharpen, LRU cache by sha256), `OcrUnavailableError` (code `OCR_ENGINE_UNAVAILABLE`), `DocumentUnreadableError` (code `DOCUMENT_UNREADABLE`). |
| `app/intelligence/__init__.py` | Done | Package marker. |
| `app/intelligence/knowledge.py` | Done, tested | `KnowledgeBase` loads `knowledge_base/accounting/*.json`, local TF-IDF `search()` (scikit-learn), `rule()`, `policy()`, `gst_rates_on(date)`, `tds_rates()`, `suggest_category(text)`, `summary()`. `default_knowledge_base()` is cached. |
| `knowledge_base/accounting/manifest.json` | Done | Version 1.0.0, file list, changelog. |
| `knowledge_base/accounting/document_types.json` | Done | Keyword weights and structural-signal weights per document type; thresholds (`minimum_confidence_for_mismatch` 0.6, `many_rows` 5, `unknown_below_score` 4). |
| `knowledge_base/accounting/vocabulary.json` | Done | Receipt and invoice field label synonyms, payment modes, reference regex per mode, flat/tower regex, bill heads, handwriting knowledge (suffix misreads, digit confusions, letter→digit map). |
| `knowledge_base/accounting/gst.json` | Done | Rate slabs by effective date (from 2025-09-22: 0, 0.25, 3, 5, 18, 40), GST rules, GSTIN regex, state codes. |
| `knowledge_base/accounting/tds.json` | Done | Reference rates for 194C/J/I/H/Q plus the no-PAN rate; section applicability is `POLICY_REQUIRED`. |
| `knowledge_base/accounting/concepts.json` | Done | Accounting concepts (debit/credit, society receipts, vendor bills, petty cash, reconciliation, duplicates, handwritten amounts, Indian grouping). |
| `knowledge_base/accounting/expense_categories.json` | Done | Keyword category *suggestions* only; never used as codes. |
| `knowledge_base/accounting/validation_rules.json` | Done | Rule catalog (ids, severity, applies_to, KB references), parameters (rounding tolerance 1.0, etc.), mandatory fields per purpose. **The rules are catalogued but not yet implemented in code.** |
| `knowledge_base/accounting/policies.json` | Done | Company policies, all `POLICY_REQUIRED` (expense ledger codes, bill head codes, bank codes, imprest limit, salary via petty cash, TDS section, GST input credit, register import template). `ROUNDING_TOLERANCE` is `TECHNICAL_DEFAULT` 1.0. |
| `app/intelligence/parsing.py` | Done, tested on real tokens | `read_handwritten_amount` (handles `/-`, `1-` suffix misreads, broken grouping `8,4001`, `.` thousands, single stray letters → digits; otherwise returns `AMBIGUOUS` with candidates, preferred candidate first), `parse_printed_amount`, `find_printed_amounts`, `split_merged_amounts("98,6231,08,300")` → [98623, 108300], `read_handwritten_date` (noise like `10-07-2.5`, `15107.25`, `04-0725`), `parse_printed_date`, `find_printed_dates`, `date_digit_variants(date, confusions)` for repair, `amount_from_words` (Indian lakh/crore + paise), `fuzzy_contains`, `similarity`, `normalise_reference`. |
| `app/intelligence/classification.py` | Done, tested on real data | `DocumentClassifier.classify(representation, selected_purpose)` → `ClassificationResult` (detected_type, label, confidence via softmax, `purpose_status` ∈ `MATCH / DOCUMENT_PURPOSE_MISMATCH / DOCUMENT_TYPE_UNCERTAIN`, scores, human-readable evidence, recommended_action, recommended_purpose, signals). `structural_signals()` detects: rows with both a date and an amount, voucher sequence (dominant x-column), balance line, month header, flat reference, GSTIN, tax breakup, Dr/Cr columns, handwriting. **Verified:** the real ledger classifies as PETTY_CASH_REGISTER at 0.99 (selecting MEMBER_RECEIPT → mismatch with evidence and "re-upload as Petty Cash / Expense Register"), and the synthetic receipt as MEMBER_RECEIPT at 1.0. |
| `app/intelligence/evidence.py` | Done | `Evidence` (page, text, bbox, ocr_confidence, engine, label, row), `FieldResult` (value, status FOUND/MISSING/AMBIGUOUS, confidence HIGH/MEDIUM/LOW, score, evidence, candidates, reasons, repaired), `missing()`. |
| `app/intelligence/confidence.py` | Done | `assess()` gives the initial score from OCR confidence, label anchoring, format and corrections (ambiguous values capped at 0.4). `apply_check(result, passed, description)` adjusts it after independent checks. HIGH ≥ 0.85, MEDIUM ≥ 0.6. |
| `app/intelligence/extraction.py` | **Written, partly tested — a bug was just fixed and not re-run** | `MemberReceiptExtractor`, `VendorInvoiceExtractor`, `RegisterExtractor`, `extractor_for(purpose, kb)`. Each `extract(representation)` returns `(data, details)`: `data` uses the purpose schema; `details` holds per-field FieldResult dicts with evidence. `RegisterExtractor` classifies cells (DATE/SERIAL/VOUCHER/AMOUNT/TOTALS/BALANCE/HEADER/TEXT), detects the payment and receipt amount columns, and groups rows by date anchors with per-column vertical offsets for skew. It also reads opening/closing balance and written totals, and keeps page and row numbers. The last error (`'_Cell' object has no attribute 'page'`, in the particulars/category branch of `_row`) was fixed by using `items[0].line`. **Re-run the extractor test script (section 6, step 1) first.** |
| `tests/fixtures/test_member_receipt.pdf`, `.png`, `make_synthetic_receipt.py` | Done | Clearly synthetic NEFT receipt. Ground truth: amount 12,450.75; date 03-Sep-2026; NEFT; UTR HDFCN52026090312345678; beneficiary bank HDFC Bank Ltd; remarks "Maintenance charges Jul-Sep 2026"; Tower B Flat 1204; remitter "Test Resident One"; amount in words "Rupees Twelve Thousand Four Hundred Fifty and Seventy Five Paise Only". |

### 4.3 This session — FILES MODIFIED

| File | Change |
|---|---|
| `app/documents/conversion.py` (Legacy Converter) | `get_images_from_upload` now uses the shared `load_page_images` (image branch and PDF branch). Behaviour and errors are unchanged (503 Poppler missing, 400 invalid PDF/image). Still uses Gemini. |
| `app/agents/extractor.py` | `source_parts` now uses `load_page_images`. `validate_extraction` supports `PETTY_CASH_REGISTER` via `CashRegisterExtraction`. Gemini providers unchanged. |
| `app/accounting/purposes.py` | New purpose `PETTY_CASH_REGISTER` ("Petty Cash / Expense Register", `row_based=True`, accepts PETTY_CASH_REGISTER/CASH_BOOK/EXPENSE_REGISTER). Adds the `DOCUMENT_TYPES` list, `DOCUMENT_TYPE_LABELS`, `accepts` on each purpose, `purpose_definition()` and `purpose_for_document_type()`. **Note:** `supported_purpose_codes()` now includes PETTY_CASH_REGISTER, but Folder_Config has no row and no Drive folders for it yet. |
| `app/accounting/schemas.py` | Adds `CashRegisterRow` (source_page, source_row, serial_no, voucher_no, date, particulars, category, debit_credit PAYMENT/RECEIPT, amount, amount_status, amount_candidates, running_balance, confidence, expense_code) and `CashRegisterExtraction` (register_period, opening/closing balance, written payment/receipt totals, rows). `ValidationIssue` gains optional `code, rule_id, row, evidence, current_value, suggested_value, action`. `MappingMissingItem` gains optional `field, suggested_category, reason, rows`. All backward compatible. |
| `app/services/gemini_client.py`, `app/workflows/accounting_graph.py`, `app/api/processing.py`, `accounting.html`, `tests/test_gemini_errors.py` | Earlier-session fixes (error classes, stage states, skipping verification when extraction failed). Already covered by the 107 passing tests. |

## 5. Design decisions already made (keep them)

1. **Provider compatibility.** The workflow keeps its LangGraph structure and the existing provider protocols:
   - `extract(source_bytes, purpose, template)`
   - `verify(source_bytes, purpose, template, extracted_data)`
   - `repair(..., verification_result)`

   New local providers declare `requires_document_representation = True`. The workflow runs the new intake/OCR/classification nodes only for such providers. Injected providers (Gemini, and the tests' Static providers) don't declare it, so OCR and classification are shown as `SKIPPED`. **Result: existing tests need no setup changes.**
2. **Sharing OCR with providers.** Local providers get the OCR representation from `DocumentOcrService`'s sha256 cache, filled by the OCR node, so OCR runs once per document.
3. **New defaults.** Local providers become the defaults in `app/app_factory.py`; Gemini providers stay importable. Add a test that the Accounting AI runs with `gemini_api_key=None`.
4. **Purpose mismatch flow.** After classification, extract with the detected type's extractor (as evidence only) and verify. Mapping is `NOT_REACHED`; validation gets a CRITICAL `DOCUMENT_PURPOSE_MISMATCH` issue (`NEEDS_ATTENTION`); human review is `NEEDS_ATTENTION`; Excel is `NOT_REACHED`. Approval is blocked with `PURPOSE_MISMATCH` and reject is allowed. The UI shows selected type, detected type, confidence, evidence list and recommended action.
5. **Row-based register data.** For PETTY_CASH_REGISTER, `extracted_data` = `CashRegisterExtraction` dict. Keep `extracted_data` flat and schema-compatible for member receipts and invoices (mapping, validation and Excel depend on it). Store evidence separately in a new job field `extraction_details`.
6. **Heavy data goes to Drive.** Upload the full `DocumentRepresentation.to_dict()` as JSON to the purpose's Review folder. `Job_State` keeps only `summary()` plus the Drive file id. Add compaction in `app/processing/stores.py` if State JSON exceeds about 45,000 characters.
7. **Progress stages.** Keep the 8 labels and add "OCR / Document reading" and "Document classification". Target status vocabulary: `NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED, NEEDS_ATTENTION, NOT_REACHED, SKIPPED`.
   - Derive statuses from per-stage fields plus `current_step` and `overall_status`.
   - Verification that ran but found problems is `NEEDS_ATTENTION`; `FAILED` means the stage itself errored (overall FAILED).
   - Update `tests/test_gemini_errors.py::test_workflow_extraction_daily_quota_fails_with_one_call`, which currently expects `DONE`, to the new vocabulary (`COMPLETED`); that assertion was written in the previous session.
   - The frontend needs visual styles for done, working, failed, attention and not-reached.
8. **Real IN_PROGRESS.** Add an optional async mode to `POST /processing/jobs` (form field `run_async=true`, used by the UI). It returns the job immediately and runs the workflow in the background (the job object stays live in memory); the UI polls `GET /processing/jobs/{id}`. Default stays synchronous so existing tests are unchanged.
9. **Audit events.** Keep the existing names that tests assert (`FILE_UPLOAD`, `EXTRACTION_STARTED/COMPLETED/FAILED`, `AI_VERIFICATION_STARTED/PASSED/FAILED`, `AI_REPAIR_FAILED`, `MAPPING_REQUIRED`, `HUMAN_APPROVED`, `EXCEL_GENERATED`). Add `OCR_COMPLETED/OCR_FAILED`, `DOCUMENT_CLASSIFIED`, `DOCUMENT_PURPOSE_MISMATCH`, `REPAIR_PERFORMED`, `VALIDATION_COMPLETED`, `HUMAN_REVIEW_REQUIRED`. Sheets allows 60 writes/min, so don't add unnecessary writes.
10. **Excel.** The first sheet stays the existing template import sheet (tests read `workbook.active`). Add sheets: Summary, Transactions (register rows), Validation, Exceptions, Source Evidence, Accounting Mapping, Audit Log. The register template is an internal review workbook until policy `REGISTER_IMPORT_TEMPLATE` is defined.

## 6. Next steps (in order)

1. **Re-test the extractors on real data.** Write a scratch script outside the repo that:
   - builds a `DocumentRepresentation` from `tests/fixtures/test_member_receipt.pdf` via `DocumentOcrService(get_settings().poppler_path, RapidOcrProvider()).represent(data, build_manifest(...))`
   - runs `MemberReceiptExtractor(default_knowledge_base()).extract(rep)` and compares against the ground truth in 4.2
   - runs the same OCR on the real ledger PDF path above and calls `RegisterExtractor(kb).extract(rep)`. Expect about 29 payment rows (vouchers 266–308), 3 cash-received rows (10,000 / 68,300 / 30,000; OCR may miss some), written totals, and a closing balance around 10,17x. Print rows; tune column and row grouping.

   Never put the real ledger's content into repo tests.
2. **Rule engine: `app/intelligence/rules.py`.** Implement every rule id in `validation_rules.json` as a pure function returning `RuleResult(rule_id, status PASS/FAIL/NOT_APPLICABLE/POLICY_REQUIRED, severity, field, row, message, evidence, current_value, suggested_value, action, kb_refs)`. Parameters come from the KB. Rules to cover:
   - **Member receipt:** amount-in-words vs figure; reference format vs payment mode; cheque details.
   - **Vendor invoice:** GST split; rate slab by invoice date; place of supply from the two GSTIN state codes; line items = taxable; taxable + taxes + cess = total; TDS ≤ taxable and at a recognised rate (section = `POLICY_REQUIRED`).
   - **Register:** row completeness; voucher duplicates and gaps; duplicate rows; dates in the register period and chronological; column total = written total; opening + receipts − payments = closing only when both balances are readable, otherwise `NOT_APPLICABLE` with a reason.
   - **All types:** amount positive, valid date, low confidence/ambiguous, duplicate document by sha256 (WARNING; CRITICAL if a previous job with the same hash COMPLETED).
3. **Independent verifier: `app/intelligence/verification.py`.** Do not reuse the extraction choice. Re-render the page, crop each value's bbox and re-read it with `TesseractOcrProvider.read_region(numeric=True)` for amounts, dates, vouchers and references. Compare readings: VERIFIED / MISMATCH / NOT_FOUND / UNCERTAIN, then run the rules. Return a dict compatible with the existing `VerificationResult` schema (`overall_status`, `fields[field, extracted_value, verified_value, status, confidence 0-1, evidence, page_number]`) plus `rule_results`. For registers, use field names like `rows[5].amount`. Overall: PASSED only if every field is VERIFIED and no CRITICAL rule failed; FAILED on MISMATCH/NOT_FOUND or a critical failure; otherwise NEEDS_REVIEW.
4. **Targeted repair: `app/intelligence/repair.py`.** Repair only the failed fields, with a stated cause:
   - resolve ambiguous suffix readings by making the payment column total match the written total (exhaustive search when ≤ 10 ambiguous values; accept only a unique matching combination)
   - repair dates outside the register period using `date_digit_variants` constrained by the period and neighbouring rows (accept only one candidate)
   - resolve amount vs amount-in-words on receipts
   - prefer the second-engine reading when it satisfies a rule

   Record a `repairs` list (field, old, new, cause, evidence). If nothing changes, route to human review, not another loop. Keep the retry limit (`AI_VERIFICATION_MAX_RETRIES`).
5. **Providers: `app/intelligence/providers.py`.** `AccountingIntelligenceProvider` protocol; `DeterministicAccountingProvider` implementing extract/verify/repair on top of the above (with `requires_document_representation = True`); and `detect_local_llm()` (reports that none is available; don't implement an LLM provider unless a runtime exists).
6. **Workflow** (`app/workflows/accounting_graph.py`, `state.py`, `routing.py`, `app/processing/jobs.py`, `stores.py`):
   - add nodes `intake` (manifest + duplicate check), `ocr` and `classify`
   - add job fields `document_manifest, ocr_status, ocr_summary, representation_drive_file_id, classification_status, classification, extraction_details, rule_results, repairs, review_reasons` (persist them in `job_to_state/job_from_state`)
   - route mismatches as in 5.4
   - add an approval guard `PURPOSE_MISMATCH` and block when CRITICAL rule failures remain
7. **Mapping** (`app/accounting/mapping.py`):
   - register rows: mapping type `EXPENSE_CATEGORY`, one missing item per distinct category with rows, suggested category (KB) and reason
   - admin-only `POST /config/mappings` that appends validated rows to Mapping_Master (use `require_admin` in `app/api/config.py`)
   - optional "save to Mapping_Master" in `/jobs/{id}/mapping` for admins
8. **Validation** (`app/accounting/validation.py`): keep the current checks and add rule-engine results as `ValidationIssue`s with evidence, current value, suggested value and action.
9. **API** (`app/api/processing.py`, `app/api/config.py`):
   - `job_payload` includes classification, extraction details (trimmed), rule results, repairs and review reasons
   - error codes `PDF_INVALID, OCR_FAILED, POPPLER_UNAVAILABLE, DOCUMENT_UNREADABLE, DOCUMENT_PURPOSE_MISMATCH, EXTRACTION_FAILED, ACCOUNTING_VALIDATION_FAILED, MAPPING_REQUIRED, LOCAL_AI_UNAVAILABLE, DRIVE_UPLOAD_FAILED, EXCEL_GENERATION_FAILED` with friendly messages and no stack traces
   - `GET /config/knowledge-base` (summary)
   - health reports OCR engine availability
   - add `PETTY_CASH_REGISTER` to the purposes list and the `/config/public` features
10. **UI** (`accounting.html`, file currently about 880 lines):
    - purpose option for the register
    - Document Classification panel (selected, detected, confidence %, result, evidence bullets, recommended action)
    - "Why review is required" table (issue, severity, field/row, evidence, current, suggested, reason, action)
    - register rows table
    - per-field confidence badges and evidence tooltips
    - stage styles and polling for async mode
    - keep `escapeHtml` on all dynamic text
11. **Excel** (`app/accounting/output.py`): the multi-sheet workbook from 5.10, reusing the shared openpyxl helpers.
12. **Drive and Sheets set-up for the new purpose — live writes; ask the owner first and report them:**
    - create `PETTY_CASH_REGISTER/{Incoming,Review,Completed,Output}` in the Shared Drive with `supportsAllDrives=True`
    - append a Folder_Config row
    - optionally a Template_Master row
13. **Tests to add** (fast; use OCR-line JSON fixtures built from *synthetic* data, plus a few real-OCR tests on synthetic images):
    - no Gemini required; PDF conversion; multi-page (generate a synthetic 2-page register PDF with PIL, including repeated headers and b/f–c/f lines); OCR on the fixture receipt
    - receipt extraction, and invoice extraction (generate a synthetic tax invoice image clearly marked SYNTHETIC)
    - register row extraction with handwriting artifacts (synthetic OCR lines like `9001-`, `04-04-25`); purpose mismatch
    - GST split, rate, total; TDS; cashbook reconciliation; duplicate row/voucher/document
    - mapping required; human review gate; Excel sheets; evidence present; NOT_REACHED stages
    - Legacy Converter still works
    - parsers: `read_handwritten_amount`, `read_handwritten_date`, `amount_from_words`, `split_merged_amounts`
14. **Run tests:**
    - `.venv\Scripts\python.exe -m pytest -q`
    - with Gemini blocked: create a pytest plugin outside the repo that patches `httpx.Client.send` to raise on `generativelanguage.googleapis.com` and counts attempts; run with `PYTHONPATH=<plugin dir>` and `-p block_gemini_network`. Expect 0 attempts.
15. **Real E2E** through the API or `http://127.0.0.1:5000/accounting.html?v=light-ui`:
    - **A)** Synthetic receipt as MEMBER_RECEIPT. Expect the classification to match, fields extracted and verified, and a stop at `NEEDS_MAPPING`.
    - **B)** Real ledger as MEMBER_RECEIPT. Expect `DOCUMENT_PURPOSE_MISMATCH` with evidence.
    - **C)** Real ledger as PETTY_CASH_REGISTER (after step 12). Expect all rows, reconciliation issues surfaced, and `NEEDS_MAPPING`.

    Report every Drive and Sheets write. Restart the backend afterwards and confirm exactly one listener on 8030.
16. **Final report:** architecture before/after, Gemini removed from the Accounting AI path, reused components, rules, knowledge base, local model availability (none), files created/modified, tests and results, E2E results, limitations, business configuration still required, run commands. Verdict: `ACCOUNTING AI READY` or `ACCOUNTING AI NOT READY` with exact blockers.

## 7. Known limitations to report honestly

- **Handwriting:** local OCR only partly reads handwriting. Registers will carry AMBIGUOUS/LOW values that need human review; that is correct behaviour, not a bug.
- **Mapping_Master is empty:** approval and Excel can't be reached without real codes. This is business configuration, not a defect.
- **Undefined policies:** everything in `policies.json` marked `POLICY_REQUIRED` must be answered by the business.
- **No local LLM:** there is no semantic interpretation beyond rules and the knowledge base; the provider interface is ready for one later.
- **"Training":** nothing has been trained. The foundation is the knowledge base, rules, schemas, evaluation fixtures and regression tests. Human corrections are recorded in Job_State and Activity_Log and can be added to the evaluation data later.
