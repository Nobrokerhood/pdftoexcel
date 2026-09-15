# Project Brief: NoBrokerHood "Accounting AI" (repo: `pdftoexcel`)

> **How to use this document (for ChatGPT):**
> You are helping a developer who runs this project locally. They will describe a problem or feature request. Your job is to write a precise, self-contained **prompt for Claude Code** (an AI coding agent that has full access to this repository, can read/edit files, run commands, and restart servers).
> Use the details below to name the exact files, functions, endpoints, sheet tabs and constraints involved. A good prompt states: the goal, the observed behaviour vs expected behaviour, which files are likely involved, any constraints (don't break tests, don't commit secrets, keep Google login), and how to verify the fix (a test, a curl call, or a UI step).
> This document contains **no secrets**. Never ask the developer to paste keys into the prompt; Claude Code can already read them from `.env` locally.

---

## 1. What the project is

**Accounting AI** is an internal tool for NoBrokerHood's housing-society accounting team. It turns uploaded accounting documents into Excel files ready to import into the accounting ERP:

- **Member Bank Receipt** (`MEMBER_RECEIPT`): a resident's payment proof (bank receipt, UPI or cheque) becomes one receipt row.
- **Vendor Invoice** (`VENDOR_INVOICE`): a supplier bill becomes one or more vendor-bill rows (one per expense line).

Pipeline: **Google login → pick purpose → upload PDF/JPG/PNG → file saved to Google Drive → Gemini AI extracts fields → second Gemini call verifies them → auto-repair if verification fails → map names to accounting codes → validate → mandatory human review (edit, confirm mappings) → approve → XLSX generated, uploaded to Drive, downloadable.**

Every step is audited to Google Sheets. Google Sheets acts as the database (users, config, mappings, logs, job state). The tool never posts to the ERP directly; a human always approves first.

The repo also contains older tools:
- **Legacy converter** (`ocr.html`): a generic document/table → CSV/Excel converter using Gemini, plus a PDF splitter.
- **Voice tool** (`voice.html`): records audio and sends it to a *separate* hosted backend (not in this repo).
- **Knowledge Bot**: a RAG Q&A endpoint over society FAQ documents (Pinecone/Chroma/TF-IDF + Gemini).

---

## 2. Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, **FastAPI** (0.141), Uvicorn |
| Workflow engine | **LangGraph** (1.2) `StateGraph` |
| AI | **Google Gemini** via `google-genai` SDK (default model `gemini-2.5-flash`) |
| Data validation | Pydantic v2 models |
| Google APIs | `gspread` (Sheets), `google-api-python-client` (Drive v3), `google-auth` (ID token verification + service account) |
| Documents | `pdf2image` (**needs Poppler binaries**), Pillow, PyPDF2, openpyxl, pandas |
| Frontend | Plain static HTML + vanilla JS (no build step), Google Identity Services button |
| Knowledge base (optional) | Pinecone / ChromaDB + sentence-transformers, scikit-learn TF-IDF fallback |
| Tests | pytest (37 tests, all use fakes and never call Google/Gemini) |
| Deploy (original) | Backend on Render (`render.yaml`), frontend on GitHub Pages (`.github/workflows/static.yml`) |

---

## 3. How it runs locally (Windows)

Project folder: `D:\Gen Ai\pdftoexcel`, Python virtualenv at `.venv`.

| Process | Command (run inside project folder) | URL |
|---|---|---|
| Backend API | `.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8030` | http://127.0.0.1:8030 (docs at `/docs`) |
| Frontend (static files, no-cache) | `.\.venv\Scripts\python.exe tools\no_cache_static_server.py` | http://127.0.0.1:5000/index.html |
| Local demo (no Google, fake services) | `.\.venv\Scripts\python.exe demo_server.py` | http://127.0.0.1:8031/index.html |
| Tests | `.\.venv\Scripts\python.exe -m pytest -q` | n/a |
| Google resource check (read-only) | `.\.venv\Scripts\python.exe tools\bootstrap_google_resources.py --check` | n/a |

The frontend chooses its backend automatically: on `localhost`/`127.0.0.1` it calls `http://127.0.0.1:8030`; anywhere else it calls the production Render URL (`https://pdftoexcel-846x.onrender.com`). The same logic is in `index.html`, `accounting.html`, `ocr.html` and `session_timeout.js`.

### Configuration (`.env` in project folder, loaded by `python-dotenv` in `app/core/config.py`)

| Variable | Purpose | Local status |
|---|---|---|
| `GOOGLE_CLIENT_ID` / `VITE_GOOGLE_CLIENT_ID` | OAuth client for Google Sign-In (backend verifies ID token audience) | set |
| `ALLOWED_EMAIL_DOMAIN` | default `nobroker.in` | set |
| `ALLOW_DOMAIN_WIDE_ACCESS` | if true, any `@nobroker.in` user not in User_Master may log in as USER | **not set (off)** |
| `GOOGLE_SERVICE_ACCOUNT_FILE` or `GOOGLE_SERVICE_ACCOUNT_JSON` | service account for Sheets + Drive | set (file `service-account.json`, git-ignored) |
| `GOOGLE_ACCOUNTING_SPREADSHEET_ID` | the single shared spreadsheet holding all tabs | set |
| `GOOGLE_DRIVE_ROOT_FOLDER_ID` | Drive root folder (only used by the health check/bootstrap) | set |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini access | key set, model default |
| `AI_VERIFICATION_MAX_RETRIES` | repair attempts (default 2) | 2 |
| `MAX_FILE_SIZE_MB` | upload limit (default 10) | default |
| `SESSION_INACTIVITY_SECONDS` (1200), `SESSION_HEARTBEAT_GRACE_SECONDS` (120) | session expiry and active-time tracking | default |
| `CORS_ALLOWED_ORIGINS` | comma list; default includes localhost:5000/8000/5500 and nobrokerhood.github.io | default |
| `GOOGLE_*_SHEET_ID` (USER_MASTER, LOGIN_AUDIT, SESSION_LOG, ACTIVITY_LOG, PROCESSING_LOG, TEMPLATE_MASTER, FOLDER_CONFIG, MAPPING_MASTER, API_USAGE) | optional legacy per-table spreadsheet overrides (if set, that table is read from `Sheet1` of that spreadsheet instead) | not set |
| `PINECONE_API_KEY`, `PINECONE_INDEX`, `PINECONE_ENVIRONMENT` | knowledge base vector search | set, but pinecone package not installed |

---

## 4. Repository map (every file)

```
pdftoexcel/
├── main.py                      # Entry point: logging + app = create_app()
├── app/
│   ├── app_factory.py           # Builds FastAPI app; wires ALL services onto app.state; CORS; API-usage middleware; mounts routers (+ optional kb router)
│   ├── core/
│   │   ├── config.py            # Settings dataclass from env vars (lru_cached get_settings())
│   │   └── errors.py            # ServiceNotConfiguredError, 503 helper
│   ├── api/                     # HTTP routers
│   │   ├── auth.py              # /auth/google-login, /auth/me, /auth/heartbeat, /auth/logout
│   │   ├── config.py            # /config/public, /config/purposes, /config/template/{p}, /config/folder-route, /config/health (ADMIN)
│   │   ├── processing.py        # /processing/* : jobs create/list/get, corrections, mapping, approve, reject, download
│   │   ├── legacy_tools.py      # /process-document/, /export-to-excel/, /split-pdf/  (NO auth)
│   │   └── audit.py             # /login-log (legacy, NO auth)
│   ├── auth/
│   │   ├── google_auth.py       # GoogleTokenVerifier: verifies Google ID token (iss, aud, email_verified)
│   │   ├── user_master.py       # UserMasterService.authorize(): looks up email in User_Master tab; roles USER/REVIEWER/ADMIN
│   │   ├── sessions.py          # SessionService: IN-MEMORY sessions, bearer tokens, inactivity expiry, heartbeat active time
│   │   └── dependencies.py      # require_session FastAPI dependency (Authorization: Bearer <token>)
│   ├── workflows/
│   │   ├── accounting_graph.py  # AccountingWorkflow: LangGraph nodes prepare→extract→verify→(repair)→map→validate→human_review; approve_and_complete()
│   │   ├── routing.py           # conditional edge functions
│   │   └── state.py             # AccountingWorkflowState TypedDict
│   ├── agents/
│   │   ├── extractor.py         # ExtractionAgent + GeminiExtractionProvider; source_parts() converts PDF pages→images (pdf2image, 120 dpi)
│   │   ├── verifier.py          # VerificationAgent + GeminiVerificationProvider (independent check, per-field status/confidence/evidence)
│   │   └── repair.py            # RepairAgent + GeminiRepairProvider (fix only mismatched fields)
│   ├── services/
│   │   └── gemini_client.py     # GeminiDocumentClient: generate_content / generate_json (response_mime_type JSON), 2 retries
│   ├── accounting/
│   │   ├── purposes.py          # MEMBER_RECEIPT, VENDOR_INVOICE definitions
│   │   ├── schemas.py           # Pydantic: MemberReceiptExtraction, VendorInvoiceExtraction, VerificationResult, MappingResult, ValidationResult, HumanCorrection
│   │   ├── templates.py         # Built-in templates (column lists) + TemplateRegistryService (Template_Master tab overrides code/name/version)
│   │   ├── folders.py           # FolderConfigService/FolderRouterService: Folder_Config tab → Drive folder per purpose & status
│   │   ├── mapping.py           # MappingMasterService: Mapping_Master lookups (exact, normalized, alias with "|")
│   │   ├── validation.py        # AccountingValidationService: required fields & amount rules → PASSED/BLOCKED
│   │   └── output.py            # TemplateOutputGenerator: builds XLSX with openpyxl
│   ├── google/
│   │   ├── sheets_service.py    # GoogleSheetsService (gspread): read tables (60s cache), append, update by key, ensure_table
│   │   ├── sheet_schemas.py     # Tab names + exact header rows for all 9 tables
│   │   ├── sheets.py            # GoogleSheetsAuditClient: legacy login audit + API usage rows
│   │   └── drive_service.py     # GoogleDriveService: validate folder, upload, move, download, create folder
│   ├── processing/
│   │   ├── jobs.py              # ProcessingJob dataclass (all statuses, data, output bytes) + simple JobRepository
│   │   ├── stores.py            # InMemoryProcessingJobStore & GoogleSheetsProcessingJobStore (persists job JSON to Job_State tab)
│   │   └── log_lifecycle.py     # ProcessingLifecycleService: updates job + Processing_Log row + Job_State on every step
│   ├── audit/
│   │   └── activity.py          # AuditLogService (Login_Audit, Session_Log, Activity_Log) & ProcessingLogService
│   └── documents/
│       ├── conversion.py        # Legacy Gemini ledger→CSV and table→Excel conversion
│       └── pdf_splitter.py      # Split PDF into N-page parts, return ZIP
├── kb/                          # Knowledge Bot (RAG)
│   ├── kb_service.py            # POST /kb-query (Pinecone → Chroma → kb_store.json TF-IDF → Gemini answer), GET /kb-status
│   ├── embeddings.py            # build/query Chroma & Pinecone with sentence-transformers all-MiniLM-L6-v2
│   ├── upload_to_pinecone.py    # script to create index & upload
│   ├── build_embeddings.py      # script to build from kb_store.json
│   ├── kb-config.json           # 12 sample society FAQ documents
│   └── KB_README.md, KB_USAGE.md
├── index.html                   # Login page (Google button → /auth/google-login; stores token in sessionStorage; link to local demo)
├── accounting.html              # Main app UI: purpose select, upload, progress, review panels, corrections, mappings, approve/reject/download, admin Config Health
├── ocr.html                     # Legacy converter + PDF splitter UI; loads knowledge-bot-v2.js
├── voice.html                   # Voice-to-Excel UI; posts audio to external https://voice-backend-83ht.onrender.com
├── session_timeout.js           # Heartbeat every N sec to /auth/heartbeat, auto-logout on inactivity
├── knowledge-bot-v2.js          # Floating chat widget; searches ./kb_store.json in the browser (does NOT call /kb-query)
├── demo_server.py               # Port 8031: same app with fake Sheets/Drive/AI, /demo/login, /demo/sample/{purpose}
├── demo_services.py             # Fake services & sample records used by demo
├── tools/
│   ├── bootstrap_google_resources.py  # --check / --bootstrap / --create-folders / --mock
│   └── no_cache_static_server.py      # static server on 127.0.0.1:5000 with no-cache headers
├── tests/
│   ├── test_accounting_workflow.py    # happy paths, repair loop, retry exhaustion, mapping, edits, reject, audit sequence, restart restore
│   ├── test_google_foundation.py      # auth, sessions, user master, folders, templates, drive, config health
│   ├── test_legacy_tools.py           # split-pdf, missing-Gemini behaviour
│   └── test_live_google_tools.py      # Gemini client schema, bootstrap mock checks
├── audit/                       # EMPTY legacy files (audit_logger.py, middleware.py, sheets_client.py). Unused.
├── docs/                        # Design docs: AI_EXTRACTION, AI_VERIFICATION, GEMINI_PROVIDER, GOOGLE_AUTH, GOOGLE_DRIVE_STRUCTURE,
│                                #   GOOGLE_RESOURCE_BOOTSTRAP, GOOGLE_SHEETS_STRUCTURE, HUMAN_REVIEW, LANGGRAPH_WORKFLOW, LIVE_GOOGLE_SETUP,
│                                #   LOCAL_DEVELOPMENT, PROCESSING_LIFECYCLE, SESSION_TRACKING, TEMPLATE_REGISTRY, WORKFLOW_PERSISTENCE, ACCOUNTING_AI_REUSE_PLAN
├── README.md, LOCAL_PROJECT_GUIDE.md
├── requirements.txt, render.yaml, start-demo.cmd, .gitignore
└── .env, service-account.json   # local secrets (git-ignored)
```

---

## 5. Authentication & sessions

1. `index.html` loads `/config/public` for the Google client ID and renders the Google Sign-In button.
2. Google returns an ID token; the page POSTs `{credential}` to **`/auth/google-login`**.
3. `GoogleTokenVerifier.verify` checks signature, issuer, audience == `GOOGLE_CLIENT_ID`, `email_verified`.
4. `UserMasterService.authorize(email)` finds the row in the **User_Master** tab (case-insensitive):
   - found and Active truthy → allowed with Role (USER/REVIEWER/ADMIN; anything else becomes USER)
   - found but inactive → 403 "User is inactive."
   - not found → allowed only if `ALLOW_DOMAIN_WIDE_ACCESS` is true and the email ends with `@nobroker.in`; else 403 **"User is not authorized."**
5. `SessionService.create_session` makes a random bearer token. **Sessions live in memory only**, so a backend restart logs everyone out.
6. The frontend stores `accounting_session_token` in `sessionStorage` and sends `Authorization: Bearer <token>`.
7. `session_timeout.js` sends heartbeats; sessions expire after 1200 s without a heartbeat.
8. Login, logout and session snapshots are appended to Login_Audit, Session_Log and Activity_Log.
9. Roles: only **ADMIN** is enforced, for `/config/health`. REVIEWER is defined but not enforced anywhere. Any user can approve their own jobs; users can only see their own jobs (`job.user_email == session.email`).

---

## 6. The accounting workflow in detail

### 6.1 HTTP endpoints (`app/api/processing.py`, all require a session)

| Method & path | What it does |
|---|---|
| `POST /processing/validate-config` `{purpose, destination_status}` | Checks template + folder route exist for purpose |
| `POST /processing/jobs` (multipart `purpose`, `file`) | Validates type (pdf/jpeg/png) and size → resolves template + 4 Drive folders → creates job → uploads source to **Incoming** folder → **runs LangGraph synchronously until human review** → returns job JSON |
| `GET /processing/jobs` | List current user's jobs (from Job_State tab) |
| `GET /processing/jobs/{id}` | Job details + `progress[]` checklist |
| `POST /processing/jobs/{id}/corrections` `{corrections:{field:value}}` | Human edits → records HumanCorrection → re-validates |
| `POST /processing/jobs/{id}/mapping` `{resolutions:[{type, source_value, target_value}]}` | Human confirms codes (BANK, BILL_HEAD, VENDOR, EXPENSE) → re-validates. **Not written back to Mapping_Master.** |
| `POST /processing/jobs/{id}/approve` | Re-validate; 409 `VALIDATION_BLOCKED` or `MAPPING_REQUIRED` if not OK; else generate XLSX → upload to **Output** folder → move source to **Completed** folder → COMPLETED |
| `POST /processing/jobs/{id}/reject` | Marks REJECTED, no output |
| `GET /processing/jobs/{id}/download` | Streams XLSX (from memory, or re-downloads from Drive after restart) |

### 6.2 LangGraph (`app/workflows/accounting_graph.py`)

```
START → prepare → extract → verify ─┬─ PASSED ───────────────→ map ─┬─ NEEDS_MAPPING → human_review → END
                                    ├─ failed & attempts left → repair → verify
                                    ├─ extraction FAILED ─────────────→ human_review
                                    └─ retries exhausted ─────────────→ human_review
                                                                   map ─ MAPPED → validate → human_review → END
```
- `route_after_verification`: `extraction_attempt <= max_retries` → repair.
- Each node calls `ProcessingLifecycleService.update()`, which updates the in-memory job, the Processing_Log row and the Job_State JSON.
- The graph keeps live job objects in `AccountingWorkflow._jobs` (in memory). Source file bytes are **not** persisted (Job_State restore sets `source_bytes=b""`).
- If Gemini throws during extraction, the job goes to `overall_status=FAILED` with `last_error` and still ends at human_review.

### 6.3 AI agents (all Gemini; each call sends prompt + page images)
- **Extraction**: "Extract only values directly supported… null for missing… one JSON object", including purpose, template code, canonical fields and JSON keys. The result is coerced through the Pydantic schema.
- **Verification**: an independent agent returns `{overall_status: PASSED|FAILED|NEEDS_REVIEW, fields:[{field, extracted_value, verified_value, status: VERIFIED|MISMATCH|NOT_FOUND|UNCERTAIN, confidence, evidence, page_number}]}`.
- **Repair**: fixes only mismatched fields using the previous extraction and the verification result.
- PDFs are rasterized with **pdf2image at 120 dpi, which requires Poppler (`pdftoppm`) on PATH**. Images are opened directly.
- One upload costs at least 2 Gemini calls (extract + verify) and up to 6 with 2 repairs.

### 6.4 Data fields

**MEMBER_RECEIPT** (JSON key → Excel column of template `NBH_MEMBER_RECEIPT_V1`):
`payment_type`→Payment Type, `bank_name_or_code`→Society Bank Name/Bank code, `reference_number`→Cheque/Ref No, `tower`→Tower No, `flat`→Flat No, `bill_head`→Bill Head, `amount`→Amount, `transaction_date`→Transaction Date, `comments`→Comments, `meter_number`→Meter No, `cheque_issuer_bank`→Cheque Issuer Bank, `cheque_date`→Cheque Date.
Validation (all CRITICAL): amount > 0, transaction_date, tower, flat, reference_number, bank_name_or_code, bill_head required.

**VENDOR_INVOICE** (template `NBH_VENDOR_BILL_V1`, one Excel row per expense):
`bill_number`, `bill_date`, `vendor_code`, `vendor_name` (used for mapping only), `due_date`, `narration`, `cgst_amount`, `sgst_amount`, `igst_amount`, `tds_amount`, `expenses[]` of `{expense_code, expense_description, expense_amount}`.
Columns: Bill Number, Bill Date, Vendor Code, Due Date, Narration, CGST Amount, SGST Amount, IGST Amount, TDS Amount, Expense Code, Expense Amount.
Validation: bill_number, bill_date, vendor_code required; at least one expense; each expense needs a code and amount > 0; taxes ≥ 0.

### 6.5 Mapping (`app/accounting/mapping.py`)
Mapping_Master rows are filtered by Purpose, Mapping Type and Active. A value matches on exact Source Value, then on normalized value (lowercase, collapsed spaces), then on Alias (split by `|`); the result is the Canonical Code.
- MEMBER_RECEIPT: BANK (bank_name_or_code), BILL_HEAD (bill_head), TOWER_FLAT (optional, "tower flat" → flat)
- VENDOR_INVOICE: VENDOR (vendor_name → vendor_code, if no code extracted), EXPENSE (expense_description → expense_code)
- Any unmapped value → `NEEDS_MAPPING` → human must resolve before approval.

---

## 7. Google Sheets "database" (one spreadsheet, 9 tabs + Sheet1)

Headers must match exactly (`app/google/sheet_schemas.py`):

| Tab | Headers | Written/read by |
|---|---|---|
| User_Master | Email, Name, Role, Active, Created At, Updated At | read at login |
| Login_Audit | Session ID, Email, Name, Login Time, Logout Time, Login Status, IP, User Agent | login/logout |
| Session_Log | Session ID, Email, Login At, Last Seen At, Logout At, Session Duration Seconds, Active Duration Seconds, Status | login, heartbeat, logout (appends a snapshot each time) |
| Activity_Log | Timestamp, Session ID, User Email, Job ID, Action, Purpose, Source File ID, Output File ID, Status, Details | every action (LOGIN, PURPOSE_SELECTED, FILE_UPLOAD, EXTRACTION_*, AI_VERIFICATION_*, MAPPING_*, HUMAN_EDIT, HUMAN_APPROVED/REJECTED, EXCEL_GENERATED, FILE_DOWNLOADED, LOGOUT) |
| Processing_Log | Job ID, Session ID, User Email, Purpose, Template Code, Source Filename, Source Drive File ID, Source Folder ID, Extraction Status, Verification Status, Mapping Status, Validation Status, Human Status, Output Filename, Output Drive File ID, Overall Status, Started At, Completed At | one row per job, updated per step |
| Template_Master | Purpose, Template Code, Template Name, Version, Output Format, Active | overrides template metadata (columns stay built-in) |
| Folder_Config | Purpose, Incoming Folder ID, Review Folder ID, Completed Folder ID, Output Folder ID, Active | Drive routing per purpose |
| Mapping_Master | Mapping Type, Purpose, Society, Source Value, Normalized Source Value, Canonical Code, Canonical Name, Alias, Active, Created By, Created At, Updated By, Updated At | code lookups |
| Job_State | Job ID, Workflow ID, Current Step, Overall Status, State JSON, Updated At | job persistence / restore after restart |

Current live data: User_Master has about 699 rows (mostly `@nobroker.in` USER, plus some rows literally named "anonymous"; the first data rows are blank). One ADMIN has been added for the local developer. Template_Master has 2 rows, Folder_Config has 2 rows, **Mapping_Master is EMPTY**. `tools/bootstrap_google_resources.py --check` reports PASS for all tabs and the Drive root.

Google Drive: per purpose, 4 folders (Incoming → Completed for the source file, Output for generated XLSX; Review is resolved but files are never moved there).

---

## 8. Other endpoints

| Path | Auth | Notes |
|---|---|---|
| `GET /` | none | health message |
| `POST /process-document/` | **none** | Gemini ledger image/PDF → flattened CSV (legacy prompt) |
| `POST /export-to-excel/` | **none** | Gemini table extraction per page → XLSX, unified columns |
| `POST /split-pdf/?pages_per_file=5` | **none** | PyPDF2 split → ZIP (no Gemini) |
| `POST /login-log` | **none** | legacy login audit append |
| `POST /kb-query` `{query, top_k, use_rag}` | **none** | RAG answer |
| `GET /kb-status` | **none** | which KB backend is available |
| Middleware | n/a | logs each request to API usage sheet **only if** `GOOGLE_API_USAGE_SHEET_ID` is set (it isn't); email comes from the client-supplied `X-User-Email` header |

---

## 9. Current state & known issues (verified locally on 2026-09-14)

**Working**
- Backend, frontend and demo servers start; `/config/public` OK; Google Sign-In token verification OK.
- Service account can read/write the spreadsheet and Drive; all tabs pass the header check.
- 36 of 37 tests pass.
- The local demo (port 8031) works end-to-end with fake AI.

**Blocking / broken right now**
1. **Gemini quota**: the configured key is valid, but Google returns `429 RESOURCE_EXHAUSTED – project has exceeded its monthly spending cap`. All AI extraction/verification/legacy conversion fails until the cap is raised or the key changes.
2. **Poppler not found on PATH**: `pdf2image` raises `PDFInfoNotInstalledError`, so **PDF uploads fail at extraction** even with a working Gemini key. JPG/PNG uploads avoid this. Fix: install Poppler for Windows and add its `bin` to PATH (or pass `poppler_path` in `source_parts()` / `conversion.py`).
3. **Mapping_Master is empty**, so every real job ends in `NEEDS_MAPPING`; users must type codes manually on each job, and these are not saved back to the sheet.
4. Test `tests/test_legacy_tools.py::test_gemini_endpoints_report_missing_configuration` fails locally because `.env` now provides a Gemini key (the test expects no key → 503 but gets 400). The problem is the test environment leaking into the test, not the app.

**Design limitations / risks worth knowing**
- Sessions and live workflow objects are in memory, so a restart logs users out. Jobs can be restored from Job_State, but without the source bytes.
- `POST /processing/jobs` runs the whole AI pipeline synchronously inside an `async` endpoint using blocking SDK calls, which blocks the event loop; long uploads can time out.
- Google Sheets quota: `update_row_by_key` reads the whole tab, then calls `update_cell` once per column, and lifecycle updates run on every workflow step. This means many API calls per job and a real risk of hitting the Sheets 60-writes/min quota under load.
- Appending with `gspread.append_row` lets Google auto-detect the table. On tabs with blank leading rows (User_Master) this has put values in the wrong columns before.
- Legacy tool and KB endpoints have **no authentication**; `ocr.html`/`voice.html` only guard client-side via `sessionStorage`.
- The REVIEWER role isn't enforced; the uploader can approve their own job.
- Member receipt template says Output Format "CSV", but the system always generates XLSX.
- Dates stay as strings, so Excel date number formats have no effect on them.
- Knowledge Bot: `kb_store.json` doesn't exist (only `kb/kb-config.json`); the pinecone/chromadb/sentence-transformers packages aren't installed; the browser widget searches `./kb_store.json` directly and never calls `/kb-query`. The KB is effectively non-functional locally.
- `voice.html` depends on an external Render service that isn't part of this repo.
- `audit/` folder contains empty files; README references another developer's path (`C:\Users\virub\...`).
- `requirements.txt` lists KB packages that aren't installed in `.venv`.

---

## 10. Constraints Claude Code should respect
- Never print, commit or move secrets (`.env`, `service-account.json`, `pdftoexcel.env` in the parent folder). They are git-ignored.
- Keep Google login and User_Master authorization intact unless explicitly asked; don't enable domain-wide access silently.
- Writes to the live Google Sheet/Drive affect real shared data. Prefer read-only checks, and state clearly when a change writes to Google.
- Keep tests passing (`python -m pytest -q`); tests use fakes from `demo_services.py`-style injections via `create_app(...)` parameters.
- After backend code or `.env` changes, restart Uvicorn on port 8030 (no `--reload` is used locally).
- The frontend has no build step; edit the HTML/JS directly and hard-refresh (the static server sends no-cache headers).

---

## 11. Template for the prompt you (ChatGPT) should produce

```
Project: NoBrokerHood Accounting AI at D:\Gen Ai\pdftoexcel (FastAPI + LangGraph + Gemini + Google Sheets/Drive; static HTML frontend).
Goal: <one sentence>
Current behaviour: <what happens, exact error text, which page/endpoint, steps to reproduce>
Expected behaviour: <what should happen>
Likely files: <e.g. app/api/processing.py, app/workflows/accounting_graph.py, accounting.html>
Constraints: don't expose secrets; keep Google login; don't break tests; say before writing to the live Google Sheet/Drive.
Verify by: <pytest test to add/run, curl call, or UI steps on http://127.0.0.1:5000/accounting.html>
Deliverable: make the change, restart servers if needed, and summarise what changed and how it was verified.
```
