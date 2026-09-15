# Local project guide

Project: D:\Gen Ai\pdftoexcel

## Open the app

- Website: http://127.0.0.1:5000/index.html
- Backend status: http://127.0.0.1:8030/
- Interactive API documentation: http://127.0.0.1:8030/docs

The frontend and backend are separate processes. Closing the browser does not stop them.

## Purpose

Accounting AI helps a housing-society accounting team turn receipts and vendor invoices into structured Excel imports. It reduces manual data entry, records processing activity, and requires a person to review the result before final output. It does not post transactions directly into an ERP.

## Main workflow

1. Sign in with Google using an authorized user.
2. Select Member Bank Receipt or Vendor Invoice.
3. Upload a PDF, JPG, JPEG, or PNG (default limit: 10 MB).
4. The backend creates a job and uploads the source to configured Google Drive folders.
5. Gemini extracts accounting fields. A verification step checks them against the source; a repair step can retry failed extraction.
6. Mapping resolves names to configured accounting codes. Validation checks required values and amounts.
7. Human Review displays the source, extracted fields, verification, mappings, and validation issues.
8. Save corrections, confirm mappings, then approve or reject. Approval is blocked by critical validation errors or missing required mappings.
9. Approved jobs generate a template-based XLSX for download. Rejected jobs do not generate output.

Member receipt fields include amount, transaction date, reference number, bank, tower, flat, and bill head. Vendor invoice fields include bill number/date, vendor code/name, expense lines, and GST/TDS amounts.

## Pages and features

| Page | Purpose | Dependencies |
| --- | --- | --- |
| index.html | Google sign-in | OAuth client and authorized user configuration |
| accounting.html | Upload, AI processing, review, approval, Excel download | Google Drive/Sheets, templates/mappings, Gemini |
| ocr.html | Legacy document-to-CSV/Excel conversion and PDF splitting | Conversion needs Gemini; splitting does not |
| voice.html | Record/upload audio and download Excel | Separate hosted voice service; it is not started by this repository |
| /docs on port 8030 | Explore/test API endpoints | Running backend |

A knowledge-base API also exists. Semantic search needs its optional vector-store/embedding dependencies and an index; AI answers need Gemini. It is not automatically ready merely because the main server runs.

## Configuration still needed for real use

Create .env in THIS project folder (the README's C:\Users\virub path belongs to the original developer).

- GOOGLE_CLIENT_ID and VITE_GOOGLE_CLIENT_ID
- GOOGLE_SERVICE_ACCOUNT_FILE (or GOOGLE_SERVICE_ACCOUNT_JSON)
- GOOGLE_ACCOUNTING_SPREADSHEET_ID
- GOOGLE_DRIVE_ROOT_FOLDER_ID
- GEMINI_API_KEY
- GEMINI_MODEL (repository default: gemini-2.5-flash)
- ALLOWED_EMAIL_DOMAIN (repository default: nobroker.in)

The shared spreadsheet holds User_Master, Login_Audit, Session_Log, Activity_Log, Processing_Log, Template_Master, Folder_Config, Mapping_Master, and Job_State. Active authorized users, templates, folder routes, and accounting mappings must be populated. Credentials and access must match those resources. See docs/LIVE_GOOGLE_SETUP.md and docs/GOOGLE_RESOURCE_BOOTSTRAP.md.

No credentials were added and no live accounting documents were uploaded during local setup. The original Google login remains enabled. The voice page sends submitted recordings to its separate Render backend.

## Run again

Open two PowerShell terminals in D:\Gen Ai\pdftoexcel.

Backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8030
```

Frontend:

```powershell
.\.venv\Scripts\python.exe -m http.server 5000 --bind 127.0.0.1
```

Press Ctrl+C in each terminal to stop a foreground server. Current background launch logs are backend.log, backend-error.log, frontend.log, and frontend-error.log.

## Code map

- main.py and app/app_factory.py: application startup and service wiring.
- app/api/: authentication, configuration, jobs, legacy tools, audit endpoints.
- app/workflows/: LangGraph processing and review workflow.
- app/agents/: extraction, verification, repair.
- app/accounting/: schemas, templates, mapping, validation, Excel generation.
- app/google/: Drive and Sheets integrations.
- app/auth/ and app/audit/: sessions, user authorization, activity logs.
- app/processing/: jobs, lifecycle, persistence.
- kb/: knowledge retrieval and AI answers.
- tests/: automated checks using fake services as well as tool tests.

## Local setup scope

A .venv was created using the installed Python interpreter with system packages available, and core backend dependencies were installed there. Optional knowledge-base model/vector database packages were not installed. PDF rasterization uses Poppler, which is already available on this machine. The checkout is a shallow copy of main at de6c3e7.


## Explore without Google

Open http://127.0.0.1:8031/index.html and choose Enter local demo. Select a purpose, click Load sample for selected purpose, edit fields, Save Edits, then Approve & Generate Excel and Download Excel. You can also reject a job. Start it again by double-clicking start-demo.cmd.

This separate demo uses the real workflow, validation and Excel generator with in-memory sample services. Extraction and AI verification are simulated, including when you upload your own file. No live Google or Gemini calls are made. Data resets on server restart. Production login is unchanged.
