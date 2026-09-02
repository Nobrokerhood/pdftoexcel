# Accounting AI Reuse Plan

This repo is the active application for the Accounting AI upgrade. The older
PDF to Excel, template conversion, and PDF splitter workflows must keep working
while the codebase is refactored toward agentic extraction, verification,
mapping, validation, and template generation.

## Current Application

- Backend: FastAPI in `main.py`.
- Frontend: static pages in `index.html`, `ocr.html`, and `voice.html`.
- AI extraction: Gemini image prompts for table extraction and template CSV
  conversion.
- Exports: CSV and XLSX generated with pandas/openpyxl.
- PDF utility: PyPDF2 split-to-ZIP endpoint.
- Audit: Google Sheets login and API usage reports.
- Knowledge bot: optional `kb` router with local/vector-store retrieval.

## Preserve

- `POST /process-document/` for template CSV conversion.
- `POST /export-to-excel/` for as-is Excel extraction.
- `POST /split-pdf/` for PDF splitting.
- `POST /login-log` for login audit.
- `GET /` for health.
- Static frontend entry points and the current Google sign-in flow until a
  backend token verification layer replaces it.
- KB routes as optional functionality.

## Refactor First

1. Move monolithic backend logic out of `main.py`.
2. Make Gemini and Google Sheets lazy optional services so local import,
   tests, and the PDF splitter do not require live secrets.
3. Keep existing route paths, response filenames, and payload shapes stable.
4. Add regression tests for import, health, PDF splitting, and missing Gemini
   behavior.
5. Use this platform as the home for accounting agent modules instead of
   creating another parallel product.

## Reuse From Accounting AI Reference

The `accounting-ai` reference project remains useful for:

- Confirmed import receipt and vendor bill field lineage.
- Rule-first validation ideas.
- No-fabrication extraction policy.
- Pydantic-style structured schemas.
- Fake or heuristic tests that do not call live Gemini/Google services.

Confirmed schema lineage:

- Import Receipt: Payment Type, Society Bank Name/Bank Code, Cheque/Ref No,
  Tower No, Flat No, Bill Head, Amount, Transaction Date, Comments, Meter No,
  Cheque Issuer Bank, Cheque Date.
- Vendor Bill: Bill Number, Bill Date, Vendor Code, Due Date, Narration, CGST
  Amount, SGST Amount, IGST Amount, TDS Amount, and expense rows.

`MEMBER_PAYMENT` remains blocked until the business definition and template
mapping are confirmed.

## Reject Or Delay

- Direct ERP posting.
- Production ERP connections.
- New parallel accounting applications.
- Import-time failures caused by missing Gemini or Google credentials.
- Hard-coded local service account paths.
- Treating client-side Google token decoding as backend authentication.

## Security And Ops Gaps

- Google identity is currently enforced only in the browser.
- Render/backend URLs and Google OAuth client IDs are hard-coded in frontend
  files.
- Audit logging depends on external Google Sheets availability.
- The existing remote tree contains a committed `venv`; do not expand it.
- Secrets must come from environment variables, never repo files.

## Next Accounting Modules

The current milestone adds the first agentic vertical slice for
`MEMBER_RECEIPT` and `VENDOR_INVOICE`: job creation, Drive source upload,
LangGraph orchestration, extraction/verification/repair boundaries, mapping,
validation, mandatory human review, approval/rejection, XLSX generation, Drive
output upload, and download.

Other accounting purposes, production voice, ERP posting, and autonomous
approval remain out of scope.

After this foundation, add the Accounting AI layer in stages:

1. Document type classifier.
2. Structured extraction schemas.
3. Rule validation and confidence reporting.
4. Template mapping and generated import files.
5. Human review state for low-confidence or invalid rows.
