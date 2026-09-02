# Live Google Setup

This project now uses one shared Google Spreadsheet for Accounting AI configuration and logs.

Default spreadsheet name:

`Accounting_AI_Config_and_Logs`

Required tabs:

- `User_Master`
- `Login_Audit`
- `Session_Log`
- `Activity_Log`
- `Processing_Log`
- `Template_Master`
- `Folder_Config`
- `Mapping_Master`
- `Job_State`

Put environment variables in:

`C:\Users\virub\Desktop\genai\AI_Agent\legacy_repos\pdftoexcel\.env`

Core variables:

```env
GOOGLE_CLIENT_ID=your-google-oauth-client-id
VITE_GOOGLE_CLIENT_ID=your-google-oauth-client-id
ALLOWED_EMAIL_DOMAIN=nobroker.in
GOOGLE_SERVICE_ACCOUNT_FILE=C:\path\to\service-account.json
GOOGLE_ACCOUNTING_SPREADSHEET_ID=shared-spreadsheet-id
GOOGLE_DRIVE_ROOT_FOLDER_ID=drive-root-folder-id
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
```

`GOOGLE_SERVICE_ACCOUNT_JSON` can be used instead of `GOOGLE_SERVICE_ACCOUNT_FILE`.

Legacy per-sheet IDs still work and take priority over the shared spreadsheet for that table:

```env
GOOGLE_USER_MASTER_SHEET_ID=
GOOGLE_LOGIN_AUDIT_SHEET_ID=
GOOGLE_SESSION_LOG_SHEET_ID=
GOOGLE_ACTIVITY_LOG_SHEET_ID=
GOOGLE_PROCESSING_LOG_SHEET_ID=
GOOGLE_TEMPLATE_MASTER_SHEET_ID=
GOOGLE_FOLDER_CONFIG_SHEET_ID=
GOOGLE_MAPPING_MASTER_SHEET_ID=
```

Keep `GOOGLE_API_USAGE_SHEET_ID` only if you still want the old API usage report sheet.
