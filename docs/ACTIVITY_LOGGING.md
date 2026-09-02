# Activity Logging

`AuditLogService` centralizes login, logout, session snapshots, and activity
append operations.

Initial fully wired actions:

- LOGIN
- LOGOUT
- PURPOSE_SELECTED

Reserved future workflow actions:

- FILE_UPLOAD
- EXTRACTION_STARTED
- EXTRACTION_COMPLETED
- EXTRACTION_FAILED
- AI_VERIFICATION_STARTED
- AI_VERIFICATION_PASSED
- AI_VERIFICATION_FAILED
- MAPPING_REQUIRED
- MAPPING_CONFIRMED
- HUMAN_EDIT
- HUMAN_APPROVED
- HUMAN_REJECTED
- EXCEL_GENERATED
- CSV_GENERATED
- FILE_DOWNLOADED

Tokens, service-account JSON, and Gemini keys must never be logged.
