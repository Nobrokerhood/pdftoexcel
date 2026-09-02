# Google Sheets Structure

All Google Sheet access goes through `GoogleSheetsService`.

## Accounting_AI_User_Master

Columns:

- Email
- Name
- Role
- Active
- Created At
- Updated At

Roles:

- USER
- REVIEWER
- ADMIN

## Accounting_AI_Login_Audit

Columns:

- Session ID
- Email
- Name
- Login Time
- Logout Time
- Login Status
- IP
- User Agent

## Accounting_AI_Session_Log

Columns:

- Session ID
- Email
- Login At
- Last Seen At
- Logout At
- Session Duration Seconds
- Active Duration Seconds
- Status

## Accounting_AI_Activity_Log

Columns:

- Timestamp
- Session ID
- User Email
- Job ID
- Action
- Purpose
- Source File ID
- Output File ID
- Status
- Details

Wired actions include login/session/config actions plus:

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
- FILE_DOWNLOADED

## Accounting_AI_Processing_Log

Columns:

- Job ID
- Session ID
- User Email
- Purpose
- Template Code
- Source Filename
- Source Drive File ID
- Source Folder ID
- Extraction Status
- Verification Status
- Mapping Status
- Validation Status
- Human Status
- Output Filename
- Output Drive File ID
- Overall Status
- Started At
- Completed At

## Accounting_AI_Template_Master

Columns:

- Purpose
- Template Code
- Template Name
- Version
- Output Format
- Active

## Accounting_AI_Folder_Config

Columns:

- Purpose
- Incoming Folder ID
- Review Folder ID
- Completed Folder ID
- Output Folder ID
- Active
