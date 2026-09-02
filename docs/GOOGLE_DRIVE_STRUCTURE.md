# Google Drive Structure

Drive routing is folder-ID based. The application must not route files by
folder display name.

Environment:

- `GOOGLE_DRIVE_ROOT_FOLDER_ID`

Folder IDs are loaded per purpose from `Accounting_AI_Folder_Config`.

Initial purposes:

- `MEMBER_RECEIPT`
- `VENDOR_INVOICE`

Initial workflow statuses:

- `incoming`
- `review`
- `completed`
- `output`

`GoogleDriveService` supports lazy authentication, folder validation, safe
filename handling, source upload, output upload, file move, and metadata lookup.

The vertical slice stores the original source in Incoming, uploads approved XLSX
output to Output, and moves the original source to Completed after successful
output upload.
