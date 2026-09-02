# Workflow Persistence

Processing jobs are stored behind a `ProcessingJobStore` abstraction.

Required methods:

- `create_job`
- `get`
- `update_job`
- `list_jobs`
- `save_state`
- `load_state`

Local development uses `InMemoryProcessingJobStore` unless `GOOGLE_ACCOUNTING_SPREADSHEET_ID` is configured.

Live Google mode uses `GoogleSheetsProcessingJobStore`, which writes one row per job into the `Job_State` tab. The row stores safe workflow state JSON only. Source bytes, output bytes, credentials, OAuth tokens, and API keys are never persisted.

The LangGraph workflow saves state through `ProcessingLifecycleService` after meaningful node transitions such as prepare, extraction, verification, mapping, validation, human review, output generation, completion, rejection, and failure.

After restart:

- `GET /processing/jobs` lists jobs restored from `Job_State`.
- `GET /processing/jobs/{job_id}` loads state from `Job_State` when not present in memory.
- Approval can continue from restored human-review state.
- Completed output download uses `output_drive_file_id` and Drive download when `output_bytes` are no longer in memory.

If state JSON grows too large later, store only metadata in `Job_State` and place full state in a Drive sidecar file.
