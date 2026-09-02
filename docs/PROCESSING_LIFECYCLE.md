# Processing Lifecycle

Processing jobs are created through `POST /processing/jobs`.

Lifecycle:

1. CREATED
2. UPLOADING
3. PROCESSING
4. NEEDS_REVIEW
5. APPROVED
6. GENERATING_OUTPUT
7. COMPLETED

Failure states:

- FAILED
- REJECTED
- NEEDS_REVIEW for mapping or human validation work

Source files are uploaded to the configured Incoming folder before LangGraph
starts. Output files are uploaded to the configured Output folder only after
human approval. The original source is moved to Completed after successful
output upload where Drive semantics allow it.

The current local repository uses an in-memory job store for the working slice.
Google Sheets remains the operational log/config layer.
