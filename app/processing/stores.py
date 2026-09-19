"""Job state persistence.

* In memory: a bounded LRU of live jobs (source/output bytes are released once a
  job is finished; Drive holds the files).
* Google Sheets `Job_State`: one row per job. State JSON is zlib-compressed and
  base64-encoded (`z1:` prefix). A state too large for one cell is written to a
  JSON file in the job's Review folder and the cell holds `drive:<file id>`.
  State is never truncated, and a state that cannot be loaded raises instead of
  being replaced by a stub (the previous loader silently returned a job with no
  extracted data when the 45k-character cell limit cut the JSON).
* A Sheets failure never corrupts the in-memory job; it is logged and retried on
  the next update, so a processing result stays recoverable.
"""

import base64
import json
import logging
import threading
import zlib
from collections import OrderedDict
from datetime import datetime
from typing import Protocol

from app.google.sheets_service import GoogleSheetsService
from app.processing.jobs import ProcessingJob

logger = logging.getLogger(__name__)

MAX_CELL_CHARS = 45000
MAX_LIVE_JOBS = 64
TERMINAL = {"COMPLETED", "REJECTED"}


class ProcessingJobStore(Protocol):
    def create(self, session_id, user_email, purpose, template_code, source_filename, source_content_type,
               source_bytes) -> ProcessingJob: ...
    def get(self, job_id: str) -> ProcessingJob: ...
    def update_job(self, job: ProcessingJob) -> bool: ...
    def list_jobs(self, email: str) -> list[ProcessingJob]: ...
    def save_state(self, job: ProcessingJob) -> bool: ...


def encode_state(state: dict) -> str:
    raw = json.dumps(state, default=str, separators=(",", ":")).encode("utf-8")
    return "z1:" + base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def decode_state(text: str) -> dict:
    if text.startswith("z1:"):
        return json.loads(zlib.decompress(base64.b64decode(text[3:])).decode("utf-8"))
    return json.loads(text)


class InMemoryProcessingJobStore:
    def __init__(self, max_live_jobs: int = MAX_LIVE_JOBS):
        self._jobs: OrderedDict[str, ProcessingJob] = OrderedDict()
        self._states: dict[str, dict] = {}
        self._lock = threading.RLock()
        self.max_live_jobs = max_live_jobs

    def _remember(self, job: ProcessingJob):
        with self._lock:
            self._jobs[job.job_id] = job
            self._jobs.move_to_end(job.job_id)
            if job.overall_status in TERMINAL:
                job.source_bytes = b""  # the source lives in Drive; do not pin it in memory
            while len(self._jobs) > self.max_live_jobs:
                old_id, old = self._jobs.popitem(last=False)
                self._states[old_id] = job_to_state(old)

    def create(self, session_id, user_email, purpose, template_code, source_filename, source_content_type,
               source_bytes) -> ProcessingJob:
        from uuid import uuid4
        job = ProcessingJob(job_id=str(uuid4()), session_id=session_id, user_email=user_email, purpose=purpose,
                            template_code=template_code, source_filename=source_filename,
                            source_content_type=source_content_type, source_bytes=source_bytes)
        self._remember(job)
        self.save_state(job)
        return job

    create_job = create

    def get(self, job_id: str) -> ProcessingJob:
        with self._lock:
            if job_id in self._jobs:
                self._jobs.move_to_end(job_id)
                return self._jobs[job_id]
        state = self.load_state(job_id)
        job = job_from_state(state)
        self._remember(job)
        return job

    def update_job(self, job: ProcessingJob) -> bool:
        self._remember(job)
        return self.save_state(job)

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        with self._lock:
            return [job for job in self._jobs.values() if job.user_email == email]

    def list_jobs(self, email: str) -> list[ProcessingJob]:
        return self.list_for_user(email)

    def save_state(self, job: ProcessingJob) -> bool:
        self._states[job.job_id] = job_to_state(job)
        return True

    def load_state(self, job_id: str) -> dict:
        try:
            return dict(self._states[job_id])
        except KeyError as exc:
            raise KeyError("JOB_NOT_FOUND") from exc


class GoogleSheetsProcessingJobStore(InMemoryProcessingJobStore):
    def __init__(self, sheets_service: GoogleSheetsService, drive_service=None):
        super().__init__()
        self.sheets_service = sheets_service
        self.drive_service = drive_service
        self._state_files: dict[str, str] = {}

    def create(self, *args, **kwargs) -> ProcessingJob:
        job = super().create(*args, **kwargs)
        self._write(job)
        return job

    create_job = create

    def update_job(self, job: ProcessingJob) -> bool:
        self._remember(job)
        self._states[job.job_id] = job_to_state(job)
        return self._write(job)

    def save_state(self, job: ProcessingJob) -> bool:
        self._states[job.job_id] = job_to_state(job)
        return self._write(job)

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        jobs: dict[str, ProcessingJob] = {j.job_id: j for j in super().list_for_user(email)}
        try:
            records = self.sheets_service.read_table("job_state")
        except Exception as exc:
            logger.warning("Job list from Sheets unavailable: %s", type(exc).__name__)
            return list(jobs.values())
        for record in records:
            job_id = str(record.get("Job ID", ""))
            if not job_id or job_id in jobs:
                continue
            try:
                state = self._decode_record(record)
            except Exception:
                continue  # an unreadable historical row is skipped, not shown as a stub
            if str(state.get("user_email", "")).strip().lower() == str(email).strip().lower():
                jobs[job_id] = job_from_state(state)
        return list(jobs.values())

    def load_state(self, job_id: str) -> dict:
        if job_id in self._states:
            return dict(self._states[job_id])
        record = self.sheets_service.lookup_table_row_by_key("job_state", "Job ID", job_id)
        if not record:
            raise KeyError("JOB_NOT_FOUND")
        return self._decode_record(record)

    def _decode_record(self, record: dict) -> dict:
        text = str(record.get("State JSON", "") or "")
        if text.startswith("drive:"):
            if self.drive_service is None:
                raise KeyError("JOB_STATE_IN_DRIVE_UNAVAILABLE")
            return decode_state(self.drive_service.download_file(text[6:]).decode("ascii"))
        if text.endswith("...[TRUNCATED]"):
            raise KeyError("JOB_STATE_TRUNCATED")  # legacy rows; never replaced by a stub
        return decode_state(text)

    def _write(self, job: ProcessingJob) -> bool:
        encoded = encode_state(job_to_state(job))
        cell = encoded
        if len(encoded) > MAX_CELL_CHARS:
            cell = self._write_drive(job, encoded)
            if cell is None:
                return False
        updates = {"Workflow ID": job.job_id, "Current Step": job.current_step, "Overall Status": job.overall_status,
                   "State JSON": cell, "Updated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

        def write():
            if self.sheets_service.update_table_row_by_key("job_state", "Job ID", job.job_id, updates):
                return True
            return self.sheets_service.append_table_row(
                "job_state", [job.job_id, job.job_id, job.current_step, job.overall_status, cell, updates["Updated At"]])

        from app.google.sheets_service import submit_write
        # Coalesced per job: a burst of status changes becomes one write of the latest state.
        return submit_write(self.sheets_service, write, coalesce_key=f"job_state:{job.job_id}")

    def _write_drive(self, job: ProcessingJob, encoded: str) -> str | None:
        if self.drive_service is None or not (job.review_folder_id or job.source_folder_id):
            logger.warning("Job state for %s exceeds one cell and no Drive folder is available.", job.job_id)
            return None
        try:
            existing = self._state_files.get(job.job_id)
            data = encoded.encode("ascii")
            if existing and hasattr(self.drive_service, "update_file_content"):
                self.drive_service.update_file_content(existing, data, "application/octet-stream")
                return f"drive:{existing}"
            file_id = self.drive_service.upload_file(f"job_state_{job.job_id}.json.z", data,
                                                     job.review_folder_id or job.source_folder_id,
                                                     "application/octet-stream")
            self._state_files[job.job_id] = file_id
            return f"drive:{file_id}"
        except Exception as exc:
            logger.warning("Job state for %s could not be written to Drive (%s).", job.job_id, type(exc).__name__)
            return None


def job_to_state(job: ProcessingJob) -> dict:
    state = job.summary()
    state.pop("source_bytes", None)
    state["workflow_id"] = job.job_id
    state["source_content_type"] = job.source_content_type
    return state


def job_from_state(state: dict) -> ProcessingJob:
    from app.accounting.schemas import HumanCorrection

    corrections = [item if isinstance(item, HumanCorrection) else HumanCorrection(**item)
                   for item in state.get("human_corrections", [])]
    job = ProcessingJob(
        job_id=state["job_id"],
        session_id=state.get("session_id", ""),
        user_email=state.get("user_email", ""),
        purpose=state.get("purpose", ""),
        template_code=state.get("template_code", ""),
        source_filename=state.get("source_filename", ""),
        source_content_type=state.get("source_content_type", "application/pdf"),
        source_bytes=b"",
        source_drive_file_id=state.get("source_drive_file_id", ""),
        source_folder_id=state.get("source_folder_id", ""),
        review_folder_id=state.get("review_folder_id", ""),
        completed_folder_id=state.get("completed_folder_id", ""),
        output_folder_id=state.get("output_folder_id", ""),
        extraction_status=state.get("extraction_status", "NOT_STARTED"),
        verification_status=state.get("verification_status", "NOT_STARTED"),
        mapping_status=state.get("mapping_status", "NOT_STARTED"),
        validation_status=state.get("validation_status", "NOT_STARTED"),
        human_status=state.get("human_status", "PENDING"),
        output_filename=state.get("output_filename", ""),
        output_drive_file_id=state.get("output_drive_file_id", ""),
        overall_status=state.get("overall_status", "CREATED"),
        current_step=state.get("current_step", "CREATED"),
        last_error=state.get("last_error", ""),
        extraction_attempt=state.get("extraction_attempt", 0),
        extracted_data=state.get("extracted_data", {}),
        verification_result=state.get("verification_result", {}),
        mapping_result=state.get("mapping_result", {}),
        validation_result=state.get("validation_result", {}),
        human_corrections=corrections,
        extraction_provider=state.get("extraction_provider", ""),
        verification_provider=state.get("verification_provider", ""),
        output_bytes=None,
        started_at=state.get("started_at", ""),
        completed_at=state.get("completed_at", ""),
        file_sha256=state.get("file_sha256", ""),
        stage_trace=list(state.get("stage_trace") or []),
    )
    return job
