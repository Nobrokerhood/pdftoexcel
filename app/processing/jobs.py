"""Processing jobs, their stage trace, and the background runner.

Stage trace: every stage the backend actually runs is recorded with state,
start/end time, duration, provider, message, warnings and errors. The UI renders
only this trace; it never invents progress.

Execution: `JobRunner` runs the workflow off the request thread in a bounded
pool, so an upload returns immediately after the file is stored and the client
polls the job. `inline` mode (used by tests) runs synchronously.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from app.accounting.schemas import HumanCorrection

logger = logging.getLogger(__name__)

STAGES = ("UPLOAD", "OCR_AND_EXTRACTION", "VERIFICATION", "REPAIR", "MAPPING", "VALIDATION", "HUMAN_REVIEW",
          "EXPORT", "DRIVE_UPLOAD")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class ProcessingJob:
    job_id: str
    session_id: str
    user_email: str
    purpose: str
    template_code: str
    source_filename: str
    source_content_type: str
    source_bytes: bytes
    source_drive_file_id: str = ""
    source_folder_id: str = ""
    review_folder_id: str = ""
    completed_folder_id: str = ""
    output_folder_id: str = ""
    extraction_status: str = "NOT_STARTED"
    verification_status: str = "NOT_STARTED"
    mapping_status: str = "NOT_STARTED"
    validation_status: str = "NOT_STARTED"
    human_status: str = "PENDING"
    output_filename: str = ""
    output_drive_file_id: str = ""
    overall_status: str = "CREATED"
    current_step: str = "CREATED"
    last_error: str = ""
    extraction_attempt: int = 0
    extracted_data: dict[str, Any] = field(default_factory=dict)
    verification_result: dict[str, Any] = field(default_factory=dict)
    mapping_result: dict[str, Any] = field(default_factory=dict)
    validation_result: dict[str, Any] = field(default_factory=dict)
    human_corrections: list[HumanCorrection] = field(default_factory=list)
    extraction_provider: str = ""
    verification_provider: str = ""
    output_bytes: bytes | None = None
    started_at: str = field(default_factory=now_text)
    completed_at: str = ""
    file_sha256: str = ""
    stage_trace: list[dict] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # -- stage trace ----------------------------------------------------------
    def stage_start(self, stage: str, message: str = "", provider: str = "", attempt: int = 1) -> dict:
        with self._lock:
            entry = {"stage": stage, "status": "RUNNING", "started_at": _iso(), "completed_at": None,
                     "duration_ms": None, "message": message, "provider": provider, "attempt": attempt,
                     "warnings": [], "errors": [], "_t0": time.perf_counter()}
            self.stage_trace.append(entry)
            return entry

    def stage_end(self, entry: dict, status: str, message: str = "", warnings=None, errors=None, provider: str = ""):
        with self._lock:
            entry["status"] = status
            entry["completed_at"] = _iso()
            t0 = entry.pop("_t0", None)
            entry["duration_ms"] = int((time.perf_counter() - t0) * 1000) if t0 else None
            if message:
                entry["message"] = message
            if provider:
                entry["provider"] = provider
            entry["warnings"] = list(warnings or entry.get("warnings") or [])[:20]
            entry["errors"] = [str(e)[:300] for e in (errors or entry.get("errors") or [])][:10]

    def public_trace(self) -> list[dict]:
        with self._lock:
            return [{k: v for k, v in e.items() if not k.startswith("_")} for e in self.stage_trace]

    def summary(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "session_id": self.session_id,
            "user_email": self.user_email,
            "purpose": self.purpose,
            "template_code": self.template_code,
            "source_filename": self.source_filename,
            "source_content_type": self.source_content_type,
            "source_drive_file_id": self.source_drive_file_id,
            "source_folder_id": self.source_folder_id,
            "review_folder_id": self.review_folder_id,
            "completed_folder_id": self.completed_folder_id,
            "output_folder_id": self.output_folder_id,
            "extraction_status": self.extraction_status,
            "verification_status": self.verification_status,
            "mapping_status": self.mapping_status,
            "validation_status": self.validation_status,
            "human_status": self.human_status,
            "output_filename": self.output_filename,
            "output_drive_file_id": self.output_drive_file_id,
            "overall_status": self.overall_status,
            "current_step": self.current_step,
            "last_error": self.last_error,
            "extraction_attempt": self.extraction_attempt,
            "extraction_provider": self.extraction_provider,
            "verification_provider": self.verification_provider,
            "extracted_data": self.extracted_data,
            "verification_result": self.verification_result,
            "mapping_result": self.mapping_result,
            "validation_result": self.validation_result,
            "human_corrections": [item.model_dump() for item in self.human_corrections],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "file_sha256": self.file_sha256,
            "stage_trace": self.public_trace(),
        }


class JobRepository:
    """In-memory job registry (legacy test helper)."""

    def __init__(self):
        self._jobs: dict[str, ProcessingJob] = {}

    def create(self, session_id, user_email, purpose, template_code, source_filename, source_content_type,
               source_bytes) -> ProcessingJob:
        job = ProcessingJob(
            job_id=str(uuid4()), session_id=session_id, user_email=user_email, purpose=purpose,
            template_code=template_code, source_filename=source_filename,
            source_content_type=source_content_type, source_bytes=source_bytes,
        )
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> ProcessingJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError("JOB_NOT_FOUND") from exc

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        return [job for job in self._jobs.values() if job.user_email == email]


class JobRunner:
    """Runs a job's workflow either inline or in a bounded background pool."""

    def __init__(self, mode: str = "background", max_workers: int = 2):
        self.mode = mode if mode in ("background", "inline") else "background"
        self._pool = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="nbh-job") \
            if self.mode == "background" else None
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def submit(self, job_id: str, fn: Callable[[], None]) -> None:
        with self._lock:
            if job_id in self._active:
                return
            self._active.add(job_id)

        def run():
            try:
                fn()
            except Exception:  # the workflow records its own failure; this is a last resort
                logger.exception("Background job %s crashed", job_id)
            finally:
                with self._lock:
                    self._active.discard(job_id)

        if self._pool is None:
            run()
        else:
            self._pool.submit(run)

    def is_active(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._active

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)
