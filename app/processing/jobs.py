from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.accounting.schemas import HumanCorrection


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    output_bytes: bytes | None = None
    started_at: str = field(default_factory=now_text)
    completed_at: str = ""

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
            "extracted_data": self.extracted_data,
            "verification_result": self.verification_result,
            "mapping_result": self.mapping_result,
            "validation_result": self.validation_result,
            "human_corrections": [item.model_dump() for item in self.human_corrections],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class JobRepository:
    def __init__(self):
        self._jobs: dict[str, ProcessingJob] = {}

    def create(
        self,
        session_id: str,
        user_email: str,
        purpose: str,
        template_code: str,
        source_filename: str,
        source_content_type: str,
        source_bytes: bytes,
    ) -> ProcessingJob:
        job = ProcessingJob(
            job_id=str(uuid4()),
            session_id=session_id,
            user_email=user_email,
            purpose=purpose,
            template_code=template_code,
            source_filename=source_filename,
            source_content_type=source_content_type,
            source_bytes=source_bytes,
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
