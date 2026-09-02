import json
from datetime import datetime
from typing import Protocol

from app.google.sheets_service import GoogleSheetsService
from app.processing.jobs import ProcessingJob


class ProcessingJobStore(Protocol):
    def create_job(
        self,
        session_id: str,
        user_email: str,
        purpose: str,
        template_code: str,
        source_filename: str,
        source_content_type: str,
        source_bytes: bytes,
    ) -> ProcessingJob:
        ...

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
        ...

    def get(self, job_id: str) -> ProcessingJob:
        ...

    def update_job(self, job: ProcessingJob) -> bool:
        ...

    def list_jobs(self, email: str) -> list[ProcessingJob]:
        ...

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        ...

    def save_state(self, job: ProcessingJob) -> bool:
        ...

    def load_state(self, job_id: str) -> dict:
        ...


class InMemoryProcessingJobStore:
    def __init__(self):
        self._jobs: dict[str, ProcessingJob] = {}
        self._states: dict[str, dict] = {}

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
        from app.processing.jobs import JobRepository

        job = JobRepository().create(
            session_id,
            user_email,
            purpose,
            template_code,
            source_filename,
            source_content_type,
            source_bytes,
        )
        self._jobs[job.job_id] = job
        self.save_state(job)
        return job

    def create_job(
        self,
        session_id: str,
        user_email: str,
        purpose: str,
        template_code: str,
        source_filename: str,
        source_content_type: str,
        source_bytes: bytes,
    ) -> ProcessingJob:
        return self.create(
            session_id,
            user_email,
            purpose,
            template_code,
            source_filename,
            source_content_type,
            source_bytes,
        )

    def get(self, job_id: str) -> ProcessingJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            state = self._states.get(job_id)
            if not state:
                raise KeyError("JOB_NOT_FOUND") from exc
            job = job_from_state(state)
            self._jobs[job_id] = job
            return job

    def update_job(self, job: ProcessingJob) -> bool:
        self._jobs[job.job_id] = job
        return self.save_state(job)

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        return [job for job in self._jobs.values() if job.user_email == email]

    def list_jobs(self, email: str) -> list[ProcessingJob]:
        return self.list_for_user(email)

    def save_state(self, job: ProcessingJob) -> bool:
        self._states[job.job_id] = job_to_state(job)
        return True

    def load_state(self, job_id: str) -> dict:
        return self._states[job_id].copy()


class GoogleSheetsProcessingJobStore(InMemoryProcessingJobStore):
    def __init__(self, sheets_service: GoogleSheetsService):
        super().__init__()
        self.sheets_service = sheets_service

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
        job = super().create(
            session_id,
            user_email,
            purpose,
            template_code,
            source_filename,
            source_content_type,
            source_bytes,
        )
        self._append_or_update_state(job)
        return job

    def get(self, job_id: str) -> ProcessingJob:
        if job_id in self._jobs:
            return self._jobs[job_id]

        state = self.load_state(job_id)
        job = job_from_state(state)
        self._jobs[job_id] = job
        return job

    def update_job(self, job: ProcessingJob) -> bool:
        self._jobs[job.job_id] = job
        self._states[job.job_id] = job_to_state(job)
        return self._append_or_update_state(job)

    def list_for_user(self, email: str) -> list[ProcessingJob]:
        jobs = []
        for record in self.sheets_service.read_table("job_state"):
            state = json.loads(record.get("State JSON", "{}") or "{}")
            if state.get("user_email") == email:
                jobs.append(job_from_state(state))
        return jobs

    def list_jobs(self, email: str) -> list[ProcessingJob]:
        return self.list_for_user(email)

    def save_state(self, job: ProcessingJob) -> bool:
        self._states[job.job_id] = job_to_state(job)
        return self._append_or_update_state(job)

    def load_state(self, job_id: str) -> dict:
        record = self.sheets_service.lookup_table_row_by_key(
            "job_state", "Job ID", job_id
        )
        if not record:
            raise KeyError("JOB_NOT_FOUND")
        return json.loads(record.get("State JSON", "{}") or "{}")

    def _append_or_update_state(self, job: ProcessingJob) -> bool:
        state = job_to_state(job)
        state_json = json.dumps(state, default=str, separators=(",", ":"))
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updates = {
            "Workflow ID": job.job_id,
            "Current Step": job.current_step,
            "Overall Status": job.overall_status,
            "State JSON": state_json,
            "Updated At": updated_at,
        }
        updated = self.sheets_service.update_table_row_by_key(
            "job_state", "Job ID", job.job_id, updates
        )
        if updated:
            return True
        return self.sheets_service.append_table_row(
            "job_state",
            [
                job.job_id,
                job.job_id,
                job.current_step,
                job.overall_status,
                state_json,
                updated_at,
            ],
        )


def job_to_state(job: ProcessingJob) -> dict:
    state = job.summary()
    state.pop("source_bytes", None)
    state["workflow_id"] = job.job_id
    state["source_content_type"] = job.source_content_type
    return state


def job_from_state(state: dict) -> ProcessingJob:
    from app.accounting.schemas import HumanCorrection

    corrections = [
        item if isinstance(item, HumanCorrection) else HumanCorrection(**item)
        for item in state.get("human_corrections", [])
    ]
    return ProcessingJob(
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
        output_bytes=None,
        started_at=state.get("started_at", ""),
        completed_at=state.get("completed_at", ""),
    )
