from datetime import datetime

from app.audit.activity import ProcessingLogService
from app.processing.jobs import ProcessingJob


class ProcessingLifecycleService:
    def __init__(self, processing_log_service: ProcessingLogService, job_store=None):
        self.processing_log_service = processing_log_service
        self.job_store = job_store

    def create_row(self, job: ProcessingJob) -> bool:
        logged = self.processing_log_service.append_started(
            job.job_id,
            job.session_id,
            job.user_email,
            job.purpose,
            job.template_code,
            job.source_filename,
            job.source_drive_file_id,
            job.source_folder_id,
            overall_status="CREATED",
        )
        if self.job_store:
            self.job_store.save_state(job)
        return logged

    # Only these changes are persisted immediately; a pure progress step
    # (current_step) stays in memory until the next milestone, which keeps a job
    # well inside the Sheets write quota.
    MILESTONE_FIELDS = {"overall_status", "extraction_status", "verification_status", "mapping_status",
                        "validation_status", "human_status", "source_drive_file_id", "output_drive_file_id",
                        "output_filename", "last_error"}

    def update(self, job: ProcessingJob, force: bool = False, **fields) -> bool:
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        if fields.get("overall_status") in {"COMPLETED", "REJECTED", "FAILED"}:
            job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        signature = tuple(str(getattr(job, name, "")) for name in sorted(self.MILESTONE_FIELDS))
        if not force and getattr(job, "_persisted_signature", None) == signature:
            return True
        job._persisted_signature = signature
        logged = self.processing_log_service.update_job(job)
        if self.job_store:
            self.job_store.update_job(job)
        return logged
