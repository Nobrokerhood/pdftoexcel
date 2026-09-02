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

    def update(self, job: ProcessingJob, **fields) -> bool:
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        if fields.get("overall_status") in {"COMPLETED", "REJECTED", "FAILED"}:
            job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logged = self.processing_log_service.update_job(job)
        if self.job_store:
            self.job_store.update_job(job)
        return logged
