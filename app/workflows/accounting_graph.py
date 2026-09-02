from langgraph.graph import END, START, StateGraph

from app.accounting.mapping import MappingMasterService
from app.accounting.output import TemplateOutputGenerator
from app.accounting.schemas import VerificationResult
from app.accounting.validation import AccountingValidationService
from app.agents.extractor import ExtractionAgent
from app.agents.repair import RepairAgent
from app.agents.verifier import VerificationAgent
from app.audit.activity import AuditLogService
from app.core.config import Settings
from app.google.drive_service import GoogleDriveError
from app.processing.jobs import ProcessingJob
from app.processing.log_lifecycle import ProcessingLifecycleService
from app.workflows.routing import (
    route_after_mapping,
    route_after_validation,
    route_after_verification,
)
from app.workflows.state import AccountingWorkflowState


class AccountingWorkflow:
    def __init__(
        self,
        settings: Settings,
        extraction_agent: ExtractionAgent,
        verification_agent: VerificationAgent,
        repair_agent: RepairAgent,
        mapping_service: MappingMasterService,
        validation_service: AccountingValidationService,
        output_generator: TemplateOutputGenerator,
        template_registry_service,
        folder_router_service,
        drive_service,
        lifecycle_service: ProcessingLifecycleService,
        audit_log_service: AuditLogService,
    ):
        self.settings = settings
        self.extraction_agent = extraction_agent
        self.verification_agent = verification_agent
        self.repair_agent = repair_agent
        self.mapping_service = mapping_service
        self.validation_service = validation_service
        self.output_generator = output_generator
        self.template_registry_service = template_registry_service
        self.folder_router_service = folder_router_service
        self.drive_service = drive_service
        self.lifecycle_service = lifecycle_service
        self.audit_log_service = audit_log_service
        self.graph = self._build_graph()
        self._jobs: dict[str, ProcessingJob] = {}

    def _build_graph(self):
        graph = StateGraph(AccountingWorkflowState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("extract", self._extract)
        graph.add_node("verify", self._verify)
        graph.add_node("repair", self._repair)
        graph.add_node("map", self._map)
        graph.add_node("validate", self._validate)
        graph.add_node("human_review", self._human_review)

        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "extract")
        graph.add_edge("extract", "verify")
        graph.add_conditional_edges(
            "verify",
            route_after_verification,
            {"map": "map", "repair": "repair", "human_review": "human_review"},
        )
        graph.add_edge("repair", "verify")
        graph.add_conditional_edges(
            "map",
            route_after_mapping,
            {"validate": "validate", "human_review": "human_review"},
        )
        graph.add_conditional_edges(
            "validate",
            route_after_validation,
            {"human_review": "human_review"},
        )
        graph.add_edge("human_review", END)
        return graph.compile()

    def run_until_review(self, job: ProcessingJob) -> ProcessingJob:
        self._jobs[job.job_id] = job
        state: AccountingWorkflowState = {
            "workflow_id": job.job_id,
            "job_id": job.job_id,
            "session_id": job.session_id,
            "user_email": job.user_email,
            "source_filename": job.source_filename,
            "source_drive_file_id": job.source_drive_file_id,
            "source_type": job.source_content_type,
            "selected_purpose": job.purpose,
            "template_code": job.template_code,
            "input_folder_id": job.source_folder_id,
            "review_folder_id": job.review_folder_id,
            "completed_folder_id": job.completed_folder_id,
            "output_folder_id": job.output_folder_id,
            "extraction_attempt": job.extraction_attempt,
            "max_retries": self.settings.ai_verification_max_retries,
            "overall_status": job.overall_status,
            "current_step": job.current_step,
        }
        final_state = self.graph.invoke(state)
        self._sync_job(job, final_state)
        return job

    def _job(self, state: AccountingWorkflowState) -> ProcessingJob:
        return self._jobs[state["job_id"]]

    def _template(self, job: ProcessingJob):
        return self.template_registry_service.get_active_template(job.purpose)

    def _sync_job(self, job: ProcessingJob, state: AccountingWorkflowState):
        job.extraction_attempt = state.get("extraction_attempt", job.extraction_attempt)
        job.extracted_data = state.get("extracted_data", job.extracted_data)
        job.verification_result = state.get("verification_result", job.verification_result)
        job.verification_status = state.get("verification_status", job.verification_status)
        job.mapping_result = state.get("mapping_result", job.mapping_result)
        job.mapping_status = state.get("mapping_status", job.mapping_status)
        job.validation_result = state.get("validation_result", job.validation_result)
        job.validation_status = state.get("validation_status", job.validation_status)
        job.human_status = state.get("human_status", job.human_status)
        job.overall_status = state.get("overall_status", job.overall_status)
        job.current_step = state.get("current_step", job.current_step)
        job.last_error = state.get("last_error", job.last_error)

    def _prepare(self, state: AccountingWorkflowState):
        job = self._job(state)
        job.current_step = "PREPARE"
        job.overall_status = "PROCESSING"
        self.lifecycle_service.update(job, current_step="PREPARE", overall_status="PROCESSING")
        return {"current_step": "PREPARE", "overall_status": "PROCESSING"}

    def _extract(self, state: AccountingWorkflowState):
        job = self._job(state)
        template = self._template(job)
        self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "EXTRACTION_STARTED", job.purpose, job.source_drive_file_id, "", "OK", "")
        try:
            data = self.extraction_agent.extract(job.source_bytes, job.purpose, template)
            job.extraction_attempt += 1
            job.extracted_data = data
            job.extraction_status = "COMPLETED"
            self.lifecycle_service.update(job, extraction_status="COMPLETED", current_step="EXTRACTING")
            self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "EXTRACTION_COMPLETED", job.purpose, job.source_drive_file_id, "", "OK", "")
            return {
                "extraction_attempt": job.extraction_attempt,
                "extracted_data": data,
                "current_step": "EXTRACTING",
            }
        except Exception as exc:
            job.extraction_status = "FAILED"
            job.overall_status = "FAILED"
            job.last_error = str(exc)
            self.lifecycle_service.update(job, extraction_status="FAILED", overall_status="FAILED", last_error=str(exc))
            self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "EXTRACTION_FAILED", job.purpose, job.source_drive_file_id, "", "FAIL", str(exc))
            return {"current_step": "EXTRACTING", "overall_status": "FAILED", "last_error": str(exc)}

    def _verify(self, state: AccountingWorkflowState):
        job = self._job(state)
        if job.overall_status == "FAILED":
            return {"verification_status": "FAILED"}
        template = self._template(job)
        self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "AI_VERIFICATION_STARTED", job.purpose, job.source_drive_file_id, "", "OK", "")
        result = self.verification_agent.verify(job.source_bytes, job.purpose, template, job.extracted_data)
        job.verification_result = result.model_dump(mode="json")
        job.verification_status = result.overall_status
        self.lifecycle_service.update(job, verification_status=result.overall_status, current_step="VERIFYING")
        event = "AI_VERIFICATION_PASSED" if result.overall_status == "PASSED" else "AI_VERIFICATION_FAILED"
        self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, event, job.purpose, job.source_drive_file_id, "", "OK" if result.overall_status == "PASSED" else "FAIL", "")
        return {
            "verification_result": job.verification_result,
            "verification_status": result.overall_status,
            "extraction_attempt": job.extraction_attempt,
            "current_step": "VERIFYING",
        }

    def _repair(self, state: AccountingWorkflowState):
        job = self._job(state)
        template = self._template(job)
        repaired = self.repair_agent.repair(
            job.source_bytes,
            job.purpose,
            template,
            job.extracted_data,
            VerificationResult(**job.verification_result),
        )
        job.extraction_attempt += 1
        job.extracted_data = repaired
        self.lifecycle_service.update(job, extraction_status="REPAIRED", current_step="REPAIR_EXTRACTION")
        return {
            "extraction_attempt": job.extraction_attempt,
            "extracted_data": repaired,
            "current_step": "REPAIR_EXTRACTION",
        }

    def _map(self, state: AccountingWorkflowState):
        job = self._job(state)
        result = self.mapping_service.map_data(job.purpose, job.extracted_data)
        job.mapping_result = result.model_dump(mode="json")
        job.mapping_status = "NEEDS_MAPPING" if result.status == "NEEDS_MAPPING" else "MAPPED"
        if result.status == "NEEDS_MAPPING":
            job.overall_status = "NEEDS_REVIEW"
            self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "MAPPING_REQUIRED", job.purpose, job.source_drive_file_id, "", "NEEDS_REVIEW", "")
        self.lifecycle_service.update(job, mapping_status=job.mapping_status, current_step="MAPPING", overall_status=job.overall_status)
        return {
            "mapping_result": job.mapping_result,
            "mapping_status": job.mapping_status,
            "current_step": "MAPPING",
            "overall_status": job.overall_status,
        }

    def _validate(self, state: AccountingWorkflowState):
        job = self._job(state)
        data = job.mapping_result.get("mapped_data") or job.extracted_data
        result = self.validation_service.validate(job.purpose, data)
        job.validation_result = result.model_dump(mode="json")
        job.validation_status = result.status
        self.lifecycle_service.update(job, validation_status=result.status, current_step="VALIDATING")
        return {
            "validation_result": job.validation_result,
            "validation_status": result.status,
            "current_step": "VALIDATING",
        }

    def _human_review(self, state: AccountingWorkflowState):
        job = self._job(state)
        job.human_status = "NEEDS_REVIEW"
        if job.overall_status != "FAILED":
            job.overall_status = "NEEDS_REVIEW"
        job.current_step = "HUMAN_REVIEW"
        self.lifecycle_service.update(job, human_status="NEEDS_REVIEW", overall_status=job.overall_status, current_step="HUMAN_REVIEW")
        return {
            "human_status": "NEEDS_REVIEW",
            "overall_status": job.overall_status,
            "current_step": "HUMAN_REVIEW",
        }

    def approve_and_complete(self, job: ProcessingJob) -> ProcessingJob:
        data = job.mapping_result.get("mapped_data") or job.extracted_data
        validation = self.validation_service.validate(job.purpose, data)
        job.validation_result = validation.model_dump(mode="json")
        job.validation_status = validation.status
        if validation.status != "PASSED":
            raise ValueError("VALIDATION_BLOCKED")
        if job.mapping_status == "NEEDS_MAPPING":
            raise ValueError("MAPPING_REQUIRED")

        template = self._template(job)
        job.human_status = "APPROVED"
        self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "HUMAN_APPROVED", job.purpose, job.source_drive_file_id, "", "OK", "")
        job.output_filename, job.output_bytes = self.output_generator.generate_xlsx(
            job.purpose, template, data, job.job_id
        )
        self.lifecycle_service.update(job, human_status="APPROVED", overall_status="GENERATING_OUTPUT", current_step="GENERATING_OUTPUT")
        try:
            job.output_drive_file_id = self.drive_service.upload_file(
                job.output_filename,
                job.output_bytes,
                job.output_folder_id,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, "EXCEL_GENERATED", job.purpose, job.source_drive_file_id, job.output_drive_file_id, "OK", job.output_filename)
            self.drive_service.move_file(job.source_drive_file_id, job.completed_folder_id)
            job.overall_status = "COMPLETED"
            job.current_step = "COMPLETE"
            self.lifecycle_service.update(job, output_filename=job.output_filename, output_drive_file_id=job.output_drive_file_id, overall_status="COMPLETED", current_step="COMPLETE")
        except GoogleDriveError as exc:
            job.overall_status = "FAILED"
            job.last_error = str(exc)
            self.lifecycle_service.update(job, overall_status="FAILED", last_error=str(exc))
            raise
        return job
