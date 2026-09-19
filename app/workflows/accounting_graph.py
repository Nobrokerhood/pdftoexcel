"""Accounting workflow: prepare -> extract -> verify -> [repair once] -> map ->
validate -> human_review; then human approval -> export -> Drive.

Single source of truth: `job.extracted_data` is the current document (rows,
evidence, ledger, reconciliation). Mapping and reviewer actions update it in
place; nothing keeps a second diverging copy.

Approval requires every blocking review item to be resolved (by the evidence or
by a human), a balanced source-candidate ledger, and valid mandatory fields.
It does NOT require the model verifier to have said PASSED: a human who has
confirmed each flagged row has resolved it.
"""

import logging

from langgraph.graph import END, START, StateGraph

from app.accounting.mapping import MappingMasterService
from app.accounting.output import OutputGenerationError, TemplateOutputGenerator
from app.accounting.reconciliation import AccountingReconciliationService
from app.accounting.validation import AccountingValidationService
from app.agents.extractor import ExtractionAgent
from app.agents.repair import RepairAgent, repair_requests
from app.agents.verifier import VerificationAgent, apply_verification
from app.audit.activity import AuditLogService
from app.core.config import Settings
from app.google.drive_service import GoogleDriveError
from app.processing.jobs import ProcessingJob
from app.processing.log_lifecycle import ProcessingLifecycleService
from app.workflows.state import AccountingWorkflowState

logger = logging.getLogger(__name__)


class ApprovalBlockedError(ValueError):
    def __init__(self, code: str, details: list[str] | None = None):
        super().__init__(code)
        self.code = code
        self.details = details or []


def _warnings_of(data: dict) -> list[str]:
    out = []
    if data.get("provider_note"):
        out.append(data["provider_note"])
    arb = data.get("arbitration") or {}
    for failure in arb.get("failures") or []:
        out.append(f"arbitration: {failure}")
    return out


class AccountingWorkflow:
    def __init__(self, settings: Settings, extraction_agent: ExtractionAgent, verification_agent: VerificationAgent,
                 repair_agent: RepairAgent, mapping_service: MappingMasterService,
                 validation_service: AccountingValidationService, output_generator: TemplateOutputGenerator,
                 template_registry_service, folder_router_service, drive_service,
                 lifecycle_service: ProcessingLifecycleService, audit_log_service: AuditLogService):
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

    # -- graph ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(AccountingWorkflowState)
        for name in ("prepare", "extract", "verify", "repair", "map", "validate", "human_review"):
            graph.add_node(name, getattr(self, f"_{name}"))
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "extract")
        graph.add_conditional_edges("extract", self._after_extract, {"verify": "verify", "human_review": "human_review"})
        graph.add_conditional_edges("verify", self._after_verify, {"repair": "repair", "map": "map"})
        graph.add_edge("repair", "verify")
        graph.add_edge("map", "validate")
        graph.add_edge("validate", "human_review")
        graph.add_edge("human_review", END)
        return graph.compile()

    def run_until_review(self, job: ProcessingJob) -> ProcessingJob:
        self._jobs[job.job_id] = job
        try:
            self.graph.invoke({"job_id": job.job_id, "repair_attempts": 0})
        finally:
            self._jobs.pop(job.job_id, None)  # never retain finished jobs here (memory)
        return job

    def _job(self, state) -> ProcessingJob:
        return self._jobs[state["job_id"]]

    def _template(self, job: ProcessingJob):
        return self.template_registry_service.get_active_template(job.purpose)

    def _activity(self, job: ProcessingJob, action: str, status: str = "OK", detail: str = ""):
        self.audit_log_service.activity(job.session_id, job.user_email, job.job_id, action, job.purpose,
                                        job.source_drive_file_id, job.output_drive_file_id, status, detail)

    def _persist(self, job: ProcessingJob, **fields):
        self.lifecycle_service.update(job, **fields)

    # -- nodes ------------------------------------------------------------------
    def _prepare(self, state):
        job = self._job(state)
        self._persist(job, current_step="PREPARE", overall_status="PROCESSING")
        return {}

    def _extract(self, state):
        job = self._job(state)
        stage = job.stage_start("OCR_AND_EXTRACTION", "Reading the document (OCR engines, AI extraction, arbitration)")
        self._persist(job, current_step="EXTRACTING")
        self._activity(job, "EXTRACTION_STARTED")
        try:
            data = self.extraction_agent.extract(job.source_bytes, job.purpose, self._template(job))
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            job.stage_end(stage, "FAILED", "Extraction failed", errors=[message])
            job.extraction_status = "FAILED"
            job.last_error = message
            self._persist(job, extraction_status="FAILED", overall_status="FAILED", last_error=message)
            self._activity(job, "EXTRACTION_FAILED", "FAIL", message)
            return {"failed": True}
        job.extracted_data = data
        job.extraction_attempt += 1
        job.extraction_status = "COMPLETED"
        job.extraction_provider = data.get("_extraction_provider") or "UNKNOWN"
        ledger = data.get("candidate_ledger") or {}
        timing = data.get("timing_ms") or {}
        job.stage_end(stage, "COMPLETED",
                      f"{len(data.get('rows') or [])} transaction row(s); {ledger.get('equation', '')}".strip("; "),
                      warnings=_warnings_of(data), provider=job.extraction_provider)
        if timing:
            stage["timing_ms"] = timing
        self._persist(job, extraction_status="COMPLETED", extraction_provider=job.extraction_provider)
        self._activity(job, "EXTRACTION_COMPLETED")
        return {"failed": False}

    def _after_extract(self, state) -> str:
        return "human_review" if state.get("failed") else "verify"

    def _verify(self, state):
        job = self._job(state)
        stage = job.stage_start("VERIFICATION", "Verifying every row by row_id against source evidence",
                                attempt=state.get("repair_attempts", 0) + 1)
        self._persist(job, current_step="VERIFYING")
        self._activity(job, "AI_VERIFICATION_STARTED")
        try:
            result = self.verification_agent.verify(job.source_bytes, job.purpose, self._template(job), job.extracted_data)
        except Exception as exc:
            # Verification failure is never PASSED: rows stay unverified for review.
            from app.agents.verifier import adapt_verification
            result = adapt_verification({"notes": [f"verification unavailable: {type(exc).__name__}"]}, job.extracted_data)
            job.stage_end(stage, "NEEDS_REVIEW", "Verification unavailable; rows require review",
                          errors=[type(exc).__name__])
        else:
            job.stage_end(stage, "COMPLETED" if result.overall_status == "PASSED" else "NEEDS_REVIEW",
                          f"{sum(1 for s in result.rows.values() if s == 'VERIFIED')}/{len(result.rows)} rows verified",
                          warnings=result.notes, provider=result.provider)
        apply_verification(job.extracted_data, result)
        job.verification_result = result.model_dump(mode="json")
        job.verification_status = result.overall_status
        job.verification_provider = result.provider
        self._persist(job, verification_status=result.overall_status, verification_provider=result.provider)
        self._activity(job, "AI_VERIFICATION_PASSED" if result.overall_status == "PASSED" else "AI_VERIFICATION_FAILED",
                       "OK" if result.overall_status == "PASSED" else "FAIL")
        return {}

    def _after_verify(self, state) -> str:
        job = self._job(state)
        if job.verification_status == "PASSED" or state.get("repair_attempts", 0) >= 1:
            return "map"
        if self.settings.ai_verification_max_retries <= 0:
            return "map"
        return "repair" if repair_requests(job.extracted_data) else "map"

    def _repair(self, state):
        job = self._job(state)
        stage = job.stage_start("REPAIR", "Field-level repair of flagged fields")
        try:
            repaired = self.repair_agent.repair(job.source_bytes, job.purpose, self._template(job), job.extracted_data)
            log = repaired.get("repair_log") or {}
            job.extracted_data = repaired
            job.extracted_data["reconciliation"] = AccountingReconciliationService().reconcile(repaired, job.purpose)
            job.stage_end(stage, "COMPLETED",
                          f"{len(log.get('applied') or [])} field(s) proposed, {len(log.get('rejected') or [])} rejected")
        except Exception as exc:
            job.stage_end(stage, "SKIPPED", "Repair unavailable; flagged fields stay for human review",
                          errors=[type(exc).__name__])
        self._persist(job, current_step="REPAIR")
        return {"repair_attempts": state.get("repair_attempts", 0) + 1}

    def _map(self, state):
        job = self._job(state)
        stage = job.stage_start("MAPPING", "Mapping bank / bill head / vendor codes")
        try:
            result = self.mapping_service.map_data(job.purpose, job.extracted_data)
            job.extracted_data = result.mapped_data
            job.mapping_status = result.status
            job.mapping_result = {"status": result.status, "missing": [m.model_dump() for m in result.missing]}
            job.stage_end(stage, "COMPLETED" if result.status == "MAPPED" else "NEEDS_REVIEW",
                          f"{len(result.missing)} unmapped value(s); source values will be exported" if result.missing
                          else "all values mapped")
        except Exception as exc:
            job.mapping_status = "NEEDS_MAPPING"
            job.mapping_result = {"status": "NEEDS_MAPPING", "missing": [], "error": type(exc).__name__}
            job.stage_end(stage, "NEEDS_REVIEW", "Mapping unavailable; source values will be exported",
                          errors=[type(exc).__name__])
        self._persist(job, mapping_status=job.mapping_status, current_step="MAPPING")
        return {}

    def _validate(self, state):
        job = self._job(state)
        stage = job.stage_start("VALIDATION", "Deterministic accounting validation and reconciliation")
        result = self.validation_service.validate(job.purpose, job.extracted_data)
        job.validation_result = result.model_dump(mode="json")
        job.validation_status = result.status
        blocking = sum(1 for i in result.issues if i.severity == "CRITICAL")
        job.stage_end(stage, "COMPLETED" if result.status == "PASSED" else "NEEDS_REVIEW",
                      f"{blocking} blocking, {len(result.issues) - blocking} advisory issue(s)")
        self._persist(job, validation_status=result.status, current_step="VALIDATING")
        return {}

    def _human_review(self, state):
        job = self._job(state)
        job.human_status = "NEEDS_REVIEW"
        if job.overall_status != "FAILED" and job.extraction_status != "FAILED":
            job.overall_status = "NEEDS_REVIEW"
            entry = job.stage_start("HUMAN_REVIEW", "Waiting for reviewer")
            entry["status"] = "WAITING"
        else:
            job.overall_status = "FAILED"
        job.current_step = "HUMAN_REVIEW"
        self._persist(job, human_status="NEEDS_REVIEW", overall_status=job.overall_status, current_step="HUMAN_REVIEW")
        return {}

    # -- human actions ----------------------------------------------------------
    def revalidate(self, job: ProcessingJob):
        job.extracted_data["reconciliation"] = AccountingReconciliationService().reconcile(job.extracted_data, job.purpose)
        result = self.validation_service.validate(job.purpose, job.extracted_data)
        job.validation_result = result.model_dump(mode="json")
        job.validation_status = result.status
        return result

    def approval_blockers(self, job: ProcessingJob) -> list[str]:
        result = self.revalidate(job)
        return [f"{i.code}: {i.message}" for i in result.issues if i.severity == "CRITICAL" and not i.resolution]

    def approve_and_complete(self, job: ProcessingJob) -> ProcessingJob:
        if job.overall_status != "NEEDS_REVIEW":
            raise ApprovalBlockedError("JOB_NOT_AWAITING_REVIEW")
        blockers = self.approval_blockers(job)
        if blockers:
            raise ApprovalBlockedError("REVIEW_ITEMS_UNRESOLVED", blockers)

        for entry in job.stage_trace:
            if entry.get("stage") == "HUMAN_REVIEW" and entry.get("status") == "WAITING":
                job.stage_end(entry, "COMPLETED", f"Approved by {job.user_email}")
        job.human_status = "APPROVED"
        self._activity(job, "HUMAN_APPROVED")
        export = job.stage_start("EXPORT", "Generating the NBH workbook")
        try:
            job.output_filename, job.output_bytes = self.output_generator.generate_xlsx(
                job.purpose, self._template(job), job.extracted_data, job.job_id,
                audit=[c.model_dump() for c in job.human_corrections])
        except OutputGenerationError as exc:
            job.stage_end(export, "FAILED", "Workbook generation refused", errors=[str(exc)])
            job.human_status = "NEEDS_REVIEW"
            raise ApprovalBlockedError("OUTPUT_INVALID", [str(exc)]) from exc
        job.stage_end(export, "COMPLETED", job.output_filename)
        self._persist(job, human_status="APPROVED", overall_status="GENERATING_OUTPUT", current_step="GENERATING_OUTPUT")

        upload = job.stage_start("DRIVE_UPLOAD", "Uploading the workbook to Output and moving the source to Completed")
        try:
            job.output_drive_file_id = self.drive_service.upload_file(
                job.output_filename, job.output_bytes, job.output_folder_id,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self._activity(job, "EXCEL_GENERATED", detail=job.output_filename)
            if job.source_drive_file_id:
                self.drive_service.move_file(job.source_drive_file_id, job.completed_folder_id)
        except GoogleDriveError as exc:
            job.stage_end(upload, "FAILED", "Drive upload failed; the workbook is still downloadable", errors=[str(exc)])
            job.overall_status = "FAILED"
            job.last_error = str(exc)
            self._persist(job, overall_status="FAILED", last_error=str(exc))
            raise
        job.stage_end(upload, "COMPLETED", "Workbook in Output; source moved to Completed")
        job.overall_status = "COMPLETED"
        job.current_step = "COMPLETE"
        self._persist(job, output_filename=job.output_filename, output_drive_file_id=job.output_drive_file_id,
                      overall_status="COMPLETED", current_step="COMPLETE")
        return job
