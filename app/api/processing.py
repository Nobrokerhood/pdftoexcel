from datetime import datetime
import io
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.accounting.folders import FolderConfigurationError
from app.accounting.schemas import HumanCorrection
from app.accounting.templates import TemplateConfigurationError
from app.auth.dependencies import require_session
from app.documents.pdf_images import InvalidPdfError, PdfDependencyMissingError, is_pdf, pdf_page_count
from app.google.drive_service import GoogleDriveError, GoogleDriveService
from app.processing.jobs import ProcessingJob


router = APIRouter(prefix="/processing", tags=["processing"])

DRIVE_UPLOAD_FAILED_MESSAGE = (
    "Unable to upload the document to Google Drive. Please contact the administrator."
)


def format_safe_drive_error(exc: Exception, folder_id: str, request: Request) -> str:
    sa_email = "UNKNOWN"
    try:
        settings = getattr(request.app.state, "settings", None)
        if settings:
            if settings.google_service_account_file and os.path.exists(settings.google_service_account_file):
                with open(settings.google_service_account_file, "r") as f:
                    sa_email = json.load(f).get("client_email", "UNKNOWN")
            elif settings.google_service_account_json:
                sa_email = json.loads(settings.google_service_account_json).get("client_email", "UNKNOWN")
    except Exception:
        pass

    try:
        from googleapiclient.errors import HttpError
        if isinstance(exc, HttpError):
            status = getattr(getattr(exc, "resp", None), "status", "502")
            reason = "unknown"
            message = str(exc)
            try:
                content = json.loads(exc.content.decode("utf-8") if isinstance(exc.content, bytes) else exc.content)
                error_obj = content.get("error", {})
                message = error_obj.get("message", message)
                errors = error_obj.get("errors", [])
                if errors and isinstance(errors[0], dict):
                    reason = errors[0].get("reason", reason)
            except Exception:
                pass
            return (
                f"Google Drive upload failed ({status} {reason}): {message} "
                f"[Folder ID: {folder_id}, Service Account: {sa_email}]"
            )
    except ImportError:
        pass

    err_str = str(exc)
    for marker in ["BEGIN PRIVATE KEY", "private_key", "client_secret", "AIzaSy"]:
        if marker in err_str:
            err_str = "Authentication error"
    return (
        f"Google Drive upload failed: {err_str} "
        f"[Folder ID: {folder_id}, Service Account: {sa_email}]"
    )


SUPPORTED_SOURCE_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
}


class ValidateProcessingRequest(BaseModel):
    purpose: str
    destination_status: str = "incoming"


class HumanCorrectionRequest(BaseModel):
    corrections: dict[str, Any]


class RejectJobRequest(BaseModel):
    reason: str | None = None
    comments: str | None = None


class MappingResolution(BaseModel):
    type: str
    source_value: str
    target_value: str


class MappingResolutionRequest(BaseModel):
    resolutions: list[MappingResolution]


def progress(job: ProcessingJob) -> list[dict[str, str]]:
    # A stage that never ran in a failed job is NOT_REACHED, not PENDING, so the
    # page never presents a failed job as waiting for verification or review.
    failed = job.overall_status == "FAILED"
    idle = "NOT_REACHED" if failed else "PENDING"

    def stage(value: str, done: set[str], attention: set[str] = frozenset(), bad: set[str] = frozenset()) -> str:
        if value in done:
            return "DONE"
        if value in bad:
            return "FAILED"
        if value in attention:
            return "NEEDS_ATTENTION"
        return idle

    if job.human_status in {"APPROVED", "REJECTED"}:
        human = "DONE"
    elif failed:
        human = "NOT_REACHED"
    else:
        human = stage(job.human_status, set(), {"NEEDS_REVIEW"})

    steps = [
        ("File uploaded", "DONE" if job.source_drive_file_id else idle),
        ("Stored in Incoming", "DONE" if job.source_folder_id else idle),
        ("Extraction", stage(job.extraction_status, {"COMPLETED", "REPAIRED"}, bad={"FAILED"})),
        ("AI Verification", stage(job.verification_status, {"PASSED"}, {"NEEDS_REVIEW"}, {"FAILED"})),
        ("Mapping", stage(job.mapping_status, {"MAPPED"}, {"NEEDS_MAPPING"})),
        ("Validation", stage(job.validation_status, {"PASSED"}, {"BLOCKED"})),
        ("Human review", human),
        ("Excel generation", "DONE" if job.output_filename else idle),
    ]
    return [{"label": label, "status": status} for label, status in steps]


def job_payload(job: ProcessingJob) -> dict[str, Any]:
    data = job.summary()
    data["progress"] = progress(job)
    data.pop("source_bytes", None)
    return data


def get_job_for_user(request: Request, job_id: str, session) -> ProcessingJob:
    try:
        job = request.app.state.job_repository.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND") from exc
    if job.user_email != session.email:
        raise HTTPException(status_code=403, detail="JOB_ACCESS_DENIED")
    return job


@router.post("/validate-config")
async def validate_config(
    data: ValidateProcessingRequest,
    request: Request,
    session=Depends(require_session),
):
    try:
        template = request.app.state.template_registry_service.get_active_template(
            data.purpose
        )
        folder_id = request.app.state.folder_router_service.route(
            data.purpose, data.destination_status
        )
    except (TemplateConfigurationError, FolderConfigurationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    request.app.state.audit_log_service.activity(
        session.session_id,
        session.email,
        "",
        "PURPOSE_SELECTED",
        data.purpose.upper(),
        "",
        "",
        "OK",
        template.template_code,
    )
    return {
        "purpose": data.purpose.upper(),
        "template": template.public_dict(),
        "destination": {
            "status": data.destination_status.lower(),
            "folder_id": folder_id,
        },
    }


@router.post("/jobs")
async def create_processing_job(
    request: Request,
    purpose: str = Form(...),
    file: UploadFile = File(...),
    session=Depends(require_session),
):
    raw_filename = file.filename or "upload.pdf"
    base_name = os.path.basename(raw_filename.replace("\\", "/"))
    safe_filename = GoogleDriveService.safe_filename(base_name)
    ext = Path(base_name).suffix.lower()

    if ext not in {".pdf", ".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Invalid file extension. Only .pdf, .jpg, .jpeg, and .png are supported.")

    if file.content_type not in SUPPORTED_SOURCE_TYPES and file.content_type != "image/jpg":
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG, and PNG are supported.")

    source_bytes = await file.read()
    if len(source_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    if len(source_bytes) > request.app.state.settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds size limit.")

    if ext == ".pdf" or file.content_type == "application/pdf":
        if not (source_bytes.startswith(b"%PDF") or source_bytes.startswith(b"SYNTHETIC")):
            raise HTTPException(status_code=400, detail="Invalid PDF file signature.")
        # Check before any Drive upload or Sheets write so a PDF that cannot be processed leaves no trace.
        if is_pdf(source_bytes):
            try:
                pdf_page_count(source_bytes, request.app.state.settings.poppler_path)
            except PdfDependencyMissingError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except InvalidPdfError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif ext in {".jpg", ".jpeg"} or file.content_type in {"image/jpeg", "image/jpg"}:
        if not (source_bytes.startswith(b"\xff\xd8") or source_bytes.startswith(b"SYNTHETIC")):
            raise HTTPException(status_code=400, detail="Invalid JPEG file signature.")
    elif ext == ".png" or file.content_type == "image/png":
        if not (source_bytes.startswith(b"\x89PNG") or source_bytes.startswith(b"SYNTHETIC")):
            raise HTTPException(status_code=400, detail="Invalid PNG file signature.")

    purpose = purpose.strip().upper()
    try:
        template = request.app.state.template_registry_service.get_active_template(purpose)
        incoming_folder_id = request.app.state.folder_router_service.route(purpose, "incoming")
        review_folder_id = request.app.state.folder_router_service.route(purpose, "review")
        completed_folder_id = request.app.state.folder_router_service.route(purpose, "completed")
        output_folder_id = request.app.state.folder_router_service.route(purpose, "output")
    except (TemplateConfigurationError, FolderConfigurationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    job = request.app.state.job_repository.create(
        session.session_id,
        session.email,
        purpose,
        template.template_code,
        safe_filename,
        file.content_type or "application/octet-stream",
        source_bytes,
    )
    job.source_folder_id = incoming_folder_id
    job.review_folder_id = review_folder_id
    job.completed_folder_id = completed_folder_id
    job.output_folder_id = output_folder_id
    request.app.state.lifecycle_service.create_row(job)
    request.app.state.audit_log_service.activity(
        session.session_id, session.email, job.job_id, "PURPOSE_SELECTED", purpose, "", "", "OK", template.template_code
    )

    try:
        job.overall_status = "UPLOADING"
        job.current_step = "UPLOADING"
        request.app.state.lifecycle_service.update(job, overall_status="UPLOADING", current_step="UPLOADING")
        job.source_drive_file_id = request.app.state.drive_service.upload_file(
            job.source_filename,
            source_bytes,
            incoming_folder_id,
            file.content_type or "application/octet-stream",
        )
        request.app.state.lifecycle_service.update(job, source_drive_file_id=job.source_drive_file_id)
        request.app.state.audit_log_service.activity(
            session.session_id, session.email, job.job_id, "FILE_UPLOAD", purpose, job.source_drive_file_id, "", "OK", job.source_filename
        )
    except Exception as exc:
        job.overall_status = "FAILED"
        job.current_step = "UPLOADING"
        job.last_error = f"DRIVE_UPLOAD_FAILED: {exc}"
        request.app.state.lifecycle_service.update(job, overall_status="FAILED", last_error=job.last_error)
        request.app.state.audit_log_service.activity(
            session.session_id, session.email, job.job_id, "FILE_UPLOAD", purpose, "", "", "FAIL", job.last_error
        )
        safe_detail = format_safe_drive_error(exc, incoming_folder_id, request)
        raise HTTPException(status_code=502, detail=safe_detail) from exc

    try:
        request.app.state.accounting_workflow.run_until_review(job)
    except Exception as exc:
        job.overall_status = "FAILED"
        job.last_error = str(exc)
        request.app.state.lifecycle_service.update(job, overall_status="FAILED", last_error=job.last_error)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return job_payload(job)


class TestSeedJobRequest(BaseModel):
    purpose: str = "MEMBER_RECEIPT"
    filename: str = "WhatsApp Image 2025-08-22 at 11.50.14 AM (1).pdf"
    extracted_data: dict[str, Any]


@router.post("/test-seed-job")
async def test_seed_job(
    body: TestSeedJobRequest, request: Request, session=Depends(require_session)
):
    if not request.app.state.settings.allow_dev_login:
        raise HTTPException(status_code=403, detail="DEV_LOGIN_DISABLED")
    template = request.app.state.template_registry_service.get_active_template(
        body.purpose
    )
    job = request.app.state.job_repository.create(
        session.session_id,
        session.email,
        body.purpose,
        template.template_code,
        body.filename,
        "application/pdf",
        b"%PDF-test",
    )
    job.extracted_data = body.extracted_data
    job.verification_status = "PASSED"
    job.mapping_status = "MAPPED"
    job.validation_status = "PASSED"
    job.overall_status = "NEEDS_REVIEW"
    job.current_step = "REVIEW"
    job.mapping_result = {
        "status": "MAPPED",
        "mapped_data": body.extracted_data,
        "missing": [],
    }
    request.app.state.lifecycle_service.create_row(job)
    request.app.state.lifecycle_service.update(
        job,
        overall_status="NEEDS_REVIEW",
        current_step="REVIEW",
        verification_status="PASSED",
        mapping_status="MAPPED",
        validation_status="PASSED",
    )
    return job_payload(job)


@router.get("/jobs/{job_id}")
async def get_processing_job(job_id: str, request: Request, session=Depends(require_session)):
    return job_payload(get_job_for_user(request, job_id, session))


@router.get("/jobs")
async def list_processing_jobs(request: Request, session=Depends(require_session)):
    repository = request.app.state.job_repository
    if hasattr(repository, "list_jobs"):
        jobs = repository.list_jobs(session.email)
    else:
        jobs = repository.list_for_user(session.email)
    return {"jobs": [job_payload(job) for job in jobs]}


@router.post("/jobs/{job_id}/corrections")
async def apply_corrections(
    job_id: str,
    data: HumanCorrectionRequest,
    request: Request,
    session=Depends(require_session),
):
    job = get_job_for_user(request, job_id, session)
    target = job.mapping_result.get("mapped_data") or job.extracted_data
    for field, new_value in data.corrections.items():
        old_value = target.get(field)
        target[field] = new_value
        if field == "rows" and isinstance(job.extracted_data, dict):
            job.extracted_data["rows"] = new_value
        job.human_corrections.append(
            HumanCorrection(
                field=field,
                old_value=old_value,
                new_value=new_value,
                user_email=session.email,
                timestamp=datetime.now().isoformat(),
            )
        )
        request.app.state.audit_log_service.activity(
            session.session_id, session.email, job.job_id, "HUMAN_EDIT", job.purpose, job.source_drive_file_id, "", "OK", field
        )
    job.mapping_result["mapped_data"] = target

    # Re-run reconciliation
    try:
        from app.accounting.reconciliation import AccountingReconciliationService
        target["reconciliation"] = AccountingReconciliationService().reconcile(target)
        if isinstance(job.extracted_data, dict):
            job.extracted_data["reconciliation"] = target["reconciliation"]
    except Exception as exc:
        pass

    validation = request.app.state.validation_service.validate(job.purpose, target)
    job.validation_result = validation.model_dump(mode="json")
    job.validation_status = validation.status
    request.app.state.lifecycle_service.update(job, validation_status=validation.status)
    return job_payload(job)


@router.post("/jobs/{job_id}/mapping")
async def resolve_mapping(
    job_id: str,
    data: MappingResolutionRequest,
    request: Request,
    session=Depends(require_session),
):
    job = get_job_for_user(request, job_id, session)
    mapped = job.mapping_result.get("mapped_data") or dict(job.extracted_data)
    missing = job.mapping_result.get("missing", [])

    for resolution in data.resolutions:
        kind = resolution.type.upper()
        if job.purpose == "MEMBER_RECEIPT":
            if kind == "BANK":
                mapped["bank_name_or_code"] = resolution.target_value
                for row in mapped.get("rows", []):
                    if isinstance(row, dict) and row.get("Society Bank Name/Bank code(Given to you by nobrokerhood)*") == resolution.source_value:
                        row["Society Bank Name/Bank code(Given to you by nobrokerhood)*"] = resolution.target_value
            if kind == "BILL_HEAD":
                mapped["bill_head"] = resolution.target_value
                for row in mapped.get("rows", []):
                    if isinstance(row, dict) and row.get("Bill Head*") == resolution.source_value:
                        row["Bill Head*"] = resolution.target_value
        if job.purpose == "VENDOR_INVOICE":
            if kind == "VENDOR":
                mapped["vendor_code"] = resolution.target_value
            if kind == "EXPENSE":
                for expense in mapped.get("expenses", []):
                    if expense.get("expense_description") == resolution.source_value:
                        expense["expense_code"] = resolution.target_value
        missing = [
            item for item in missing
            if not (item.get("type", "").upper() == kind and item.get("source_value") == resolution.source_value)
        ]

    job.mapping_result = {
        "status": "MAPPED" if not missing else "NEEDS_MAPPING",
        "mapped_data": mapped,
        "missing": missing,
    }
    job.mapping_status = job.mapping_result["status"]
    request.app.state.audit_log_service.activity(
        session.session_id, session.email, job.job_id, "MAPPING_CONFIRMED", job.purpose, job.source_drive_file_id, "", "OK", ""
    )
    validation = request.app.state.validation_service.validate(job.purpose, mapped)
    job.validation_result = validation.model_dump(mode="json")
    job.validation_status = validation.status
    request.app.state.lifecycle_service.update(job, mapping_status=job.mapping_status, validation_status=validation.status)
    return job_payload(job)


@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    try:
        request.app.state.accounting_workflow.approve_and_complete(job)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GoogleDriveError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return job_payload(job)


@router.post("/jobs/{job_id}/reject")
async def reject_job(
    job_id: str,
    request: Request,
    body: RejectJobRequest | None = None,
    session=Depends(require_session),
):
    job = get_job_for_user(request, job_id, session)
    if job.overall_status not in {"NEEDS_REVIEW", "FAILED"}:
        raise HTTPException(status_code=409, detail="JOB_NOT_AWAITING_REVIEW")
    job.human_status = "REJECTED"
    job.overall_status = "REJECTED"
    job.current_step = "REJECTED"
    reason_str = (body.reason if body else "") or "USER_REJECTED"
    comments_str = (body.comments if body else "") or ""
    job.last_error = f"REJECTED ({reason_str}): {comments_str}".strip(": ")
    request.app.state.audit_log_service.activity(
        session.session_id, session.email, job.job_id, "HUMAN_REJECTED", job.purpose, job.source_drive_file_id, "", "OK", reason_str
    )
    request.app.state.lifecycle_service.update(job, human_status="REJECTED", overall_status="REJECTED", current_step="REJECTED", last_error=job.last_error)
    return job_payload(job)


@router.get("/jobs/{job_id}/download")
async def download_output(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    if not job.output_bytes or job.overall_status != "COMPLETED":
        if not job.output_drive_file_id or job.overall_status != "COMPLETED":
            raise HTTPException(status_code=404, detail="OUTPUT_NOT_READY")
        try:
            job.output_bytes = request.app.state.drive_service.download_file(
                job.output_drive_file_id
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="OUTPUT_DOWNLOAD_FAILED") from exc
    request.app.state.audit_log_service.activity(
        session.session_id, session.email, job.job_id, "FILE_DOWNLOADED", job.purpose, job.source_drive_file_id, job.output_drive_file_id, "OK", job.output_filename
    )
    return StreamingResponse(
        iter([job.output_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={job.output_filename}"},
    )


@router.get("/jobs/{job_id}/source")
async def get_source_document(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    source_bytes = job.source_bytes
    if not source_bytes and job.source_drive_file_id:
        try:
            source_bytes = request.app.state.drive_service.download_file(job.source_drive_file_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="SOURCE_DOCUMENT_DOWNLOAD_FAILED") from exc
    if not source_bytes:
        raise HTTPException(status_code=404, detail="SOURCE_DOCUMENT_NOT_FOUND")

    mime = job.source_content_type or "application/pdf"
    return StreamingResponse(
        io.BytesIO(source_bytes),
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{job.source_filename}"'},
    )
