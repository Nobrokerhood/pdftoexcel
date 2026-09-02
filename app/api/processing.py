from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.accounting.folders import FolderConfigurationError
from app.accounting.schemas import HumanCorrection
from app.accounting.templates import TemplateConfigurationError
from app.auth.dependencies import require_session
from app.google.drive_service import GoogleDriveError
from app.processing.jobs import ProcessingJob


router = APIRouter(prefix="/processing", tags=["processing"])

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


class MappingResolution(BaseModel):
    type: str
    source_value: str
    target_value: str


class MappingResolutionRequest(BaseModel):
    resolutions: list[MappingResolution]


def progress(job: ProcessingJob) -> list[dict[str, str]]:
    steps = [
        ("File uploaded", bool(job.source_drive_file_id)),
        ("Stored in Incoming", bool(job.source_folder_id)),
        ("Extraction", job.extraction_status == "COMPLETED"),
        ("AI Verification", job.verification_status == "PASSED"),
        ("Mapping", job.mapping_status == "MAPPED"),
        ("Validation", job.validation_status == "PASSED"),
        ("Human review", job.human_status in {"NEEDS_REVIEW", "APPROVED", "REJECTED"}),
        ("Excel generation", bool(job.output_filename)),
    ]
    return [
        {"label": label, "status": "DONE" if done else "PENDING"}
        for label, done in steps
    ]


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
    if file.content_type not in SUPPORTED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG, and PNG are supported.")

    source_bytes = await file.read()
    if len(source_bytes) > request.app.state.settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds size limit.")

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
        file.filename or "upload",
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
        raise HTTPException(status_code=502, detail="DRIVE_UPLOAD_FAILED") from exc

    try:
        request.app.state.accounting_workflow.run_until_review(job)
    except Exception as exc:
        job.overall_status = "FAILED"
        job.last_error = str(exc)
        request.app.state.lifecycle_service.update(job, overall_status="FAILED", last_error=job.last_error)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
            if kind == "BILL_HEAD":
                mapped["bill_head"] = resolution.target_value
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
async def reject_job(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    job.human_status = "REJECTED"
    job.overall_status = "REJECTED"
    job.current_step = "REJECTED"
    request.app.state.audit_log_service.activity(
        session.session_id, session.email, job.job_id, "HUMAN_REJECTED", job.purpose, job.source_drive_file_id, "", "OK", ""
    )
    request.app.state.lifecycle_service.update(job, human_status="REJECTED", overall_status="REJECTED", current_step="REJECTED")
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
