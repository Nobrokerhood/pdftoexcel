"""Processing API: upload, background processing, row-id-keyed review, approval."""

import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.accounting import review
from app.accounting.folders import FolderConfigurationError
from app.accounting.templates import TemplateConfigurationError
from app.auth.dependencies import require_session
from app.documents.pdf_images import InvalidPdfError, PdfDependencyMissingError, is_pdf, pdf_page_count
from app.google.drive_service import GoogleDriveError, GoogleDriveService
from app.processing.jobs import ProcessingJob
from app.workflows.accounting_graph import ApprovalBlockedError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/processing", tags=["processing"])

DRIVE_UPLOAD_FAILED_MESSAGE = "Unable to upload the document to Google Drive. Please contact the administrator."
SUPPORTED_SOURCE_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/jpg"}
EXTENSIONS = {".pdf": "PDF", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "20"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(60_000_000)))


class ValidateProcessingRequest(BaseModel):
    purpose: str
    destination_status: str = "incoming"


class HumanCorrectionRequest(BaseModel):
    corrections: dict[str, Any]
    reason: str | None = None


class RowEditRequest(BaseModel):
    values: dict[str, Any] = {}
    reason: str | None = None
    confirm: bool = True


class ReasonRequest(BaseModel):
    reason: str | None = None


class PromoteRequest(BaseModel):
    values: dict[str, Any] = {}
    reason: str | None = None


class RejectJobRequest(BaseModel):
    reason: str | None = None
    comments: str | None = None


class MappingResolution(BaseModel):
    type: str
    source_value: str
    target_value: str


class MappingResolutionRequest(BaseModel):
    resolutions: list[MappingResolution]


# ---------------------------------------------------------------------------
# payloads
# ---------------------------------------------------------------------------

STAGE_LABELS = {
    "UPLOAD": "File uploaded to Drive (Incoming)",
    "OCR_AND_EXTRACTION": "OCR + extraction",
    "VERIFICATION": "Verification",
    "REPAIR": "Repair",
    "MAPPING": "Mapping",
    "VALIDATION": "Validation",
    "HUMAN_REVIEW": "Human review",
    "EXPORT": "Excel generation",
    "DRIVE_UPLOAD": "Output to Drive",
}


def progress(job: ProcessingJob) -> list[dict[str, str]]:
    """Derived ONLY from the backend stage trace; a stage never shown as done unless it ran."""
    out = []
    for entry in job.public_trace():
        status = entry.get("status")
        ui = {"COMPLETED": "DONE", "RUNNING": "RUNNING", "FAILED": "FAILED", "NEEDS_REVIEW": "NEEDS_ATTENTION",
              "WAITING": "NEEDS_ATTENTION", "SKIPPED": "SKIPPED"}.get(status, status)
        out.append({"label": STAGE_LABELS.get(entry.get("stage"), entry.get("stage")), "status": ui,
                    "stage": entry.get("stage"), "message": entry.get("message", ""),
                    "duration_ms": entry.get("duration_ms")})
    return out


def job_payload(job: ProcessingJob) -> dict[str, Any]:
    data = job.summary()
    data["progress"] = progress(job)
    data["review_items"] = [i for i in (job.validation_result or {}).get("issues") or []]
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


def _require_reviewable(job: ProcessingJob):
    if job.overall_status != "NEEDS_REVIEW":
        raise HTTPException(status_code=409, detail="JOB_NOT_AWAITING_REVIEW")


# ---------------------------------------------------------------------------
# upload validation (before anything touches Drive or Sheets)
# ---------------------------------------------------------------------------

def validate_upload(filename: str, content_type: str | None, data: bytes, settings) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file extension. Only .pdf, .jpg, .jpeg and .png are supported.")
    if (content_type or "") not in SUPPORTED_SOURCE_TYPES:
        raise HTTPException(status_code=400, detail="Only PDF, JPG, JPEG and PNG are supported.")
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(data) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds size limit.")
    kind = EXTENSIONS[ext]
    # The declared type must match the actual bytes (extension spoofing).
    if kind == "PDF":
        if not is_pdf(data):
            raise HTTPException(status_code=400, detail="Invalid PDF file signature.")
        try:
            pages = pdf_page_count(data, settings.poppler_path)
        except PdfDependencyMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except InvalidPdfError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if pages > MAX_PDF_PAGES:
            raise HTTPException(status_code=400, detail=f"The PDF has {pages} pages; at most {MAX_PDF_PAGES} are supported.")
        return "application/pdf"
    signature_ok = data.startswith(b"\xff\xd8") if kind == "JPEG" else data.startswith(b"\x89PNG\r\n\x1a\n")
    if not signature_ok:
        raise HTTPException(status_code=400, detail=f"Invalid {kind} file signature.")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=400, detail="Image dimensions are too large to process safely.")
            img.verify()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc
    return "image/jpeg" if kind == "JPEG" else "image/png"


def format_safe_drive_error(exc: Exception) -> str:
    """Status and Google reason code only (e.g. 403 storageQuotaExceeded); never
    folder ids, service-account identity or raw exception text."""
    try:
        import json
        from googleapiclient.errors import HttpError
        if isinstance(exc, HttpError):
            status = getattr(getattr(exc, "resp", None), "status", "error")
            reason = ""
            try:
                content = exc.content.decode("utf-8") if isinstance(exc.content, bytes) else exc.content
                errors = (json.loads(content).get("error") or {}).get("errors") or []
                reason = str((errors[0] or {}).get("reason", "")) if errors else ""
            except Exception:
                reason = ""
            return f"Google Drive upload failed ({status}{' ' + reason if reason else ''})."
    except ImportError:
        pass
    import re
    match = re.search(r"\b([45]\d\d)\s+([A-Za-z]{4,40})\b", str(exc))
    if match:
        return f"Google Drive upload failed ({match.group(1)} {match.group(2)})."
    return DRIVE_UPLOAD_FAILED_MESSAGE


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@router.post("/validate-config")
async def validate_config(data: ValidateProcessingRequest, request: Request, session=Depends(require_session)):
    try:
        template = request.app.state.template_registry_service.get_active_template(data.purpose)
        folder_id = request.app.state.folder_router_service.route(data.purpose, data.destination_status)
    except (TemplateConfigurationError, FolderConfigurationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    request.app.state.audit_log_service.activity(session.session_id, session.email, "", "PURPOSE_SELECTED",
                                                 data.purpose.upper(), "", "", "OK", template.template_code)
    return {"purpose": data.purpose.upper(), "template": template.public_dict(),
            "destination": {"status": data.destination_status.lower(), "folder_id": folder_id}}


@router.post("/jobs")
async def create_processing_job(request: Request, purpose: str = Form(...), file: UploadFile = File(...),
                                session=Depends(require_session)):
    state = request.app.state
    from app.core.resources import ocr_profile
    profile = ocr_profile()
    if profile["below_minimum"]:
        # Refuse up front rather than be OOM-killed mid-job (measured: OCR needs > 512 MB).
        raise HTTPException(status_code=503, detail=(
            f"This server instance has {profile['memory_limit_mb']} MB of memory; document OCR needs at least "
            f"1024 MB. Processing is disabled until the instance is upgraded."))
    raw_filename = file.filename or "upload.pdf"
    base_name = os.path.basename(raw_filename.replace("\\", "/"))
    safe_filename = GoogleDriveService.safe_filename(base_name)
    source_bytes = await file.read()
    content_type = validate_upload(base_name, file.content_type, source_bytes, state.settings)

    purpose = purpose.strip().upper()
    try:
        template = state.template_registry_service.get_active_template(purpose)
        folders = {k: state.folder_router_service.route(purpose, k) for k in ("incoming", "review", "completed", "output")}
    except (TemplateConfigurationError, FolderConfigurationError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Configuration unavailable for upload: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Configuration (Google Sheets) is temporarily unavailable; "
                                                    "please retry in a minute.") from exc

    job = state.job_repository.create(session.session_id, session.email, purpose, template.template_code,
                                      safe_filename, content_type, source_bytes)
    job.file_sha256 = hashlib.sha256(source_bytes).hexdigest()
    job.source_folder_id, job.review_folder_id = folders["incoming"], folders["review"]
    job.completed_folder_id, job.output_folder_id = folders["completed"], folders["output"]
    state.lifecycle_service.create_row(job)
    state.audit_log_service.activity(session.session_id, session.email, job.job_id, "PURPOSE_SELECTED", purpose,
                                     "", "", "OK", template.template_code)

    upload = job.stage_start("UPLOAD", "Uploading the source document to Drive (Incoming)")
    try:
        state.lifecycle_service.update(job, overall_status="UPLOADING", current_step="UPLOADING")
        job.source_drive_file_id = state.drive_service.upload_file(job.source_filename, source_bytes,
                                                                   folders["incoming"], content_type)
        job.stage_end(upload, "COMPLETED", f"{len(source_bytes)} bytes stored in Incoming")
        state.lifecycle_service.update(job, source_drive_file_id=job.source_drive_file_id, overall_status="PROCESSING",
                                       current_step="QUEUED")
        state.audit_log_service.activity(session.session_id, session.email, job.job_id, "FILE_UPLOAD", purpose,
                                         job.source_drive_file_id, "", "OK", job.source_filename)
    except Exception as exc:
        logger.warning("Drive upload failed for job %s: %s", job.job_id, exc)
        job.stage_end(upload, "FAILED", "Drive upload failed", errors=[format_safe_drive_error(exc)])
        job.last_error = f"DRIVE_UPLOAD_FAILED: {format_safe_drive_error(exc)}"
        state.lifecycle_service.update(job, overall_status="FAILED", last_error=job.last_error)
        state.audit_log_service.activity(session.session_id, session.email, job.job_id, "FILE_UPLOAD", purpose, "", "",
                                         "FAIL", job.last_error)
        raise HTTPException(status_code=502, detail=format_safe_drive_error(exc)) from exc

    workflow = state.accounting_workflow

    def run():
        try:
            workflow.run_until_review(job)
        except Exception as exc:
            logger.exception("Workflow crashed for job %s", job.job_id)
            job.last_error = f"PROCESSING_FAILED: {type(exc).__name__}"
            state.lifecycle_service.update(job, overall_status="FAILED", last_error=job.last_error)

    state.job_runner.submit(job.job_id, run)
    return job_payload(job)


@router.get("/jobs/{job_id}")
async def get_processing_job(job_id: str, request: Request, session=Depends(require_session)):
    return job_payload(get_job_for_user(request, job_id, session))


@router.get("/jobs")
async def list_processing_jobs(request: Request, session=Depends(require_session)):
    jobs = request.app.state.job_repository.list_jobs(session.email)
    return {"jobs": [{k: v for k, v in job_payload(job).items()
                      if k not in ("extracted_data", "verification_result", "validation_result", "mapping_result")}
                     for job in jobs]}


def _after_review_action(request: Request, job: ProcessingJob, entries, session, action: str):
    job.human_corrections.extend(entries)
    request.app.state.accounting_workflow.revalidate(job)
    for entry in entries:
        request.app.state.audit_log_service.activity(session.session_id, session.email, job.job_id,
                                                     f"HUMAN_{entry.action}", job.purpose, job.source_drive_file_id,
                                                     "", "OK", f"{entry.row_id or ''} {entry.field}")
    # Reviewer actions change the document itself: always persist them.
    request.app.state.lifecycle_service.update(job, force=True, validation_status=job.validation_status)
    return job_payload(job)


def _review_call(fn):
    try:
        return fn()
    except review.ReviewError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/rows/{row_id}")
async def edit_row(job_id: str, row_id: str, body: RowEditRequest, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.edit_row(job.extracted_data, row_id, body.values, session.email,
                                                   body.reason, confirm=body.confirm))
    return _after_review_action(request, job, entries, session, "EDIT")


@router.post("/jobs/{job_id}/rows/{row_id}/confirm")
async def confirm_row(job_id: str, row_id: str, body: ReasonRequest, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.confirm_row(job.extracted_data, row_id, session.email, body.reason))
    return _after_review_action(request, job, entries, session, "CONFIRM")


@router.post("/jobs/{job_id}/rows")
async def add_row(job_id: str, body: PromoteRequest, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.add_row(job.extracted_data, body.values, session.email, body.reason))
    return _after_review_action(request, job, entries, session, "ADD_ROW")


@router.post("/jobs/{job_id}/rows/{row_id}/delete")
async def delete_row(job_id: str, row_id: str, body: ReasonRequest, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.delete_row(job.extracted_data, row_id, session.email, body.reason))
    return _after_review_action(request, job, entries, session, "DELETE_ROW")


@router.post("/jobs/{job_id}/candidates/{candidate_id}/dismiss")
async def dismiss_candidate(job_id: str, candidate_id: str, body: ReasonRequest, request: Request,
                            session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.dismiss_candidate(job.extracted_data, candidate_id, session.email, body.reason))
    return _after_review_action(request, job, entries, session, "DISMISS")


@router.post("/jobs/{job_id}/candidates/{candidate_id}/promote")
async def promote_candidate(job_id: str, candidate_id: str, body: PromoteRequest, request: Request,
                            session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    entries = _review_call(lambda: review.promote_candidate(job.extracted_data, candidate_id, body.values,
                                                            session.email, body.reason))
    return _after_review_action(request, job, entries, session, "PROMOTE")


@router.post("/jobs/{job_id}/corrections")
async def apply_corrections(job_id: str, data: HumanCorrectionRequest, request: Request, session=Depends(require_session)):
    """Review-grid save: rows are matched by _row_id (never by position)."""
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    rows = data.corrections.get("rows")
    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="corrections.rows must be a list of rows with _row_id")
    entries = _review_call(lambda: review.apply_row_list(job.extracted_data, rows, session.email, data.reason))
    return _after_review_action(request, job, entries, session, "EDIT")


@router.post("/jobs/{job_id}/mapping")
async def resolve_mapping(job_id: str, data: MappingResolutionRequest, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    _require_reviewable(job)
    column_for = {"BANK": "Society Bank Name/Bank code(Given to you by nobrokerhood)*", "BILL_HEAD": "Bill Head*"}
    entries = []
    for resolution in data.resolutions:
        kind = resolution.type.upper()
        column = column_for.get(kind)
        if column:
            for row in job.extracted_data.get("rows") or []:
                if column in (row.get("_edited_fields") or []):
                    continue  # reviewer's own value is authoritative
                if row.get(column) == resolution.source_value:
                    before = row[column]
                    row[column] = resolution.target_value
                    entries.append(review._audit(session.email, "MAPPING", row.get("_row_id"), column, before,
                                                 resolution.target_value, "mapping resolved by reviewer"))
        elif kind == "VENDOR":
            detail = job.extracted_data.setdefault("vendor_detail", {})
            entries.append(review._audit(session.email, "MAPPING", None, "vendor_code", detail.get("vendor_code"),
                                         resolution.target_value, "vendor mapping resolved by reviewer"))
            detail["vendor_code"] = resolution.target_value
        job.extracted_data["mapping_missing"] = [
            m for m in job.extracted_data.get("mapping_missing") or []
            if not (str(m.get("type", "")).upper() == kind and m.get("source_value") == resolution.source_value)]
    job.mapping_status = "MAPPED" if not job.extracted_data.get("mapping_missing") else "NEEDS_MAPPING"
    job.mapping_result = {"status": job.mapping_status, "missing": job.extracted_data.get("mapping_missing") or []}
    request.app.state.lifecycle_service.update(job, mapping_status=job.mapping_status)
    return _after_review_action(request, job, entries, session, "MAPPING")


@router.post("/jobs/{job_id}/approve")
async def approve_job(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    try:
        request.app.state.accounting_workflow.approve_and_complete(job)
    except ApprovalBlockedError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "blocking": exc.details[:50]}) from exc
    except GoogleDriveError as exc:
        raise HTTPException(status_code=502, detail=format_safe_drive_error(exc)) from exc
    return job_payload(job)


@router.post("/jobs/{job_id}/reject")
async def reject_job(job_id: str, request: Request, body: RejectJobRequest | None = None, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    if job.overall_status not in {"NEEDS_REVIEW", "FAILED"}:
        raise HTTPException(status_code=409, detail="JOB_NOT_AWAITING_REVIEW")
    reason = (body.reason if body else "") or "USER_REJECTED"
    comments = (body.comments if body else "") or ""
    job.human_status = job.overall_status = job.current_step = "REJECTED"
    job.last_error = f"REJECTED ({reason}): {comments}".strip(": ")
    request.app.state.audit_log_service.activity(session.session_id, session.email, job.job_id, "HUMAN_REJECTED",
                                                 job.purpose, job.source_drive_file_id, "", "OK", reason)
    request.app.state.lifecycle_service.update(job, human_status="REJECTED", overall_status="REJECTED",
                                               current_step="REJECTED", last_error=job.last_error)
    return job_payload(job)


def _attachment(filename: str, inline: bool = False) -> str:
    safe = "".join(ch for ch in filename if ch.isalnum() or ch in "._- ") or "download"
    return f"{'inline' if inline else 'attachment'}; filename=\"{safe}\"; filename*=UTF-8''{quote(filename)}"


@router.get("/jobs/{job_id}/download")
async def download_output(job_id: str, request: Request, session=Depends(require_session)):
    job = get_job_for_user(request, job_id, session)
    if job.overall_status != "COMPLETED" and not job.output_bytes:
        raise HTTPException(status_code=404, detail="OUTPUT_NOT_READY")
    if not job.output_bytes:
        if not job.output_drive_file_id:
            raise HTTPException(status_code=404, detail="OUTPUT_NOT_READY")
        try:
            job.output_bytes = request.app.state.drive_service.download_file(job.output_drive_file_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="OUTPUT_DOWNLOAD_FAILED") from exc
    request.app.state.audit_log_service.activity(session.session_id, session.email, job.job_id, "FILE_DOWNLOADED",
                                                 job.purpose, job.source_drive_file_id, job.output_drive_file_id, "OK",
                                                 job.output_filename)
    return StreamingResponse(iter([job.output_bytes]),
                             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": _attachment(job.output_filename)})


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
    return StreamingResponse(io.BytesIO(source_bytes), media_type=job.source_content_type or "application/pdf",
                             headers={"Content-Disposition": _attachment(job.source_filename, inline=True)})


@router.get("/jobs/{job_id}/rows/{row_id}/crop")
async def row_crop(job_id: str, row_id: str, request: Request, session=Depends(require_session)):
    """The source region a row was read from (canonical page pixels), as PNG."""
    job = get_job_for_user(request, job_id, session)
    evidence = (job.extracted_data.get("row_evidence") or {}).get(row_id)
    if not evidence or not evidence.get("bbox") or not evidence.get("page"):
        raise HTTPException(status_code=404, detail="ROW_HAS_NO_SOURCE_REGION")
    rep = request.app.state.ocr_service.cached(job.file_sha256) if job.file_sha256 else None
    image = rep.page_image(int(evidence["page"])) if rep else None
    if image is None:
        raise HTTPException(status_code=404, detail="CROP_UNAVAILABLE_OPEN_SOURCE_DOCUMENT")
    x0, y0, x1, y1 = evidence["bbox"]
    pad = max(10, int((y1 - y0) * 0.3))
    crop = image.crop((0, max(0, y0 - pad), image.width, min(image.height, y1 + pad)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png",
                             headers={"Cache-Control": "private, max-age=300"})
