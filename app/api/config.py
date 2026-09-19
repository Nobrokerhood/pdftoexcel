from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.accounting.folders import FolderConfigurationError
from app.accounting.purposes import PURPOSES
from app.accounting.templates import TemplateConfigurationError
from app.auth.dependencies import require_session
from app.documents.pdf_images import poppler_available
from app.google.sheets_service import GoogleSheetsNotConfiguredError


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/public")
async def public_config(request: Request):
    settings = request.app.state.settings
    if not settings.google_client_id:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth client is not configured on the server.",
        )
    return {
        "application_name": "Accounting AI",
        "google_client_id": settings.google_client_id,
        "allowed_email_domain": settings.allowed_email_domain,
        "allow_dev_login": bool(settings.allow_dev_login and settings.environment != "production"),
        "features": {
            "member_receipt": True,
            "vendor_invoice": True,
            "legacy_tools": True,
        },
    }


@router.get("/diagnostic")
async def config_diagnostic(request: Request):
    from app.core.config import get_config_diagnostic

    return get_config_diagnostic(request.app.state.settings)


def require_admin(session):
    if session.role != "ADMIN":
        raise HTTPException(status_code=403, detail="ADMIN_REQUIRED")


@router.get("/purposes")
async def purposes(session=Depends(require_session)):
    return {
        "purposes": [
            {"code": purpose.code, "label": purpose.label}
            for purpose in PURPOSES
            if purpose.enabled
        ],
        "user": {"email": session.email, "role": session.role},
    }


@router.get("/template/{purpose}")
async def active_template(purpose: str, request: Request, session=Depends(require_session)):
    try:
        template = request.app.state.template_registry_service.get_active_template(
            purpose
        )
    except TemplateConfigurationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    request.app.state.audit_log_service.activity(
        session.session_id,
        session.email,
        "",
        "PURPOSE_SELECTED",
        purpose.upper(),
        "",
        "",
        "OK",
        template.template_code,
    )
    return template.public_dict()


@router.get("/folder-route")
async def folder_route(
    request: Request,
    purpose: str = Query(...),
    status: str = Query("incoming"),
    session=Depends(require_session),
):
    try:
        folder_id = request.app.state.folder_router_service.route(purpose, status)
    except FolderConfigurationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "purpose": purpose.upper(),
        "status": status.lower(),
        "folder_id": folder_id,
    }


@router.get("/health")
async def config_health(request: Request, session=Depends(require_session)):
    require_admin(session)
    settings = request.app.state.settings
    sheets = request.app.state.sheets_service
    drive = request.app.state.drive_service

    statuses = {
        "google_auth": "READY" if settings.google_client_id else "MISSING",
        "service_account": (
            "READY"
            if settings.google_service_account_json or settings.google_service_account_file
            else "MISSING"
        ),
        "spreadsheet": (
            "READY"
            if settings.google_accounting_spreadsheet_id
            or settings.google_user_master_sheet_id
            else "MISSING"
        ),
        "user_master": "MISSING",
        "folder_config": "MISSING",
        "mapping_master": "MISSING",
        "drive_root": "MISSING",
        "gemini": "CONFIGURED" if settings.gemini_api_key else "MISSING",
        "pdf_processing": "READY" if poppler_available(settings.poppler_path) else "MISSING",
    }

    for table_key, status_key in [
        ("user_master", "user_master"),
        ("folder_config", "folder_config"),
        ("mapping_master", "mapping_master"),
    ]:
        try:
            sheets.read_table(table_key)
            statuses[status_key] = "READY"
        except GoogleSheetsNotConfiguredError:
            statuses[status_key] = "MISSING"
        except Exception:
            statuses[status_key] = "NO_ACCESS"

    if settings.google_drive_root_folder_id:
        try:
            drive.validate_folder_id(settings.google_drive_root_folder_id)
            statuses["drive_root"] = "READY"
        except Exception:
            statuses["drive_root"] = "NO_ACCESS"

    return {"status": statuses}


_CAPABILITY_CACHE: dict = {}
_CAPABILITY_TTL_SECONDS = 60


@router.get("/capabilities")
async def capability_health(request: Request):
    """Capability report (status only: never credentials, keys, ids or paths).

    Unauthenticated so deployment health checks can read it; cached for 60 s so
    health polling cannot turn into repeated OCR inference. It never spends a
    Gemini call: the live Gemini probe is /config/capabilities/live (admin only).
    """
    import time
    from starlette.concurrency import run_in_threadpool
    from app.documents.capabilities import collect_capabilities

    cached = _CAPABILITY_CACHE.get("report")
    if not cached or time.time() - cached[0] > _CAPABILITY_TTL_SECONDS:
        state = request.app.state
        report = await run_in_threadpool(collect_capabilities, state.settings, False, state.ocr_service,
                                         state.drive_service, state.sheets_service, state.gemini_client)
        _CAPABILITY_CACHE["report"] = (time.time(), report)
    report = _CAPABILITY_CACHE["report"][1]
    return JSONResponse(status_code=200 if report["overall"] in {"READY", "DEGRADED"} else 503, content=report)


@router.get("/capabilities/live")
async def capability_health_live(request: Request, session=Depends(require_session)):
    """Same report plus one real Gemini call. Admin only (it costs quota)."""
    from starlette.concurrency import run_in_threadpool
    from app.documents.capabilities import collect_capabilities

    require_admin(session)
    state = request.app.state
    report = await run_in_threadpool(collect_capabilities, state.settings, True, state.ocr_service,
                                     state.drive_service, state.sheets_service)
    return JSONResponse(status_code=200 if report["overall"] in {"READY", "DEGRADED"} else 503, content=report)


@router.get("/version")
async def version():
    from app.documents.capabilities import deployed_version
    return deployed_version()
