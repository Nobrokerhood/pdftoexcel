from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.accounting.folders import FolderConfigurationError
from app.accounting.purposes import PURPOSES
from app.accounting.templates import TemplateConfigurationError
from app.auth.dependencies import require_session
from app.google.sheets_service import GoogleSheetsNotConfiguredError


router = APIRouter(prefix="/config", tags=["config"])


@router.get("/public")
async def public_config(request: Request):
    settings = request.app.state.settings
    return {
        "application_name": "Accounting AI",
        "google_client_id": settings.frontend_google_client_id,
        "allowed_email_domain": settings.allowed_email_domain,
        "features": {
            "member_receipt": True,
            "vendor_invoice": True,
            "legacy_tools": True,
        },
    }


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
