import logging
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import uuid

from app.accounting.folders import FolderConfigService, FolderRouterService
from app.accounting.mapping import MappingMasterService
from app.accounting.output import TemplateOutputGenerator
from app.accounting.templates import TemplateRegistryService
from app.accounting.validation import AccountingValidationService
from app.agents.extractor import ExtractionAgent, GeminiExtractionProvider
from app.agents.repair import GeminiRepairProvider, RepairAgent
from app.agents.verifier import GeminiVerificationProvider, VerificationAgent
from app.api.auth import router as auth_router
from app.api.audit import router as audit_router
from app.api.config import router as config_router
from app.api.legacy_tools import router as legacy_tools_router
from app.api.processing import router as processing_router
from app.audit.activity import AuditLogService, ProcessingLogService
from app.auth.google_auth import GoogleTokenVerifier
from app.auth.sessions import SessionService
from app.auth.user_master import UserMasterService
from app.core.config import get_settings, validate_production_config
from app.google.drive_service import GoogleDriveService
from app.google.sheets import GoogleSheetsAuditClient
from app.google.sheets_service import GoogleSheetsService
from app.processing.log_lifecycle import ProcessingLifecycleService
from app.processing.stores import (
    GoogleSheetsProcessingJobStore,
    InMemoryProcessingJobStore,
)
from app.services.gemini_client import GeminiDocumentClient
from app.workflows.accounting_graph import AccountingWorkflow


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app(
    settings=None,
    sheets_service=None,
    drive_service=None,
    google_token_verifier=None,
    session_service=None,
    extraction_provider=None,
    verification_provider=None,
    repair_provider=None,
    job_store=None,
) -> FastAPI:
    settings = settings or get_settings()
    prod_errors = validate_production_config(settings)
    if prod_errors:
        raise RuntimeError(f"Production configuration validation failed: {'; '.join(prod_errors)}")

    sheets_service = sheets_service or GoogleSheetsService(settings)
    docs_url = "/docs" if settings.enable_docs else None
    redoc_url = "/redoc" if settings.enable_docs else None
    openapi_url = "/openapi.json" if settings.enable_docs else None

    app = FastAPI(
        title="NoBrokerHood Accounting AI",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )
    app.state.settings = settings
    app.state.sheets_service = sheets_service
    app.state.drive_service = drive_service or GoogleDriveService(settings)
    app.state.google_token_verifier = google_token_verifier or GoogleTokenVerifier(
        settings
    )
    app.state.session_service = session_service or SessionService(settings)
    app.state.user_master_service = UserMasterService(settings, sheets_service)
    app.state.audit_log_service = AuditLogService(settings, sheets_service)
    app.state.processing_log_service = ProcessingLogService(settings, sheets_service)
    if job_store:
        app.state.job_store = job_store
    elif settings.google_accounting_spreadsheet_id:
        app.state.job_store = GoogleSheetsProcessingJobStore(sheets_service)
    else:
        app.state.job_store = InMemoryProcessingJobStore()
    app.state.lifecycle_service = ProcessingLifecycleService(
        app.state.processing_log_service, app.state.job_store
    )
    app.state.template_registry_service = TemplateRegistryService(
        settings, sheets_service
    )
    folder_config_service = FolderConfigService(settings, sheets_service)
    app.state.folder_config_service = folder_config_service
    app.state.folder_router_service = FolderRouterService(folder_config_service)
    app.state.audit_client = GoogleSheetsAuditClient(settings, sheets_service)
    app.state.gemini_client = GeminiDocumentClient(settings)
    app.state.mapping_service = MappingMasterService(settings, sheets_service)
    app.state.validation_service = AccountingValidationService()
    app.state.output_generator = TemplateOutputGenerator()
    app.state.job_repository = app.state.job_store
    from app.documents.ocr import DocumentOcrService, RapidOcrProvider
    app.state.ocr_service = DocumentOcrService(settings.poppler_path, RapidOcrProvider(), dpi=150)
    app.state.extraction_agent = ExtractionAgent(
        extraction_provider or GeminiExtractionProvider(app.state.gemini_client, ocr_service=app.state.ocr_service)
    )
    app.state.verification_agent = VerificationAgent(
        verification_provider or GeminiVerificationProvider(app.state.gemini_client, ocr_service=app.state.ocr_service)
    )
    app.state.repair_agent = RepairAgent(
        repair_provider or GeminiRepairProvider(app.state.gemini_client)
    )
    app.state.accounting_workflow = AccountingWorkflow(
        settings,
        app.state.extraction_agent,
        app.state.verification_agent,
        app.state.repair_agent,
        app.state.mapping_service,
        app.state.validation_service,
        app.state.output_generator,
        app.state.template_registry_service,
        app.state.folder_router_service,
        app.state.drive_service,
        app.state.lifecycle_service,
        app.state.audit_log_service,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_usage_logger(request: Request, call_next):
        if request.url.path == "/login-log":
            return await call_next(request)

        start_time = time.time()
        status = "OK"
        email = request.headers.get("X-User-Email", "anonymous")
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                status = "FAIL"
            return response
        except Exception:
            status = "FAIL"
            raise
        finally:
            process_time = round(time.time() - start_time, 3)
            ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
            app.state.audit_client.append_usage(
                email,
                request.method,
                request.url.path,
                status,
                process_time,
                ip,
                user_agent,
            )

    app.include_router(auth_router)
    app.include_router(config_router)
    app.include_router(processing_router)
    app.include_router(legacy_tools_router)
    app.include_router(audit_router)

    try:
        from kb.kb_service import router as kb_router

        app.include_router(kb_router)
        logger.info("Knowledge Bot router loaded successfully")
    except Exception as exc:
        logger.warning("Knowledge Bot not available: %s", exc)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        incident_id = str(uuid.uuid4())[:8]
        logger.error(
            "Unhandled exception [incident=%s] on %s %s: %s",
            incident_id,
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"An internal error occurred (Incident ID: {incident_id}). Please contact support."},
        )

    @app.get("/health", tags=["monitoring"])
    def health():
        return {"status": "healthy", "service": "nbh-accounting-ai"}

    @app.get("/readiness", tags=["monitoring"])
    def readiness():
        from app.documents.pdf_images import poppler_available

        has_creds = bool(settings.google_service_account_json or settings.google_service_account_file)
        has_gemini = bool(settings.gemini_api_key)
        has_poppler = poppler_available(settings.poppler_path)
        is_ready = has_gemini and has_creds and has_poppler
        return {
            "status": "ready" if is_ready else "not_ready",
            "checks": {
                "gemini_configured": has_gemini,
                "google_credentials_configured": has_creds,
                "poppler_available": has_poppler,
                "rapidocr_ready": True,
            },
        }

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def login_page():
        return FileResponse(PROJECT_ROOT / "index.html")

    @app.get("/{page_name}", include_in_schema=False)
    def frontend_page(page_name: str):
        if page_name not in {"accounting.html", "ocr.html", "voice.html", "session_timeout.js"}:
            raise HTTPException(status_code=404, detail="Not Found")
        return FileResponse(PROJECT_ROOT / page_name)

    from pathlib import Path
    from fastapi.responses import FileResponse

    root_dir = Path(__file__).resolve().parent.parent

    for static_file in [
        "index.html",
        "accounting.html",
        "ocr.html",
        "voice.html",
        "session_timeout.js",
        "knowledge-bot-v2.js",
        "test_launcher.html",
    ]:
        file_path = root_dir / static_file
        if file_path.exists():
            media_type = "text/html" if static_file.endswith(".html") else "application/javascript"

            def make_handler(p, mt):
                async def handler():
                    return FileResponse(p, media_type=mt)
                return handler

            app.add_api_route(f"/{static_file}", make_handler(file_path, media_type), methods=["GET"], include_in_schema=False)

    return app
