import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.accounting.folders import FolderConfigService, FolderRouterService
from app.accounting.mapping import MappingMasterService
from app.accounting.output import TemplateOutputGenerator
from app.accounting.templates import TemplateRegistryService
from app.accounting.validation import AccountingValidationService
from app.agents.extractor import ExtractionAgent, GeminiExtractionProvider
from app.agents.repair import GeminiRepairProvider, RepairAgent
from app.agents.verifier import GeminiVerificationProvider, VerificationAgent
from app.api.audit import router as audit_router
from app.api.auth import router as auth_router
from app.api.config import router as config_router
from app.api.legacy_tools import router as legacy_tools_router
from app.api.processing import router as processing_router
from app.audit.activity import AuditLogService, ProcessingLogService
from app.auth.dependencies import require_session
from app.auth.google_auth import GoogleTokenVerifier
from app.auth.sessions import SessionService
from app.auth.user_master import UserMasterService
from app.core.config import get_settings, validate_production_config
from app.documents.ocr import DocumentOcrService
from app.documents.ocr_orchestrator import build_default_orchestrator
from app.google.drive_service import GoogleDriveService
from app.google.sheets import GoogleSheetsAuditClient
from app.google.sheets_service import GoogleSheetsService
from app.processing.jobs import JobRunner
from app.processing.log_lifecycle import ProcessingLifecycleService
from app.processing.stores import GoogleSheetsProcessingJobStore, InMemoryProcessingJobStore
from app.services.gemini_client import GeminiDocumentClient
from app.workflows.accounting_graph import AccountingWorkflow

logger = logging.getLogger(__name__)

STATIC_FILES = ("index.html", "accounting.html", "ocr.html", "voice.html", "session_timeout.js",
                "knowledge-bot-v2.js", "test_launcher.html")


def create_app(settings=None, sheets_service=None, drive_service=None, google_token_verifier=None,
               session_service=None, extraction_provider=None, verification_provider=None, repair_provider=None,
               job_store=None, ocr_service=None, job_runner=None) -> FastAPI:
    settings = settings or get_settings()
    prod_errors = validate_production_config(settings)
    if prod_errors:
        raise RuntimeError(f"Production configuration validation failed: {'; '.join(prod_errors)}")

    sheets_service = sheets_service or GoogleSheetsService(settings)
    if isinstance(sheets_service, GoogleSheetsService) and sheets_service.write_queue is None:
        from app.google.sheets_service import SheetsWriteQueue
        sheets_service.write_queue = SheetsWriteQueue(synchronous=settings.job_execution == "inline")
    app = FastAPI(title="NoBrokerHood Accounting AI",
                  docs_url="/docs" if settings.enable_docs else None,
                  redoc_url="/redoc" if settings.enable_docs else None,
                  openapi_url="/openapi.json" if settings.enable_docs else None)
    state = app.state
    state.settings = settings
    state.sheets_service = sheets_service
    state.drive_service = drive_service or GoogleDriveService(settings)
    state.google_token_verifier = google_token_verifier or GoogleTokenVerifier(settings)
    state.session_service = session_service or SessionService(settings)
    state.user_master_service = UserMasterService(settings, sheets_service)
    state.audit_log_service = AuditLogService(settings, sheets_service)
    state.processing_log_service = ProcessingLogService(settings, sheets_service)
    if job_store:
        state.job_store = job_store
    elif settings.google_accounting_spreadsheet_id:
        state.job_store = GoogleSheetsProcessingJobStore(sheets_service, state.drive_service)
    else:
        state.job_store = InMemoryProcessingJobStore()
    state.job_repository = state.job_store
    state.lifecycle_service = ProcessingLifecycleService(state.processing_log_service, state.job_store)
    state.template_registry_service = TemplateRegistryService(settings, sheets_service)
    state.folder_config_service = FolderConfigService(settings, sheets_service)
    state.folder_router_service = FolderRouterService(state.folder_config_service)
    state.audit_client = GoogleSheetsAuditClient(settings, sheets_service)
    state.gemini_client = GeminiDocumentClient(settings)
    state.mapping_service = MappingMasterService(settings, sheets_service)
    state.validation_service = AccountingValidationService()
    state.output_generator = TemplateOutputGenerator()
    # The only OCR orchestration instance: every consumer shares it.
    state.ocr_service = ocr_service or DocumentOcrService(settings.poppler_path, build_default_orchestrator(),
                                                          cache_size=8)
    state.extraction_agent = ExtractionAgent(
        extraction_provider or GeminiExtractionProvider(state.gemini_client, ocr_service=state.ocr_service))
    state.verification_agent = VerificationAgent(
        verification_provider or GeminiVerificationProvider(state.gemini_client, ocr_service=state.ocr_service))
    state.repair_agent = RepairAgent(repair_provider or GeminiRepairProvider(state.gemini_client))
    state.accounting_workflow = AccountingWorkflow(
        settings, state.extraction_agent, state.verification_agent, state.repair_agent, state.mapping_service,
        state.validation_service, state.output_generator, state.template_registry_service,
        state.folder_router_service, state.drive_service, state.lifecycle_service, state.audit_log_service)
    state.job_runner = job_runner or JobRunner(settings.job_execution, settings.job_concurrency)

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_allowed_origins), allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    # Usage logging is fire-and-forget and only for authenticated API calls, so an
    # anonymous flood cannot exhaust the Sheets quota that job state depends on.
    usage_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="usage-log")

    @app.middleware("http")
    async def api_usage_logger(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        if request.headers.get("authorization") and request.url.path.startswith(("/processing", "/auth", "/config")):
            args = (request.headers.get("X-User-Email", "authenticated"), request.method, request.url.path,
                    "FAIL" if response.status_code >= 400 else "OK", round(time.time() - start, 3),
                    request.client.host if request.client else "unknown", request.headers.get("user-agent", "unknown"))
            usage_pool.submit(lambda: state.audit_client.append_usage(*args))
        return response

    app.include_router(auth_router)
    app.include_router(config_router)
    app.include_router(processing_router)
    app.include_router(legacy_tools_router, dependencies=[Depends(require_session)])
    app.include_router(audit_router, dependencies=[Depends(require_session)])
    try:
        from kb.kb_service import router as kb_router
        app.include_router(kb_router, dependencies=[Depends(require_session)])
    except Exception as exc:
        logger.warning("Knowledge Bot not available: %s", type(exc).__name__)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        incident_id = str(uuid.uuid4())[:8]
        logger.error("Unhandled exception [incident=%s] on %s %s: %s", incident_id, request.method,
                     request.url.path, type(exc).__name__, exc_info=True)
        return JSONResponse(status_code=500, content={
            "detail": f"An internal error occurred (Incident ID: {incident_id}). Please contact support."})

    @app.get("/health", tags=["monitoring"])
    def health():
        # Liveness only. Readiness (engines, Poppler, Gemini, Drive, Sheets) is /config/capabilities.
        return {"status": "healthy", "service": "nbh-accounting-ai"}

    _readiness_cache: dict = {}
    _readiness_lock = threading.Lock()

    @app.get("/readiness", tags=["monitoring"])
    def readiness():
        from app.documents.capabilities import collect_capabilities
        with _readiness_lock:
            cached = _readiness_cache.get("value")
            if not cached or time.time() - cached[0] > 300:
                report = collect_capabilities(settings, ocr_service=state.ocr_service)
                _readiness_cache["value"] = (time.time(), report)
            report = _readiness_cache["value"][1]
        body = {"status": "ready" if report["overall"] != "NOT_READY" else "not_ready", "overall": report["overall"],
                "missing_required": report["missing_required"]}
        return JSONResponse(status_code=200 if report["overall"] != "NOT_READY" else 503, content=body)

    root_dir = Path(__file__).resolve().parent.parent

    @app.get("/")
    def root():
        return {"message": "NoBrokerHood PDF to Excel & Split API running."}

    for static_file in STATIC_FILES:
        file_path = root_dir / static_file
        if file_path.exists():
            media_type = "text/html" if static_file.endswith(".html") else "application/javascript"

            def make_handler(p, mt):
                async def handler():
                    return FileResponse(p, media_type=mt)
                return handler

            app.add_api_route(f"/{static_file}", make_handler(file_path, media_type), methods=["GET"],
                              include_in_schema=False)
    return app
