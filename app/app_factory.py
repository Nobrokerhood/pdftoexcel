import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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
from app.core.config import get_settings
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
    sheets_service = sheets_service or GoogleSheetsService(settings)
    app = FastAPI(title="NoBrokerHood PDF to Excel & Split Tool")
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
    app.state.extraction_agent = ExtractionAgent(
        extraction_provider or GeminiExtractionProvider(app.state.gemini_client)
    )
    app.state.verification_agent = VerificationAgent(
        verification_provider or GeminiVerificationProvider(app.state.gemini_client)
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

    @app.get("/")
    def root():
        return {"message": "NoBrokerHood PDF to Excel & Split API running."}

    return app
