"""Runtime capability probing for the document-processing stack.

Every capability is proven by exercising it, never by an import check (the
paddlepaddle 3.3.1 runtime imports cleanly and then fails inside its C++
executor). OCR engines are probed on the SAME instances the pipeline uses, so
the report cannot diverge from what processing actually has. Reports status
only: never credentials, keys, ids or paths.
"""

import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

READY = "READY"
UNAVAILABLE = "UNAVAILABLE"
NOT_CONFIGURED = "NOT_CONFIGURED"

# Missing any of these makes the service NOT_READY instead of quietly degrading.
REQUIRED_IN_PRODUCTION = ("rapidocr", "pdf_renderer", "image_processing", "gemini")
# The ensemble's second opinion. Without it OCR disagreement cannot be detected
# (more rows go to human review) but the pipeline stays safe: DEGRADED.
RECOMMENDED_IN_PRODUCTION = ("paddleocr",)


@dataclass
class CapabilityStatus:
    name: str
    status: str
    detail: str = ""
    version: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timed(fn) -> tuple[Any, int]:
    start = time.monotonic()
    result = fn()
    return result, int((time.monotonic() - start) * 1000)


def _probe_engine(provider, name: str) -> CapabilityStatus:
    try:
        ok, ms = _timed(provider.available)
        if not ok:
            return CapabilityStatus(name, UNAVAILABLE,
                                    getattr(provider, "unavailable_reason", "") or "inference probe failed",
                                    duration_ms=ms)
        version = "installed"
        try:
            if name == "paddleocr":
                import paddle
                import paddleocr
                version = f"{getattr(paddleocr, '__version__', 'installed')} (paddlepaddle {paddle.__version__}, cpu)"
            elif name == "rapidocr":
                import rapidocr
                version = getattr(rapidocr, "__version__", "") or "installed"
        except Exception:
            pass
        return CapabilityStatus(name, READY, "inference probe passed", version, ms)
    except Exception as exc:
        return CapabilityStatus(name, UNAVAILABLE, f"{type(exc).__name__}: {str(exc)[:160]}")


def probe_rapidocr(provider=None) -> CapabilityStatus:
    from app.documents.ocr_engines import RapidOcrProvider
    return _probe_engine(provider or RapidOcrProvider(), "rapidocr")


def probe_paddleocr(provider=None) -> CapabilityStatus:
    from app.documents.ocr_engines import PaddleOcrProvider
    return _probe_engine(provider or PaddleOcrProvider(), "paddleocr")


def probe_pdf_renderer(poppler_path: str | None) -> CapabilityStatus:
    try:
        from pdf2image import pdfinfo_from_bytes
        minimal = (b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                   b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                   b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 99 99]>>endobj\ntrailer<</Root 1 0 R>>")
        info, ms = _timed(lambda: pdfinfo_from_bytes(minimal, poppler_path=poppler_path))
        return CapabilityStatus("pdf_renderer", READY, f"poppler responded ({info.get('Pages', '?')} page)", duration_ms=ms)
    except Exception as exc:
        return CapabilityStatus("pdf_renderer", UNAVAILABLE, f"{type(exc).__name__}: {str(exc)[:160]}")


def probe_image_processing() -> CapabilityStatus:
    try:
        import cv2
        import numpy as np

        def _run():
            gray = cv2.cvtColor(np.zeros((16, 16, 3), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            return True

        _, ms = _timed(_run)
        return CapabilityStatus("image_processing", READY, "cvtColor + CLAHE probe passed", cv2.__version__, ms)
    except Exception as exc:
        return CapabilityStatus("image_processing", UNAVAILABLE, f"{type(exc).__name__}: {str(exc)[:160]}")


BLOCKING_GEMINI_ERRORS = {"GeminiSpendCapError", "GeminiQuotaExhaustedError", "GeminiAuthenticationError",
                          "GeminiFallbackUnavailableError"}


def probe_gemini(settings, live: bool = False, client=None) -> CapabilityStatus:
    """Configuration check by default; `live=True` spends one real API call.

    Without a live call, the most recent REAL call of the running service is
    used: a key that exists but is spend-capped is not reported READY.
    """
    if not getattr(settings, "gemini_api_key", None):
        return CapabilityStatus("gemini", NOT_CONFIGURED, "GEMINI_API_KEY is not set")
    model = getattr(settings, "gemini_model", "")
    history = list(getattr(client, "call_history", []) or []) if client is not None else []
    last = history[-1] if history else None
    if not live and last and last.get("status") == "FAILED" and last.get("error") in BLOCKING_GEMINI_ERRORS             and time.time() - float(last.get("at") or 0) < 900:
        return CapabilityStatus("gemini", UNAVAILABLE, f"last real call failed: {last['error']}", model)
    if not live and last and last.get("status") == "SUCCESS":
        return CapabilityStatus("gemini", READY, "last real call succeeded", model)
    if not live:
        return CapabilityStatus("gemini", READY, "api key configured (no live call made)", model)
    try:
        from app.services.gemini_client import GeminiDocumentClient
        _, ms = _timed(lambda: GeminiDocumentClient(settings).generate_json(
            ['Return exactly this JSON and nothing else: {"ping":"ok"}'], retries=1))
        return CapabilityStatus("gemini", READY, "live call succeeded", model, ms)
    except Exception as exc:
        return CapabilityStatus("gemini", UNAVAILABLE, f"live call failed: {type(exc).__name__}", model)


def probe_drive(settings, drive_service) -> CapabilityStatus:
    folder = getattr(settings, "google_shared_drive_id", None) or getattr(settings, "google_drive_root_folder_id", None)
    if drive_service is None or not (getattr(settings, "google_service_account_json", None)
                                     or getattr(settings, "google_service_account_file", None)):
        return CapabilityStatus("drive", NOT_CONFIGURED, "service account not configured")
    if not folder:
        return CapabilityStatus("drive", NOT_CONFIGURED, "no shared drive / root folder configured")
    try:
        _, ms = _timed(lambda: drive_service.get_file_metadata(folder))
        return CapabilityStatus("drive", READY, "metadata read of the configured root succeeded", duration_ms=ms)
    except Exception as exc:
        return CapabilityStatus("drive", UNAVAILABLE, f"metadata read failed: {type(exc).__name__}")


def probe_sheets(settings, sheets_service) -> CapabilityStatus:
    if sheets_service is None or not getattr(settings, "google_accounting_spreadsheet_id", None):
        return CapabilityStatus("sheets", NOT_CONFIGURED, "accounting spreadsheet not configured")
    try:
        _, ms = _timed(lambda: sheets_service.spreadsheet().title)
        return CapabilityStatus("sheets", READY, "spreadsheet opened", duration_ms=ms)
    except Exception as exc:
        return CapabilityStatus("sheets", UNAVAILABLE, f"open failed: {type(exc).__name__}")


def deployed_version() -> dict:
    """Commit the running code was built from (never secret)."""
    commit = (os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "").strip()
    if not commit:
        try:
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3,
                                    cwd=root).stdout.strip()
        except Exception:
            commit = ""
    return {"commit": commit or "unknown", "branch": os.getenv("RENDER_GIT_BRANCH", "") or "unknown",
            "service": os.getenv("RENDER_SERVICE_NAME", "") or "local"}


def collect_capabilities(settings, live_gemini: bool = False, ocr_service=None, drive_service=None,
                         sheets_service=None, gemini_client=None) -> dict[str, Any]:
    engines = {}
    if ocr_service is not None:
        for provider in ocr_service.orchestrator.providers:
            engines[provider.name] = provider
    from app.core.resources import ocr_profile
    profile = ocr_profile()
    if ocr_service is not None and "paddleocr" not in engines:
        paddle = CapabilityStatus("paddleocr", UNAVAILABLE,
                                  "; ".join(profile["reasons"]) or "disabled by OCR_ENSEMBLE=off")
    else:
        paddle = _probe_engine(engines["paddleocr"], "paddleocr") if "paddleocr" in engines else probe_paddleocr()
    if profile["below_minimum"]:
        # Loading an OCR model here would OOM-kill this instance (measured in production).
        note = (f"not probed: container memory {profile['memory_limit_mb']} MB is below the "
                f"measured OCR minimum (RapidOCR-only needs >= 1024 MB)")
        rapid = CapabilityStatus("rapidocr", UNAVAILABLE, note)
        paddle = CapabilityStatus("paddleocr", UNAVAILABLE, note)
    else:
        rapid = _probe_engine(engines["rapidocr"], "rapidocr") if "rapidocr" in engines else probe_rapidocr()
    probes = [
        rapid,
        paddle,
        probe_pdf_renderer(getattr(settings, "poppler_path", None)),
        probe_image_processing(),
        probe_gemini(settings, live=live_gemini, client=gemini_client),
    ]
    if drive_service is not None or sheets_service is not None:
        probes += [probe_drive(settings, drive_service), probe_sheets(settings, sheets_service)]
    limit = profile["memory_limit_mb"]
    probes.append(CapabilityStatus(
        "memory", UNAVAILABLE if profile["below_minimum"] else READY,
        (f"container limit {limit} MB" if limit else "no container memory limit detected")
        + f"; OCR ensemble={profile['ensemble']}, dpi={profile['dpi']}"
        + (f" ({'; '.join(profile['reasons'])})" if profile["reasons"] else "")
        + ("; full OCR ensemble needs >= 2560 MB" if limit and limit < 2560 else ""),
        str(limit or ""), 0))
    by_name = {p.name: p for p in probes}
    required = list(REQUIRED_IN_PRODUCTION) + [n for n in ("drive", "sheets") if n in by_name] + ["memory"]
    missing_required = [n for n in required if by_name[n].status != READY]
    missing_recommended = [n for n in RECOMMENDED_IN_PRODUCTION if by_name[n].status != READY]
    overall = "NOT_READY" if missing_required else ("DEGRADED" if missing_recommended else READY)
    ocr_engines = [n for n in ("rapidocr", "paddleocr") if by_name[n].status == READY]
    return {
        "schema_version": 2,
        "environment": str(getattr(settings, "environment", "development")).lower(),
        "overall": overall,
        "ocr_engines_ready": ocr_engines,
        "ocr_engine_count": len(ocr_engines),
        "ensemble_possible": len(ocr_engines) >= 2,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "capabilities": {p.name: p.to_dict() for p in probes},
        "version": deployed_version(),
    }
