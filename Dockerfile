# ==============================================================================
# NoBrokerHood Accounting AI: production image
# Same Python minor version and pinned dependencies as the verified test runs.
# ==============================================================================
FROM python:3.13-slim

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    JOB_EXECUTION=background \
    JOB_CONCURRENCY=2 \
    GIT_COMMIT=${GIT_COMMIT} \
    PORT=8000

# poppler-utils: PDF rendering; libgl1/libglib2.0-0: OpenCV; libgomp1: PaddlePaddle CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils libgl1 libglib2.0-0 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 1001 appuser

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chown -R appuser:appuser /app
USER appuser

# Download the PaddleOCR mobile models at build time (not on the first request),
# and prove both engines can run inference inside the image.
RUN python -c "from app.documents.ocr_engines import RapidOcrProvider, PaddleOcrProvider; \
r, p = RapidOcrProvider(), PaddleOcrProvider(); \
assert r.available(), r.unavailable_reason; assert p.available(), p.unavailable_reason; print('ocr engines ready')"

EXPOSE 8000

# /config/capabilities returns 503 when a required capability is missing, so a
# degraded container fails its probe instead of serving traffic.
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/config/capabilities || exit 1

# ONE worker: job state, sessions and OCR representations live in-process;
# concurrency comes from the bounded background job pool (JOB_CONCURRENCY).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
