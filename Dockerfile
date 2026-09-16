# ==============================================================================
# Production Dockerfile for NoBrokerHood (NBH) Accounting AI
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8000

# Install system dependencies:
# - poppler-utils: Required for PDF to image rasterization (pdf2image)
# - libgl1 & libglib2.0-0: Required by OpenCV / ONNX runtime for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root application user
RUN useradd -m -u 1001 appuser

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code (secrets excluded via .dockerignore)
COPY . /app

# Ensure proper permissions for non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose production port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI server startup command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--no-access-log"]
