"""Runtime resource profile: size the OCR stack to the memory actually available.

Measured in the production image (python:3.13-slim, peak cgroup memory):
* Python stack + RapidOCR loaded ........................ ~516 MB
* + PaddleOCR loaded ..................................... ~830 MB
* handwritten page, RapidOCR + PaddleOCR + variant, 200 dpi ~1872 MB
* same page, RapidOCR only, 200 dpi ...................... ~837 MB
* same page, RapidOCR only, 150 dpi ...................... ~657-699 MB

Loading PaddleOCR on a smaller instance crashed the production container
(HTTP 503 then restart). So the ensemble is enabled only when the container
limit covers the measured peak with headroom, and the canonical DPI drops on
sub-gigabyte instances. Explicit OCR_ENSEMBLE / OCR_CANONICAL_DPI settings win.
"""

import os
from functools import lru_cache

ENSEMBLE_MIN_MB = 2560        # 1872 MB measured peak + headroom
FULL_DPI_MIN_MB = 1024        # RapidOCR-only 200 dpi peaked at 837 MB
MINIMUM_MB = 768              # below this even 150 dpi RapidOCR (~700 MB) is unsafe


@lru_cache
def container_memory_limit_mb() -> int | None:
    """cgroup v2 / v1 memory limit of this container, or None when unlimited/unknown."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        if value >= 1 << 60:  # cgroup v1 "unlimited"
            return None
        return value // (1024 * 1024)
    return None


def ocr_profile() -> dict:
    limit = container_memory_limit_mb()
    explicit_ensemble = os.getenv("OCR_ENSEMBLE")
    explicit_dpi = os.getenv("OCR_CANONICAL_DPI")
    reasons = []
    if explicit_ensemble:
        ensemble = explicit_ensemble.strip().lower()
        reasons.append(f"OCR_ENSEMBLE={ensemble} set explicitly")
    elif limit is not None and limit < ENSEMBLE_MIN_MB:
        ensemble = "off"
        reasons.append(f"PaddleOCR disabled: container memory {limit} MB < {ENSEMBLE_MIN_MB} MB "
                       f"(measured ensemble peak 1872 MB)")
    elif limit is None and os.getenv("ENVIRONMENT", "").strip().lower() == "production":
        # Fail safe: on a runtime that does not expose its memory limit (Render's
        # native runtime), loading PaddleOCR crashed the instance twice. Enable the
        # ensemble explicitly (OCR_ENSEMBLE=auto) only on an instance >= 2560 MB.
        ensemble = "off"
        reasons.append("PaddleOCR disabled: production memory limit not detectable; set OCR_ENSEMBLE=auto "
                       f"only on an instance with >= {ENSEMBLE_MIN_MB} MB")
    else:
        ensemble = "auto"
    if explicit_dpi:
        dpi = int(explicit_dpi)
    elif limit is not None and limit < FULL_DPI_MIN_MB:
        dpi = 150
        reasons.append(f"canonical DPI lowered to 150: container memory {limit} MB < {FULL_DPI_MIN_MB} MB")
    else:
        dpi = 200
    return {"memory_limit_mb": limit, "ensemble": ensemble, "dpi": dpi, "reasons": reasons,
            "below_minimum": limit is not None and limit < MINIMUM_MB}


def process_memory_mb() -> dict:
    """Current and peak resident memory of this process (Linux /proc), in MB."""
    out = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                out[key] = int(value.strip().split()[0]) // 1024
    except OSError:
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                out[key] = int(value.strip().split()[0]) // 1024
    except OSError:
        pass
    return out
