"""OCR orchestration: the only layer that selects OCR providers.

For each page it:
1. runs the primary engine on the page's canonical image;
2. assesses the result and decides, with recorded reasons, whether the page is
   difficult (handwriting, low confidence, noise);
3. on a difficult page, runs the second engine and a pixel-filter preprocessing
   variant, all on the SAME canonical image, so every result shares one
   coordinate space;
4. keeps every result as evidence, and scores each one with field-aware metrics
   to choose which one supplies the primary line set. The losing results are
   kept, not discarded.

Neither engine is assumed to be better: RapidOCR outperformed PaddleOCR on the
handwritten benchmark page, so selection is decided per page by measured
evidence.
"""

import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from app.documents.ocr_contract import OcrProvider, OcrResult
from app.documents.ocr_quality import OcrQualityAssessor

logger = logging.getLogger(__name__)

ENSEMBLE_AUTO = "auto"      # second engine only on difficult pages
ENSEMBLE_ALWAYS = "always"  # second engine on every scanned page
ENSEMBLE_OFF = "off"        # primary engine only (reported as DEGRADED)

# Printed text reads at ~0.97+. Below these the page is treated as difficult.
DIFFICULT_MEAN_CONFIDENCE = 0.90
DIFFICULT_LOW_LINE_RATIO = 0.15
LOW_LINE_CONFIDENCE = 0.80

_AMOUNT_TOKEN = re.compile(r"\d[\d,]*(?:\.\d{1,2})?\s*(?:/-|/=|1-)?")
_DATE_TOKEN = re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b")


def _clahe(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(enhanced).convert("RGB")


def _adaptive_threshold(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    return Image.fromarray(thresh).convert("RGB")


def _denoise(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(denoised)
    return Image.fromarray(enhanced).convert("RGB")


# Pixel filters only: none changes image size or geometry, so a line box from a
# variant is directly comparable with a box from the canonical image.
VARIANT_FILTERS = {
    "contrast_clahe": _clahe,
    "adaptive_threshold": _adaptive_threshold,
    "denoise_contrast": _denoise,
}


@dataclass
class ResultScore:
    key: str
    lines: int
    mean_confidence: float
    amount_tokens: int
    date_tokens: int
    noise_lines: int
    score: float


@dataclass
class OcrRoutingDecision:
    page: int
    difficulty: str
    reasons: list[str] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    selected: str = ""
    selection_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def score_result(result: OcrResult) -> ResultScore:
    """Field-aware quality, not line_count x mean_confidence.

    Each line contributes its confidence; lines carrying accounting evidence
    (amount-like or date-like tokens) count double, and noise lines (one or two
    characters, or mostly symbols) subtract. This rewards reading the fields
    that matter and penalises the fragment spam that inflates line counts.
    """
    amount_tokens = date_tokens = noise = 0
    total = 0.0
    for line in result.lines:
        text = line.text.strip()
        alnum = sum(ch.isalnum() for ch in text)
        is_noise = len(text) <= 2 or alnum < max(1, len(text) // 2)
        if is_noise:
            noise += 1
            total -= 0.5
            continue
        weight = 1.0
        if _DATE_TOKEN.search(text):
            date_tokens += 1
            weight += 1.0
        elif _AMOUNT_TOKEN.search(text) and sum(ch.isdigit() for ch in text) >= 2:
            amount_tokens += 1
            weight += 1.0
        total += line.confidence * weight
    return ResultScore(
        key=result.key,
        lines=len(result.lines),
        mean_confidence=result.mean_confidence,
        amount_tokens=amount_tokens,
        date_tokens=date_tokens,
        noise_lines=noise,
        score=round(total, 3),
    )


@dataclass
class PageOcrEvidence:
    page: int
    width: int
    height: int
    results: list[OcrResult]
    primary: OcrResult
    decision: OcrRoutingDecision


class OcrOrchestrator:
    def __init__(
        self,
        providers: list[OcrProvider],
        ensemble: str = ENSEMBLE_AUTO,
        max_variants: int = 1,
        page_time_budget_s: float = 120.0,
    ):
        if not providers:
            raise ValueError("at least one OCR provider is required")
        self.providers = list(providers)
        self.ensemble = ensemble if ensemble in {ENSEMBLE_AUTO, ENSEMBLE_ALWAYS, ENSEMBLE_OFF} else ENSEMBLE_AUTO
        self.max_variants = max(0, int(max_variants))
        self.page_time_budget_s = page_time_budget_s
        self.assessor = OcrQualityAssessor()

    # -- capability ---------------------------------------------------------
    def available_providers(self) -> list[OcrProvider]:
        return [p for p in self.providers if p.available()]

    def status(self) -> dict:
        return {
            "ensemble": self.ensemble,
            "engines": [
                {"engine": p.name, "available": p.available(), "reason": getattr(p, "unavailable_reason", "")}
                for p in self.providers
            ],
        }

    # -- reading ------------------------------------------------------------
    def _difficulty(self, result: OcrResult) -> tuple[bool, list[str]]:
        reasons = []
        if not result.lines:
            return True, ["primary engine returned no text"]
        low = sum(1 for line in result.lines if line.confidence < LOW_LINE_CONFIDENCE) / len(result.lines)
        if result.mean_confidence < DIFFICULT_MEAN_CONFIDENCE:
            reasons.append(f"mean confidence {result.mean_confidence:.3f} < {DIFFICULT_MEAN_CONFIDENCE}")
        if low > DIFFICULT_LOW_LINE_RATIO:
            reasons.append(f"{low:.0%} of lines below {LOW_LINE_CONFIDENCE} confidence")
        quality = self.assessor.evaluate([(l.text, l.confidence, l.bbox) for l in result.lines])
        if quality.needs_adaptive_fallback:
            reasons.extend(quality.review_reasons)
        return bool(reasons), reasons

    def read_page(self, page: int, canonical: Image.Image) -> PageOcrEvidence:
        started = time.monotonic()
        available = self.available_providers()
        if not available:
            from app.documents.ocr_engines import OcrUnavailableError
            reasons = "; ".join(f"{p.name}: {getattr(p, 'unavailable_reason', '')}" for p in self.providers)
            raise OcrUnavailableError(reasons)

        primary_engine = available[0]
        results: list[OcrResult] = []
        decision = OcrRoutingDecision(page=page, difficulty="UNKNOWN")
        for p in self.providers:
            if p not in available:
                decision.skipped.append(f"{p.name}: unavailable ({getattr(p, 'unavailable_reason', '')})")

        first = primary_engine.read(canonical, page_idx=page, variant="canonical")
        results.append(first)
        decision.runs.append({"engine": first.engine, "variant": first.variant, "lines": len(first.lines),
                              "duration_ms": first.duration_ms, "trigger": "primary"})

        difficult, reasons = self._difficulty(first)
        decision.difficulty = "DIFFICULT" if difficult else "CLEAN"
        decision.reasons = reasons or ["primary result is high confidence"]

        secondary = [p for p in available if p is not primary_engine]
        run_second = bool(secondary) and (
            self.ensemble == ENSEMBLE_ALWAYS or (self.ensemble == ENSEMBLE_AUTO and difficult)
        )
        if secondary and not run_second:
            decision.skipped.append(
                f"{secondary[0].name}: not run (ensemble={self.ensemble}, page {'difficult' if difficult else 'clean'})"
            )
        if run_second:
            for engine in secondary:
                if time.monotonic() - started > self.page_time_budget_s:
                    decision.skipped.append(f"{engine.name}: page time budget exhausted")
                    break
                try:
                    res = engine.read(canonical, page_idx=page, variant="canonical")
                    results.append(res)
                    decision.runs.append({"engine": res.engine, "variant": res.variant, "lines": len(res.lines),
                                          "duration_ms": res.duration_ms, "trigger": "difficult page ensemble"})
                except Exception as exc:  # one engine failing must not lose the page
                    decision.skipped.append(f"{engine.name}: failed during read ({type(exc).__name__})")
                    logger.warning("OCR engine %s failed on page %s: %s", engine.name, page, exc)

        if difficult and self.max_variants:
            quality = self.assessor.evaluate([(l.text, l.confidence, l.bbox) for l in first.lines])
            preferred = "adaptive_threshold" if quality.garbage_ratio > 0.08 else "contrast_clahe"
            for name in [preferred, *[n for n in VARIANT_FILTERS if n != preferred]][: self.max_variants]:
                if time.monotonic() - started > self.page_time_budget_s:
                    decision.skipped.append(f"variant {name}: page time budget exhausted")
                    break
                try:
                    filtered = VARIANT_FILTERS[name](canonical)
                    res = primary_engine.read(filtered, page_idx=page, variant=name)
                    results.append(res)
                    decision.runs.append({"engine": res.engine, "variant": res.variant, "lines": len(res.lines),
                                          "duration_ms": res.duration_ms, "trigger": "difficult page variant"})
                except Exception as exc:
                    decision.skipped.append(f"variant {name}: failed ({type(exc).__name__})")

        scores = [score_result(r) for r in results]
        decision.scores = [asdict(s) for s in scores]
        best_index = max(range(len(results)), key=lambda i: scores[i].score)
        primary = results[best_index]
        decision.selected = primary.key
        decision.selection_reason = (
            f"highest field-aware score {scores[best_index].score} "
            f"({scores[best_index].amount_tokens} amount-like, {scores[best_index].date_tokens} date-like, "
            f"{scores[best_index].noise_lines} noise lines)"
        )
        return PageOcrEvidence(
            page=page, width=canonical.width, height=canonical.height,
            results=results, primary=primary, decision=decision,
        )


def build_default_orchestrator(ensemble: str | None = None) -> OcrOrchestrator:
    """The production engine set. The only place engine classes are instantiated."""
    from app.documents.ocr_engines import PaddleOcrProvider, RapidOcrProvider

    from app.core.resources import ocr_profile
    mode = (ensemble or ocr_profile()["ensemble"]).strip().lower()
    providers: list[OcrProvider] = [RapidOcrProvider()]
    if mode != ENSEMBLE_OFF:
        providers.append(PaddleOcrProvider())
    return OcrOrchestrator(providers, ensemble=mode)
