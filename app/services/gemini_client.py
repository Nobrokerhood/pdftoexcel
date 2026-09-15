import logging
import json
import time
from typing import Any, Callable, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import Settings
from app.core.errors import ExternalServiceUnavailableError, ServiceNotConfiguredError


logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRY_DELAY_SECONDS = 2

GEMINI_SPEND_CAP_MESSAGE = (
    "Gemini API is unavailable because the configured Google project has reached "
    "its monthly spending cap. Check Gemini AI Studio billing/spend settings or "
    "configure a different valid Gemini project/API key."
)

# HTTP codes worth retrying (same set the google-genai SDK treats as transient).
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# A 429 RESOURCE_EXHAUSTED is also returned for short-lived rate limits, which
# are retryable. Only these phrases mark the project-level spend cap, which
# stays exhausted until billing settings change.
SPEND_CAP_MARKERS = ("spending cap", "spend cap")

GEMINI_QUOTA_EXHAUSTED_MESSAGE = (
    "Gemini daily quota has been exhausted for the configured project. Please use an "
    "available/billed Gemini project or try again after the quota resets."
)
GEMINI_AUTH_MESSAGE = "Gemini authentication is not configured correctly."

# Quota ids in a 429 QuotaFailure that only reset daily (e.g.
# "GenerateRequestsPerDayPerProjectPerModel-FreeTier"); retrying them within a job is wasted.
DAILY_QUOTA_MARKER = "perday"
AUTH_FAILURE_REASONS = ("API_KEY_INVALID", "API_KEY_SERVICE_BLOCKED", "API_KEY_HTTP_REFERRER_BLOCKED")

PRIMARY_CAP_COOLDOWN_SECONDS = 15 * 60


class GeminiSpendCapError(ExternalServiceUnavailableError):
    """The Google project behind the Gemini API key has hit its spending cap."""

    def __init__(self, message: str = GEMINI_SPEND_CAP_MESSAGE):
        super().__init__(message)


class GeminiFallbackUnavailableError(GeminiSpendCapError):
    """The primary key is capped and the fallback key failed too."""

    def __init__(self, reason: str):
        super().__init__(
            f"{GEMINI_SPEND_CAP_MESSAGE} The fallback Gemini API key also failed ({reason})."
        )


class GeminiQuotaExhaustedError(ExternalServiceUnavailableError):
    """A daily request quota of the Google project behind the key is used up."""

    def __init__(self):
        super().__init__(GEMINI_QUOTA_EXHAUSTED_MESSAGE)


class GeminiAuthenticationError(ServiceNotConfiguredError):
    """The Gemini API key is invalid, blocked or lacks permission."""

    def __init__(self):
        super().__init__(GEMINI_AUTH_MESSAGE)


class GeminiTemporarilyUnavailableError(ExternalServiceUnavailableError):
    """Retryable Gemini errors (rate limit, overload, timeout) persisted through every retry."""

    def __init__(self, reason: str, attempts: int, rate_limited: bool = False):
        if rate_limited:
            message = (
                f"Gemini is temporarily rate limited ({reason}) after {attempts} attempt(s). "
                "Please try again shortly."
            )
        else:
            message = (
                f"Gemini is temporarily unavailable ({reason}) after {attempts} attempt(s). "
                "Please try again in a few minutes."
            )
        super().__init__(message)


def is_spend_cap_error(exc: BaseException) -> bool:
    if not isinstance(exc, genai_errors.APIError) or exc.code != 429:
        return False
    message = str(exc.message or "").lower()
    return any(marker in message for marker in SPEND_CAP_MARKERS)


def _error_details(exc: genai_errors.APIError) -> list[dict]:
    body = exc.details if isinstance(exc.details, dict) else {}
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    return [item for item in error.get("details") or [] if isinstance(item, dict)]


def is_daily_quota_error(exc: BaseException) -> bool:
    if not isinstance(exc, genai_errors.APIError) or exc.code != 429:
        return False
    for item in _error_details(exc):
        for violation in item.get("violations") or []:
            if DAILY_QUOTA_MARKER in str(violation.get("quotaId", "")).lower():
                return True
    return False


def is_authentication_error(exc: BaseException) -> bool:
    if not isinstance(exc, genai_errors.APIError):
        return False
    if exc.code in (401, 403):
        return True
    if exc.code == 400:
        reasons = {str(item.get("reason", "")) for item in _error_details(exc)}
        return bool(reasons.intersection(AUTH_FAILURE_REASONS)) or "api key not valid" in str(exc.message or "").lower()
    return False


def is_retryable_error(exc: BaseException) -> bool:
    if is_spend_cap_error(exc) or is_daily_quota_error(exc):
        return False
    if isinstance(exc, genai_errors.APIError):
        return exc.code in RETRYABLE_STATUS_CODES
    # Network failures and malformed model output keep the previous retry behaviour.
    return True


def describe_error(exc: BaseException) -> str:
    """Short, credential-free description for logs."""
    if isinstance(exc, genai_errors.APIError):
        return f"{type(exc).__name__} {exc.code} {exc.status}"
    if isinstance(exc.__cause__, genai_errors.APIError):
        return describe_error(exc.__cause__)
    return type(exc).__name__


class GeminiDocumentClient:
    """Gemini access with an optional fallback key.

    Policy per request:
    - The primary key is used first, with the normal retry rules.
    - Only a spending-cap error on the primary switches to the fallback key (when
      GEMINI_FALLBACK_ENABLED is true and GEMINI_API_KEY_FALLBACK is set). Rate
      limits, server errors and invalid-key/permission errors never switch keys.
    - The fallback gets one bounded run of the same retry rules. If it fails with
      an API error, GeminiFallbackUnavailableError is raised; keys never alternate.
    - After a primary spending-cap error, requests go straight to the fallback for
      PRIMARY_CAP_COOLDOWN_SECONDS, then the primary is tried again.
    """

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self._client = None
        self._fallback_client = None
        self._clock = clock
        self._primary_capped_until = 0.0

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")

        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        logger.info("Gemini client configured for model: %s", self.settings.gemini_model)
        return self._client

    def fallback_available(self) -> bool:
        fallback_key = (self.settings.gemini_api_key_fallback or "").strip()
        primary_key = (self.settings.gemini_api_key or "").strip()
        return bool(
            self.settings.gemini_fallback_enabled
            and fallback_key
            and fallback_key != primary_key
        )

    def _get_fallback_client(self):
        if self._fallback_client is None:
            self._fallback_client = genai.Client(
                api_key=self.settings.gemini_api_key_fallback.strip()
            )
            logger.info("Gemini fallback client configured.")
        return self._fallback_client

    def _request(self, call: Callable[[Any], T], retries: int) -> T:
        primary = self._get_client()
        use_fallback = self.fallback_available()

        if use_fallback and self._clock() < self._primary_capped_until:
            logger.info("Primary Gemini key is in spending-cap cooldown; using fallback key.")
            return self._fallback_request(call, retries)

        try:
            return self._with_retries(lambda: call(primary), retries)
        except GeminiSpendCapError:
            if not use_fallback:
                raise
            self._primary_capped_until = self._clock() + PRIMARY_CAP_COOLDOWN_SECONDS
            logger.warning("Primary Gemini key hit its spending cap; trying fallback key.")
            return self._fallback_request(call, retries)

    def _fallback_request(self, call: Callable[[Any], T], retries: int) -> T:
        fallback = self._get_fallback_client()
        try:
            result = self._with_retries(lambda: call(fallback), retries)
        except (ExternalServiceUnavailableError, genai_errors.APIError) as exc:
            reason = describe_error(exc)
            logger.error("Fallback Gemini key also failed (%s).", reason)
            raise GeminiFallbackUnavailableError(reason) from exc
        logger.info("Primary Gemini key hit spending cap; fallback key used.")
        return result

    def _with_retries(self, request: Callable[[], T], retries: int) -> T:
        for attempt in range(1, retries + 1):
            try:
                return request()
            except Exception as exc:
                if is_spend_cap_error(exc):
                    logger.error(
                        "Gemini request rejected: project monthly spending cap reached "
                        "(%s). Not retrying.",
                        describe_error(exc),
                    )
                    raise GeminiSpendCapError() from exc
                if is_daily_quota_error(exc):
                    logger.error("Gemini request rejected: daily quota exhausted (%s). Not retrying.", describe_error(exc))
                    raise GeminiQuotaExhaustedError() from exc
                if is_authentication_error(exc):
                    logger.error("Gemini request rejected: authentication failed (%s). Not retrying.", describe_error(exc))
                    raise GeminiAuthenticationError() from exc
                if not is_retryable_error(exc):
                    raise
                if attempt >= retries:
                    if isinstance(exc, genai_errors.APIError):
                        raise GeminiTemporarilyUnavailableError(
                            describe_error(exc), attempt, rate_limited=exc.code == 429
                        ) from exc
                    raise
                logger.warning(
                    "Gemini request failed (%s), attempt %s of %s; retrying.",
                    describe_error(exc),
                    attempt,
                    retries,
                )
                time.sleep(RETRY_DELAY_SECONDS)
        raise ValueError("retries must be at least 1.")

    def generate_content(self, prompt_parts: list[Any], retries: int = 2):
        return self._request(
            lambda client: client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            ),
            retries,
        )

    def generate_json(self, prompt_parts: list[Any], retries: int = 2):
        def request(client):
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = response.text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                cleaned = text.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)

        return self._request(request, retries)
