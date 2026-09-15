from fastapi import HTTPException


class ExternalServiceUnavailableError(RuntimeError):
    """Raised when an external service cannot serve requests and retrying will not help."""


class ServiceNotConfiguredError(ExternalServiceUnavailableError):
    """Raised when an optional external service is required but not configured."""


def service_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)
