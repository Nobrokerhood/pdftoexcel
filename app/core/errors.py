from fastapi import HTTPException


class ServiceNotConfiguredError(RuntimeError):
    """Raised when an optional external service is required but not configured."""


def service_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=503, detail=detail)
