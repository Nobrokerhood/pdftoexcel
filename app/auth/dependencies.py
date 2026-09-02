from fastapi import Header, HTTPException, Request


def bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def require_session(
    request: Request,
    authorization: str | None = Header(default=None),
):
    token = bearer_token(authorization)
    try:
        return request.app.state.session_service.get_session(token)
    except Exception as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
