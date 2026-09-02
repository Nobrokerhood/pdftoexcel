from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.auth.dependencies import bearer_token, require_session
from app.auth.google_auth import AuthError
from app.auth.sessions import SessionError
from app.auth.user_master import AuthorizationError


router = APIRouter(prefix="/auth", tags=["auth"])


class GoogleLoginRequest(BaseModel):
    credential: str


class HeartbeatRequest(BaseModel):
    user_active: bool = True
    page_visible: bool = True


@router.post("/google-login")
async def google_login(data: GoogleLoginRequest, request: Request):
    try:
        verified = request.app.state.google_token_verifier.verify(data.credential)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        user = request.app.state.user_master_service.authorize(
            verified.email, verified.name
        )
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    session = request.app.state.session_service.create_session(user)
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    request.app.state.audit_log_service.login(
        session.session_id, user.email, user.name, ip, user_agent
    )
    request.app.state.audit_log_service.activity(
        session.session_id, user.email, "", "LOGIN", "auth", "", "", "OK", ""
    )
    request.app.state.audit_log_service.session_snapshot(session)

    response = session.public_dict()
    response["session_token"] = session.token
    return response


@router.get("/me")
async def me(session=Depends(require_session)):
    return session.public_dict()


@router.post("/heartbeat")
async def heartbeat(
    data: HeartbeatRequest,
    request: Request,
    session=Depends(require_session),
):
    updated = request.app.state.session_service.heartbeat(
        session.token,
        user_active=data.user_active,
        page_visible=data.page_visible,
    )
    request.app.state.audit_log_service.session_snapshot(updated)
    return updated.public_dict()


@router.post("/logout")
async def logout(request: Request):
    token = bearer_token(request.headers.get("authorization"))
    try:
        session = request.app.state.session_service.logout(token)
    except SessionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    request.app.state.audit_log_service.logout(session.session_id, session.email)
    request.app.state.audit_log_service.session_snapshot(session)
    return session.public_dict()
