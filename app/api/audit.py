from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter()


class LoginLog(BaseModel):
    email: str
    name: str | None = ""
    login_time: str


@router.post("/login-log")
async def log_login(data: LoginLog, request: Request):
    ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logged = request.app.state.audit_client.append_login(
        data.email, data.name, data.login_time, ip, user_agent
    )
    if not logged:
        raise HTTPException(status_code=503, detail="Login audit is not configured.")

    return {"status": "logged"}
