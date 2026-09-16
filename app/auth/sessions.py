import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets

from app.auth.user_master import AuthorizedUser
from app.core.config import Settings


class SessionError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


@dataclass
class SessionRecord:
    session_id: str
    token: str
    email: str
    name: str
    role: str
    login_at: datetime
    last_seen_at: datetime
    last_activity_at: datetime
    logout_at: datetime | None = None
    active_duration_seconds: int = 0
    status: str = "ACTIVE"

    def public_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "login_at": iso(self.login_at),
            "last_seen_at": iso(self.last_seen_at),
            "logout_at": iso(self.logout_at),
            "active_duration_seconds": self.active_duration_seconds,
            "status": self.status,
        }


class SessionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._sessions_by_token: dict[str, SessionRecord] = {}
        self._tokens_by_session_id: dict[str, str] = {}

    def _sign_token(self, session_id: str, user: AuthorizedUser, now: datetime) -> str:
        secret = (self.settings.session_secret or "").strip()
        if not secret:
            return secrets.token_urlsafe(32)
        payload = json.dumps({
            "sid": session_id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "login_at": iso(now),
            "ts": int(now.timestamp()),
        })
        b64_payload = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        sig = hmac.new(secret.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"nbh_{b64_payload}_{sig}"

    def create_session(self, user: AuthorizedUser) -> SessionRecord:
        now = utc_now()
        session_id = secrets.token_urlsafe(24)
        token = self._sign_token(session_id, user, now)
        session = SessionRecord(
            session_id=session_id,
            token=token,
            email=user.email,
            name=user.name,
            role=user.role,
            login_at=now,
            last_seen_at=now,
            last_activity_at=now,
        )
        self._sessions_by_token[session.token] = session
        self._tokens_by_session_id[session.session_id] = session.token
        return session

    def _recover_signed_session(self, token: str) -> SessionRecord | None:
        secret = (self.settings.session_secret or "").strip()
        if not secret or not token.startswith("nbh_"):
            return None
        parts = token.split("_")
        if len(parts) != 3:
            return None
        _, b64_payload, sig = parts
        expected_sig = hmac.new(secret.encode("utf-8"), b64_payload.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, sig):
            return None

        try:
            padding = (4 - len(b64_payload) % 4) % 4
            raw_json = base64.urlsafe_b64decode(b64_payload + "=" * padding)
            data = json.loads(raw_json)
            now = utc_now()
            ts = data.get("ts", 0)
            if (now.timestamp() - ts) > max(self.settings.session_inactivity_seconds * 3, 86400):
                return None

            login_at = datetime.fromisoformat(data["login_at"]) if data.get("login_at") else now
            session = SessionRecord(
                session_id=data["sid"],
                token=token,
                email=data["email"],
                name=data.get("name", ""),
                role=data.get("role", "USER"),
                login_at=login_at,
                last_seen_at=now,
                last_activity_at=now,
                status="ACTIVE",
            )
            self._sessions_by_token[token] = session
            self._tokens_by_session_id[session.session_id] = token
            return session
        except Exception:
            return None

    def get_session(self, token: str | None) -> SessionRecord:
        if not token:
            raise SessionError("Missing session token.")
        session = self._sessions_by_token.get(token)
        if not session:
            session = self._recover_signed_session(token)
        if not session:
            raise SessionError("Invalid session token.")
        if session.status != "ACTIVE":
            raise SessionError("Session is not active.")
        if (utc_now() - session.last_seen_at).total_seconds() > self.settings.session_inactivity_seconds:
            session.status = "EXPIRED"
            raise SessionError("Session expired.")
        return session

    def heartbeat(
        self,
        token: str,
        user_active: bool = True,
        page_visible: bool = True,
    ) -> SessionRecord:
        session = self.get_session(token)
        now = utc_now()
        delta = max(0, int((now - session.last_seen_at).total_seconds()))
        if user_active and page_visible:
            session.active_duration_seconds += min(
                delta, self.settings.session_heartbeat_grace_seconds
            )
            session.last_activity_at = now
        session.last_seen_at = now
        return session

    def logout(self, token: str | None) -> SessionRecord:
        session = self.get_session(token)
        session.logout_at = utc_now()
        session.status = "LOGGED_OUT"
        return session

