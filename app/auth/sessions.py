import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

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

    def create_session(self, user: AuthorizedUser) -> SessionRecord:
        now = utc_now()
        session = SessionRecord(
            session_id=secrets.token_urlsafe(24),
            token=secrets.token_urlsafe(32),
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

    def get_session(self, token: str | None) -> SessionRecord:
        if not token:
            raise SessionError("Missing session token.")
        session = self._sessions_by_token.get(token)
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
