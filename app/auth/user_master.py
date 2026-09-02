from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.core.config import Settings
from app.google.sheets_service import GoogleSheetsNotConfiguredError, GoogleSheetsService


class AuthorizationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthorizedUser:
    email: str
    name: str
    role: str
    active: bool = True


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "active"}


class UserMasterService:
    VALID_ROLES = {"USER", "REVIEWER", "ADMIN"}

    def __init__(self, settings: Settings, sheets_service: GoogleSheetsService):
        self.settings = settings
        self.sheets_service = sheets_service

    def lookup_user(self, email: str) -> AuthorizedUser | None:
        try:
            record = self.sheets_service.lookup_table_row_by_key(
                "user_master",
                "Email",
                email.lower(),
            )
        except GoogleSheetsNotConfiguredError:
            record = None
        if not record:
            return None

        role = str(record.get("Role", "USER")).strip().upper() or "USER"
        if role not in self.VALID_ROLES:
            role = "USER"

        return AuthorizedUser(
            email=str(record.get("Email", email)).strip().lower(),
            name=str(record.get("Name", "")).strip(),
            role=role,
            active=_truthy(record.get("Active", False)),
        )

    def authorize(self, email: str, fallback_name: str = "") -> AuthorizedUser:
        email = email.strip().lower()
        user = self.lookup_user(email)
        if user:
            if not user.active:
                raise AuthorizationError("User is inactive.")
            return user

        domain = self.settings.allowed_email_domain
        if (
            self.settings.allow_domain_wide_access
            and domain
            and email.endswith(f"@{domain.lower()}")
        ):
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return AuthorizedUser(
                email=email,
                name=fallback_name,
                role="USER",
                active=True,
            )

        raise AuthorizationError("User is not authorized.")
