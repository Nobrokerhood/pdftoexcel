from dataclasses import dataclass

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token

from app.core.config import Settings


class AuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedGoogleUser:
    email: str
    name: str


class GoogleTokenVerifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    def verify(self, credential: str) -> VerifiedGoogleUser:
        if not credential:
            raise AuthError("Missing Google ID token.")
        if not self.settings.google_client_id:
            raise AuthError("GOOGLE_CLIENT_ID is not configured.")

        try:
            payload = id_token.verify_oauth2_token(
                credential, GoogleRequest(), self.settings.google_client_id
            )
        except Exception as exc:
            raise AuthError("Invalid Google ID token.") from exc

        issuer = payload.get("iss")
        if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
            raise AuthError("Invalid Google token issuer.")

        audience = payload.get("aud")
        if audience != self.settings.google_client_id:
            raise AuthError("Invalid Google token audience.")

        if not payload.get("email_verified"):
            raise AuthError("Google email is not verified.")

        email = str(payload.get("email", "")).strip().lower()
        if not email:
            raise AuthError("Google token did not include an email.")

        return VerifiedGoogleUser(email=email, name=str(payload.get("name", "")))
