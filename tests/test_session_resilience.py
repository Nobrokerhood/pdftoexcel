from dataclasses import replace
import time
import pytest
from app.auth.sessions import SessionService, utc_now
from app.auth.user_master import AuthorizedUser
from tests.test_google_foundation import settings as base_settings


def _test_settings(session_secret="prod-resilience-secret-123", inactivity=1200):
    return replace(
        base_settings(),
        session_secret=session_secret,
        session_inactivity_seconds=inactivity,
    )


def test_signed_token_creation_and_recovery_across_server_restarts():
    settings = _test_settings()
    svc1 = SessionService(settings)
    user = AuthorizedUser(email="ops@nobroker.in", name="Ops Engineer", role="ADMIN")
    session = svc1.create_session(user)

    assert session.token.startswith("nbh_")
    assert session.email == "ops@nobroker.in"

    # Simulate new server instance or separate worker process (empty in-memory cache)
    svc2 = SessionService(settings)
    recovered = svc2.get_session(session.token)

    assert recovered.email == "ops@nobroker.in"
    assert recovered.role == "ADMIN"
    assert recovered.session_id == session.session_id
    assert recovered.status == "ACTIVE"


def test_tampered_token_is_rejected():
    settings = _test_settings()
    svc1 = SessionService(settings)
    user = AuthorizedUser(email="test@nobroker.in", name="Test", role="USER")
    session = svc1.create_session(user)

    # Tamper with token signature
    parts = session.token.split("_")
    tampered_token = f"{parts[0]}_{parts[1]}_badsignature00000"

    svc2 = SessionService(settings)
    with pytest.raises(Exception) as exc:
        svc2.get_session(tampered_token)
    assert "Invalid session token" in str(exc.value)


def test_expired_inactivity_rejects_session():
    settings = _test_settings(inactivity=1)
    svc = SessionService(settings)
    user = AuthorizedUser(email="test@nobroker.in", name="Test", role="USER")
    session = svc.create_session(user)

    time.sleep(1.2)
    with pytest.raises(Exception) as exc:
        svc.get_session(session.token)
    assert "Session expired" in str(exc.value)


def test_heartbeat_keeps_session_alive():
    settings = _test_settings()
    svc = SessionService(settings)
    user = AuthorizedUser(email="test@nobroker.in", name="Test", role="USER")
    session = svc.create_session(user)

    updated = svc.heartbeat(session.token, user_active=True, page_visible=True)
    assert updated.status == "ACTIVE"
    assert updated.last_seen_at >= session.last_seen_at
