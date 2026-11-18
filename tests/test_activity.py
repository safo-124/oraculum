"""Admin Activity page: visit and question tracking, privacy, and the summary."""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from app.db.models import ActivityEvent
from app.services.activity import actor_kind, visitor_id
from app.api.auth import AuthActor
from app.config import Settings

from test_usage import PASSWORD, _db, _email, _promote_to_admin, _register

HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def _events(**where):
    session = _db()
    try:
        stmt = select(ActivityEvent)
        for key, value in where.items():
            stmt = stmt.where(getattr(ActivityEvent, key) == value)
        return list(session.scalars(stmt))
    finally:
        session.close()


def _clear() -> None:
    session = _db()
    try:
        session.execute(delete(ActivityEvent))
        session.commit()
    finally:
        session.close()


class TestVisitorIdentity:
    def test_hash_is_stable_salted_and_not_the_ip(self) -> None:
        class Req:
            client = type("C", (), {"host": "203.0.113.7"})()
            headers = {}

        a = visitor_id(Req(), Settings(analytics_salt="salt-a"))
        assert a == visitor_id(Req(), Settings(analytics_salt="salt-a"))
        assert a != visitor_id(Req(), Settings(analytics_salt="salt-b"))
        assert "203" not in a and len(a) == 16

    def test_actor_kinds(self) -> None:
        assert actor_kind(None) == "guest"
        assert actor_kind(AuthActor(mode="guest", role="guest", api_key_hash=None, user_id=None)) == "guest"
        assert actor_kind(AuthActor(mode="session", role="user", api_key_hash=None, user_id=uuid.uuid4())) == "user"
        assert actor_kind(AuthActor(mode="session", role="admin", api_key_hash=None, user_id=uuid.uuid4())) == "admin"


class TestTracking:
    def test_visit_beacon_records_guest_visit_without_ip(self, client_with_settings) -> None:
        _clear()
        client = client_with_settings(activity_tracking_enabled=True, analytics_salt="t")
        response = client.post("/analytics/visit", json={"path": "/chat?x=1"}, headers=HEADERS)
        assert response.status_code == 204
        rows = _events(kind="visit")
        assert len(rows) == 1
        assert rows[0].actor_kind in ("guest", "local")  # local when anonymous mode is on
        assert rows[0].path == "/chat"
        assert "testclient" not in rows[0].visitor

    def test_signed_in_visit_is_attributed(self, client_with_settings) -> None:
        _clear()
        client = client_with_settings(activity_tracking_enabled=True, analytics_salt="t")
        email = _email("visit-user")
        _register(client, email)
        client.post("/analytics/visit", json={"path": "/"}, headers=HEADERS)
        rows = _events(kind="visit")
        assert rows[-1].actor_kind == "user" and rows[-1].user_id is not None

    def test_tracking_off_records_nothing(self, client_with_settings) -> None:
        _clear()
        client = client_with_settings(activity_tracking_enabled=False)
        assert client.post("/analytics/visit", json={"path": "/"}, headers=HEADERS).status_code == 204
        assert _events() == []

    def test_chat_question_is_recorded_without_text(self, client_with_settings) -> None:
        _clear()
        client = client_with_settings(activity_tracking_enabled=True, analytics_salt="t")
        email = _email("asker")
        _register(client, email)
        client.post("/chat", json={"question": "secret question text"}, headers=HEADERS)
        rows = _events(kind="question")
        assert len(rows) == 1 and rows[0].actor_kind == "user"
        assert all("secret" not in (getattr(rows[0], c) or "") for c in ("visitor", "path", "kind"))


class TestSummary:
    def test_admin_summary_counts_guests_and_users(self, client_with_settings) -> None:
        _clear()
        settings = dict(activity_tracking_enabled=True, analytics_salt="t", guest_mode_enabled=True,
                        auth_allow_anonymous=False)
        # A signed-in user visits and asks twice.
        user = client_with_settings(**settings)
        email = _email("active")
        _register(user, email)
        user.post("/analytics/visit", json={"path": "/chat"}, headers=HEADERS)
        user.post("/chat", json={"question": "q1"}, headers=HEADERS)
        user.post("/chat", json={"question": "q2"}, headers=HEADERS)
        # A guest asks once (no cookie: a fresh client).
        guest = client_with_settings(**settings)
        guest.post("/analytics/visit", json={"path": "/"}, headers=HEADERS)
        guest.post("/chat", json={"question": "g1"}, headers=HEADERS)
        # An admin reads the page.
        admin = client_with_settings(**settings)
        admin_email = _email("activity-admin")
        _register(admin, admin_email)
        _promote_to_admin(admin_email)
        body = admin.get("/admin/activity?days=7").json()

        totals = body["totals"]
        assert totals["guest_questions"] == 1
        assert totals["guests_tried"] == 1
        assert totals["user_questions"] == 2
        assert totals["active_users"] >= 1  # other tests' uploaders share this database
        assert totals["visitors"] == 1  # same test-client IP => one hashed visitor
        assert len(body["daily"]) == 7
        assert sum(d["user_questions"] for d in body["daily"]) == 2
        row = next(u for u in body["users"] if u["email"] == email)
        assert row["questions"] == 2 and row["last_active"] is not None
        assert all(u["email"] != admin_email for u in body["users"])  # admins excluded

    def test_regular_users_cannot_read_it(self, client_with_settings) -> None:
        client = client_with_settings()
        _register(client, _email("nosy"))
        assert client.get("/admin/activity").status_code == 403
