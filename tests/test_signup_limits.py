"""Self-registration guards and per-account upload quotas (public demo)."""

import uuid

from app.db import repositories
from app.services.rate_limit import FixedWindowRateLimiter, get_rate_limiter

from test_usage import PASSWORD, _db, _email, _promote_to_admin, _register

HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def _upload(client, name: str, body: bytes):
    return client.post(
        "/documents", files={"file": (name, body, "text/markdown")}, headers=HEADERS
    )


def _users() -> int:
    session = _db()
    try:
        return repositories.count_users(session)
    finally:
        session.close()


class TestRegistrationGuards:
    def test_sign_ups_are_rate_limited_per_ip(self, client_with_settings) -> None:
        client = client_with_settings(auth_register_rate_limit_per_minute=2)
        limiter = FixedWindowRateLimiter()  # the fixture's limiter allows everything
        client.app.dependency_overrides[get_rate_limiter] = lambda: limiter
        codes = [
            client.post(
                "/auth/register",
                json={"email": _email("burst"), "password": PASSWORD},
                headers=HEADERS,
            ).status_code
            for _ in range(3)
        ]
        assert codes == [201, 201, 429]

    def test_account_cap_blocks_new_sign_ups(self, client_with_settings) -> None:
        client = client_with_settings(auth_max_users=_users())
        response = client.post(
            "/auth/register",
            json={"email": _email("capped"), "password": PASSWORD},
            headers=HEADERS,
        )
        assert response.status_code == 403
        assert "guest" in response.json()["detail"]

    def test_no_cap_by_default(self, client_with_settings) -> None:
        _register(client_with_settings(), _email("uncapped"))


class TestUploadQuota:
    def test_document_count_quota(self, client_with_settings) -> None:
        client = client_with_settings(upload_max_documents_per_user=1)
        _register(client, _email("quota-docs"))
        assert _upload(client, "a.md", b"# First " + uuid.uuid4().bytes.hex().encode()).status_code == 202
        second = _upload(client, "b.md", b"# Second " + uuid.uuid4().bytes.hex().encode())
        assert second.status_code == 403
        assert "1 documents per account" in second.json()["detail"]

    def test_byte_quota_counts_the_incoming_file(self, client_with_settings) -> None:
        client = client_with_settings(upload_max_bytes_per_user=1024 * 1024)
        _register(client, _email("quota-bytes"))
        big = b"# Big\n" + b"x" * (1024 * 1024)
        response = _upload(client, "big.md", big)
        assert response.status_code == 403
        assert "1 MB" in response.json()["detail"]
        assert _upload(client, "small.md", b"# Small " + uuid.uuid4().bytes.hex().encode()).status_code == 202

    def test_admins_are_exempt(self, client_with_settings) -> None:
        client = client_with_settings(upload_max_documents_per_user=0)
        email = _email("quota-admin")
        _register(client, email)
        _promote_to_admin(email)
        assert _upload(client, "admin.md", b"# Admin " + uuid.uuid4().bytes.hex().encode()).status_code == 202
