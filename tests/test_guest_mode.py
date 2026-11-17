"""Guest mode: anonymous visitors may chat about the shared corpus only."""

from fastapi.testclient import TestClient

from app.api.auth import AuthActor, answer_cache_owner_key, document_owner_filter
from app.api.chat import get_generator
from app.api.documents import get_embedder, get_storage, get_vector_store
from app.api.retrieve import get_bm25_index, get_reranker
from app.config import Settings, get_settings
from app.main import create_app
from app.services.rate_limit import FixedWindowRateLimiter, get_rate_limiter
from app.services.storage import DocumentStorage

GUEST = AuthActor(mode="guest", role="guest", api_key_hash=None, user_id=None)


def _guest_client(tmp_path, fakes, limiter=None, **overrides) -> TestClient:
    embedder, store, generator, bm25 = fakes
    app = create_app()
    app.dependency_overrides[get_storage] = lambda: DocumentStorage(tmp_path / "uploads")
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_generator] = lambda: generator
    app.dependency_overrides[get_bm25_index] = lambda: bm25
    app.dependency_overrides[get_reranker] = lambda: None
    shared_limiter = limiter or FixedWindowRateLimiter()
    app.dependency_overrides[get_rate_limiter] = lambda: shared_limiter
    settings = dict(
        guest_mode_enabled=True,
        auth_allow_anonymous=False,
        api_key=None,
        chat_cache_enabled=False,
        usage_ledger_enabled=False,
        usage_daily_budget_usd=None,
        usage_monthly_budget_usd=None,
        guest_rate_limit_per_minute=100,
        guest_global_rate_limit_per_minute=100,
    )
    settings.update(overrides)
    app.dependency_overrides[get_settings] = lambda: Settings(**settings)
    return TestClient(app, headers={"X-Requested-With": "XMLHttpRequest"})


def _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25):
    return fake_embedder, fake_vector_store, fake_generator, fake_bm25


class TestGuestScope:
    def test_guest_sees_shared_corpus_only(self) -> None:
        assert document_owner_filter(GUEST) is None
        assert answer_cache_owner_key(GUEST) == "guest"

    def test_auth_config_advertises_guest_mode(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25) -> None:
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25))
        body = client.get("/auth/config").json()
        assert body["guest_mode"] is True and body["auth_required"] is True

    def test_guest_mode_never_advertised_with_api_key(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25) -> None:
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25), api_key="k" * 32)
        assert client.get("/auth/config").json()["guest_mode"] is False


class TestGuestRoutes:
    def test_guest_can_chat_statelessly(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25, monkeypatch) -> None:
        import app.api.chat as chat_module

        seen = []
        monkeypatch.setattr(chat_module, "_visible_document_scope", lambda actor, doc: seen.append(actor.mode) or set())
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25))
        response = client.post("/chat", json={"question": "leave?", "conversation_id": None})
        assert response.status_code == 200
        assert response.json()["conversation_id"] is None  # nothing stored
        assert seen == ["guest"]

    def test_guest_cannot_upload_or_list_conversations(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25) -> None:
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25))
        upload = client.post("/documents", files={"file": ("a.md", b"# hi", "text/markdown")})
        assert upload.status_code == 401
        assert client.get("/conversations").status_code == 401
        assert client.get("/usage/me").status_code == 401
        assert client.post("/retrieve", json={"question": "x"}).status_code == 401

    def test_guest_mode_off_requires_sign_in(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25) -> None:
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25), guest_mode_enabled=False)
        assert client.post("/chat", json={"question": "leave?"}).status_code == 401


class TestGuestRateLimits:
    def test_per_ip_limit(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25, monkeypatch) -> None:
        import app.api.chat as chat_module

        monkeypatch.setattr(chat_module, "_visible_document_scope", lambda actor, doc: set())
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25),
                               guest_rate_limit_per_minute=2)
        codes = [client.post("/chat", json={"question": "q"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]

    def test_global_cap_is_shared_across_ips(self, tmp_path, fake_embedder, fake_vector_store, fake_generator, fake_bm25, monkeypatch) -> None:
        import app.api.chat as chat_module

        monkeypatch.setattr(chat_module, "_visible_document_scope", lambda actor, doc: set())
        limiter = FixedWindowRateLimiter()
        client = _guest_client(tmp_path, _fakes(fake_embedder, fake_vector_store, fake_generator, fake_bm25),
                               limiter=limiter, guest_global_rate_limit_per_minute=2)
        codes = [
            client.post("/chat", json={"question": "q"}, headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
            for i in range(3)
        ]
        assert codes[-1] == 429
