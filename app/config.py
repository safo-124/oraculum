from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor all relative paths to the repository root, not the process working
# directory. A server started from any directory must see the same .env and
# the same uploads folder, otherwise records in the database and files on
# disk silently diverge.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or a local .env file.

    Every model is open-weight and runs on infrastructure you control: the
    chat model behind a llama.cpp server (LLM_BASE_URL), and the embedding,
    reranking and NLI encoders in-process with PyTorch. No API keys.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "oraculum"
    database_url: str | None = None

    # Ingestion. Chunk sizes are in the EMBEDDING model's tokens: bge-small
    # reads at most 512, so 350 leaves headroom for the tokenizer fallback.
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 50
    # "embedding" = count with the embedding model's Hugging Face tokenizer;
    # "regex" = word/punctuation fallback (offline tests).
    chunk_tokenizer: str = "embedding"
    max_file_size_bytes: int = 10 * 1024 * 1024
    uploads_dir: str = "data/uploads"

    # Local encoders (PyTorch via sentence-transformers). "cpu" or "cuda".
    model_device: str = "cpu"
    # Load every model at startup instead of on the first request.
    models_preload: bool = True

    # Embeddings and vector indexing
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    embedding_query_prefix: str = ""
    chroma_dir: str = "data/chroma"
    vector_collection: str = "documents"
    # 5 passages keep the prompt small enough for CPU inference.
    retrieval_top_k: int = Field(default=5, ge=1, le=20)

    retrieval_hybrid: bool = True
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    bm25_db_path: str = "data/bm25.db"

    # Cross-encoder reranking of the fused candidates.
    retrieval_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    # Answer generation: an open model behind llama.cpp's llama-server (or
    # any server with the same /v1/chat/completions wire format).
    llm_base_url: str = "http://127.0.0.1:8081"
    chat_model: str = "qwen2.5-1.5b-instruct"
    chat_max_tokens: int = 400
    chat_temperature: float = 0.0
    chat_timeout_seconds: float = 180.0
    chat_max_retries: int = 2
    # Request token log-probabilities (inputs to the confidence model).
    chat_logprobs: bool = True

    # Calibrated abstention (app/services/confidence.py).
    verifier_enabled: bool = True
    verifier_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    confidence_calibrator_path: str = "models/calibrator.json"
    # Overrides the threshold stored in the calibrator file when set.
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    # Multi-turn chat (pre-Phase 7)
    # Prior user/assistant turns replayed into the prompt when a request
    # carries a conversation_id. 0 disables history (stateless behaviour).
    # Bounded so an old thread cannot blow the context window.
    chat_history_turns: int = Field(default=4, ge=0, le=20)
    # Rewrite follow-up questions ("what about part-timers?") into a
    # standalone retrieval query - only when history is present; stateless
    # questions are already standalone and are never rewritten. On failure
    # the raw question is used, so this can never break a request.
    chat_query_rewrite: bool = True

    # Semantic answer cache (off until eval-validated).
    # A question whose embedding is at least min_similarity cosine-similar
    # to a cached one - within max_age_hours, against an unchanged corpus
    # fingerprint, same document filter and same model - is answered from
    # cache: no retrieval, no LLM call. Only "answered" responses are ever
    # stored (refusals cost nothing to regenerate and caching them could
    # hide newly indexed evidence). Any cache failure skips the cache and
    # the request continues down the normal path.
    chat_cache_enabled: bool = False
    chat_cache_min_similarity: float = Field(default=0.97, ge=0.0, le=1.0)
    chat_cache_max_age_hours: float = Field(default=24.0, gt=0.0)
    chat_cache_collection: str = "chat_cache"

    # Multi-query expansion (off until eval-validated).
    # The retrieval query is rephrased into N variants; original + variants
    # are embedded in one batch, searched on both the vector store and BM25,
    # and fused with RRF. Runs after the follow-up rewrite and applies to
    # stateless questions too. On any failure it degrades to the original
    # query alone, so it can never break a request.
    chat_query_expansion: bool = False
    chat_query_expansion_variants: int = Field(default=2, ge=1, le=5)

    # Reliability and security
    rate_limit_per_minute: int = 60  # 0 disables rate limiting
    api_key: str | None = None  # when set, all non-health endpoints require it

    # Request hardening (WS-4). Declared request bodies above the cap are
    # refused with 413 before parsing; per-file upload bounds are enforced
    # separately while streaming (max_file_size_bytes). 12 MB leaves slack
    # over the 10 MB upload limit for multipart framing.
    max_request_body_bytes: int = 12 * 1024 * 1024
    # Serve /docs, /redoc and /openapi.json. Set DOCS_ENABLED=false in
    # production so the API surface is not advertised; dev keeps it true
    # (the user tests through Swagger).
    docs_enabled: bool = True

    # Cost governance (WS-5). Global default budgets in USD; per-user
    # overrides live on the users table (admin-managed). None/0 disables
    # that window entirely. Enforcement only guards /chat* (the LLM spend
    # path) and fails open on ledger errors - telemetry never blocks.
    usage_daily_budget_usd: float | None = None
    usage_monthly_budget_usd: float | None = None
    # Fraction of a budget that flips the X-Usage-Warn header on.
    usage_warn_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    # Append one usage_events row per chat/retrieve/ingest call. Set
    # USAGE_LEDGER_ENABLED=false to stop writing (tests pin it false so the
    # suite never leaves rows in the developer's database).
    usage_ledger_enabled: bool = True

    # Identity and accounts (pre-Phase-7 WS-3)
    # Self-rolled email+password accounts with DB-backed session cookies.
    # The API_KEY is the ADMIN credential: a request bearing it acts as the
    # admin service account; every other client must log in. When API_KEY is
    # unset, a session is still required by default — anonymous callers would
    # otherwise spend LLM tokens. AUTH_ALLOW_ANONYMOUS=true opts back into
    # the old open local scope (tests and offline tooling only).
    auth_session_ttl_days: float = Field(default=30.0, gt=0.0)
    auth_cookie_name: str = "session"
    # Set true when serving behind HTTPS in production; local http keeps it
    # false so the browser will send the cookie back over http://127.0.0.1.
    auth_cookie_secure: bool = False
    auth_allow_registration: bool = True
    # Default OFF: every user logs in (protects token spend). The single
    # local scope without a session exists only when explicitly enabled.
    auth_allow_anonymous: bool = False
    auth_login_rate_limit_per_minute: int = 5  # per IP AND per email; 0 disables

    # Guest mode (public demo): visitors without an account may ask
    # questions about the SHARED corpus and read its documents. Guests
    # cannot upload, delete, save conversations, vote or see usage; their
    # chats are stateless. Every answer is CPU time on the server, so guests
    # get a tight per-IP limit plus a global per-minute cap.
    guest_mode_enabled: bool = False
    guest_rate_limit_per_minute: int = 4
    guest_global_rate_limit_per_minute: int = 20

    # Logging: rotating file in addition to stderr. Set LOG_FILE= (empty) to
    # disable file logging entirely.
    log_file: str = "data/logs/oraculum.log"

    # Frontend (dev server origins; production build is same-origin)
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @property
    def uploads_path(self) -> Path:
        """Absolute uploads directory; relative values resolve to the project root."""
        return self._project_path(self.uploads_dir)

    @property
    def chroma_path(self) -> Path:
        """Absolute ChromaDB persistence directory anchored to the project root."""
        return self._project_path(self.chroma_dir)

    @property
    def bm25_path(self) -> Path:
        """Absolute SQLite FTS5 (BM25) index file anchored to the project root."""
        return self._project_path(self.bm25_db_path)

    @property
    def calibrator_path(self) -> Path:
        return self._project_path(self.confidence_calibrator_path)

    @property
    def log_file_path(self) -> Path:
        """Absolute log-file path anchored to the project root."""
        return self._project_path(self.log_file)

    @staticmethod
    def _project_path(value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
