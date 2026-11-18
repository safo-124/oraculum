# Oraculum

A retrieval-augmented question-answering system that runs entirely on **open models**, cites its evidence, and **knows when not to answer**. It ingests documents (PDF, TXT, Markdown), retrieves passages with hybrid search and a cross-encoder reranker, and answers with a small open language model. Each answer then gets a **calibrated confidence score**: the estimated probability that it is correct. Answers below a threshold, chosen to meet a target error rate, are withheld instead of shown.

**Live demo:** see the link at the top of the GitHub page. No account is needed: ask about the demo documents. It runs on CPU, so answers take 10–30 seconds.

## Research question: when should a RAG system refuse to answer?

A grounded-looking answer can still be wrong. The model may cite a passage that is related but doesn't contain the answer, or write an answer the evidence does not support. Oraculum treats "should I show this answer?" as a selective-prediction problem. It combines three independent views of the same answer:

| View | Signals | Model |
|---|---|---|
| Retrieval | relevance of the best passage, lead over the runner-up | cross-encoder `ms-marco-MiniLM-L-6-v2` |
| Generation | the model's own P(answer vs. abstain), mean and minimum token log-probability of the answer | Qwen2.5-Instruct log-probabilities via llama.cpp |
| Verification | does a cited passage *entail* the answer (and the question + answer)? contradiction probability | NLI cross-encoder `nli-deberta-v3-xsmall` |

A logistic calibrator maps the signals to P(correct). The study ([`research/`](research/)) measures:

- **which signals actually predict correctness** (AUROC, area under the risk-coverage curve), single signals vs. groups vs. all combined;
- **whether the scores are calibrated** (expected calibration error, Brier score, reliability diagram);
- **what the gate does for users**: out-of-sample coverage and error rate at a target risk, and how often the system answers questions that have *no* answer;
- **whether it transfers**: a calibrator fitted on Wikipedia (SQuAD 2.0) is frozen and applied to the in-domain policy documents.

The benchmark is SQuAD 2.0 dev, whose unanswerable questions were written adversarially against the same paragraphs, so retrieval always finds plausible but insufficient evidence. All evaluation uses **cross-validation grouped by document**, so every score comes from a calibrator that never saw that document. The gate threshold is chosen inside the training folds, and the 95% confidence intervals come from bootstrapping over documents.

```bash
uv run python -m research build-squad            # sample SQuAD 2.0 dev (12 articles, balanced)
uv run python -m research build-acme             # in-domain set from evals/ + samples/
uv run python -m research run squad              # full pipeline per question, resumable
uv run python -m research run acme
uv run python -m research study --transfer acme  # evals/study/REPORT.md + models/calibrator.json
```

Results: [`evals/study/REPORT.md`](evals/study/REPORT.md), written by the last command.

## Stack

- **Language model:** Qwen2.5-Instruct (0.5B on the 4 GB demo server, 1.5B recommended; GGUF, 4-bit) served by [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, using JSON-schema grammar decoding and token log-probabilities. Any server with the `/v1/chat/completions` wire format works (Ollama, vLLM, TGI).
- **Encoders (PyTorch, Hugging Face):** `BAAI/bge-small-en-v1.5` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, `cross-encoder/nli-deberta-v3-xsmall` answer verifier, all through sentence-transformers
- **API:** Python 3.13, FastAPI + Uvicorn, [uv](https://docs.astral.sh/uv/)
- **Retrieval:** ChromaDB (vectors) + SQLite FTS5 (BM25), fused with Reciprocal Rank Fusion, then reranked
- **Relational database:** PostgreSQL via SQLAlchemy 2.0 + Alembic
- **Frontend:** React 19 + TypeScript + Vite + Tailwind
- **Testing:** pytest (528 tests, no model downloads needed), vitest (59 tests)
- **Deployment:** systemd + Caddy on a single CPU server ([`deploy/README.md`](deploy/README.md))

No external AI API is used anywhere, and no API key is needed.

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL (local instance; needed from Phase 5 onward)

### Install

```powershell
git clone <repo-url>
cd oraculum
uv sync                    # CPU PyTorch; add --extra research for the study's plots
```

### Start the model server

Download a llama.cpp release for your OS from
<https://github.com/ggml-org/llama.cpp/releases> and the model file
`qwen2.5-1.5b-instruct-q4_k_m.gguf` from
<https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF>, then:

```powershell
llama-server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --host 127.0.0.1 --port 8081 --ctx-size 4096
```

The embedding, reranking and NLI encoders download from Hugging Face on
first use (about 250 MB). The test suite never downloads anything.

### Configure environment

Copy `.env.example` to `.env` and fill in your own values:

```powershell
Copy-Item .env.example .env
```

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (`postgresql://user:pass@host:port/db`) |
| `LLM_BASE_URL` | The model server (default `http://127.0.0.1:8081`, llama.cpp `llama-server`) |
| `CHAT_MODEL` | Model name reported in responses and logs (default `qwen2.5-1.5b-instruct`) |
| `CHAT_MAX_TOKENS` | Max tokens per generated answer (default `400`) |
| `CHAT_TEMPERATURE` | Sampling temperature (default `0.0`) |
| `CHAT_TIMEOUT_SECONDS` | Model-server request timeout (default `180`; CPU inference is slow) |
| `CHAT_LOGPROBS` | Request token log-probabilities for the confidence model (default `true`) |
| `MODEL_DEVICE` | Device for the PyTorch encoders: `cpu` or `cuda` (default `cpu`) |
| `MODELS_PRELOAD` | Load all encoders at startup instead of on first request (default `true`) |
| `EMBEDDING_MODEL` | Hugging Face embedding model (default `BAAI/bge-small-en-v1.5`) |
| `EMBEDDING_BATCH_SIZE` | Texts per encoder batch (default `32`) |
| `EMBEDDING_QUERY_PREFIX` | Instruction prepended to queries only (default empty) |
| `RETRIEVAL_RERANK` | Rerank fused candidates with a cross-encoder (default `true`) |
| `RERANK_MODEL` | Cross-encoder reranker (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `RERANK_CANDIDATES` | Candidates retrieved before reranking down to top-k (default `20`) |
| `VERIFIER_ENABLED` | NLI check that cited passages entail the answer (default `true`) |
| `VERIFIER_MODEL` | NLI cross-encoder (default `cross-encoder/nli-deberta-v3-xsmall`) |
| `CONFIDENCE_CALIBRATOR_PATH` | Fitted calibrator; answers are only gated when it exists (default `models/calibrator.json`) |
| `CONFIDENCE_THRESHOLD` | Override the calibrator's threshold (default unset) |
| `CHUNK_SIZE_TOKENS` | Target chunk size in *embedding-model* tokens (default `350`; bge reads at most 512) |
| `CHUNK_OVERLAP_TOKENS` | Overlap between adjacent chunks (default `50`) |
| `CHUNK_TOKENIZER` | `embedding` (the embedding model's tokenizer) or `regex` (offline fallback) |
| `MAX_FILE_SIZE_BYTES` | Upload size limit (default `10485760`) |
| `UPLOADS_DIR` | Where uploaded files are stored (default `data/uploads`) |
| `CHROMA_DIR` | ChromaDB persistence directory (default `data/chroma`) |
| `RETRIEVAL_TOP_K` | Passages given to the model for `/retrieve` + `/chat` (default `5`; per-request `top_k` overrides) |
| `RETRIEVAL_HYBRID` | Fuse BM25 term search with vector search (default `true`; `false` = legacy vector-only) |
| `RETRIEVAL_RRF_K` | Reciprocal Rank Fusion constant (default `60`) |
| `BM25_DB_PATH` | SQLite FTS5 term-index file (default `data/bm25.db`) |
| `AUTH_REGISTER_RATE_LIMIT_PER_MINUTE` | Sign-ups per IP per minute (default `3`; `0` disables) |
| `AUTH_MAX_USERS` | Ceiling on the number of accounts (default unset = unlimited) |
| `UPLOAD_MAX_DOCUMENTS_PER_USER` | Per-account document quota for regular users (default unset) |
| `UPLOAD_MAX_BYTES_PER_USER` | Per-account upload bytes quota for regular users (default unset) |
| `GUEST_MODE_ENABLED` | Public demo: visitors chat about the shared corpus without an account (default `false`) |
| `GUEST_RATE_LIMIT_PER_MINUTE` | Guest questions per IP per minute (default `4`) |
| `GUEST_GLOBAL_RATE_LIMIT_PER_MINUTE` | Guest questions per minute across all visitors (default `20`) |
| `CHAT_HISTORY_TURNS` | Prior turns replayed into the prompt on `conversation_id` requests (default `4`; `0` disables history) |
| `CHAT_QUERY_REWRITE` | Rewrite follow-up questions into a standalone retrieval query when history exists (default `true`; stateless questions are never rewritten) |
| `CHAT_CACHE_ENABLED` | Semantic answer cache: an embedding-similar question is answered from cache (default `false` — off until eval-validated) |
| `CHAT_CACHE_MIN_SIMILARITY` | Cosine similarity needed for a cache hit (default `0.97`) |
| `CHAT_CACHE_MAX_AGE_HOURS` | Cached answers older than this are ignored and pruned (default `24`) |
| `CHAT_QUERY_EXPANSION` | Multi-query expansion: rephrase the query into variants and fuse all rankings (default `false` — off until eval-validated) |
| `CHAT_QUERY_EXPANSION_VARIANTS` | Expansion variants per query when enabled (default `2`; `1`–`5`) |
| `RATE_LIMIT_PER_MINUTE` | Per-IP cap on `/chat` + `/retrieve` (default `60`; `0` disables) |
| `API_KEY` | When set, scripts/curl must send `X-API-Key: <key>` (401 otherwise). Browser users instead sign in — see *Identity and access* below |
| `AUTH_ALLOW_ANONYMOUS` | Opt-in open mode for offline tooling: no key and no session passes (default `false` — sign-in required; never enable in a multi-user deployment) |
| `AUTH_ALLOW_REGISTRATION` | Whether `POST /auth/register` accepts new signups (default `true`) |
| `AUTH_SESSION_TTL_DAYS` | Login session lifetime in days (default `30`) |
| `AUTH_COOKIE_SECURE` | Send the session cookie over HTTPS only (default `false`; set `true` behind TLS) |
| `AUTH_LOGIN_RATE_LIMIT_PER_MINUTE` | Per-IP **and** per-email cap on `/auth/login` (default `5`; `0` disables) |
| `DOCS_ENABLED` | Serve `/docs`, `/redoc` and `/openapi.json` (default `true`; set `false` on public deployments) |
| `USAGE_LEDGER_ENABLED` | Append one `usage_events` cost row per chat/retrieve/ingest request (default `true`; ledger failures never break a request) |
| `USAGE_DAILY_BUDGET_USD` | Global default daily spend cap in USD (default unset = no cap) |
| `USAGE_MONTHLY_BUDGET_USD` | Global default monthly spend cap in USD (default unset = no cap) |
| `USAGE_WARN_RATIO` | Fraction of a budget past which chat responses carry `X-Usage-Warn` (default `0.8`) |
| `LOG_FILE` | Rotating application log file (default `data/logs/oraculum.log`; set empty to disable) |

`.env` is git-ignored. Never commit real credentials.

Note: `DATABASE_URL` accepts the plain PostgreSQL format (`postgresql://...`) or the SQLAlchemy dialect format (`postgresql+psycopg://...`). Both are normalized automatically.

### Create the database

```powershell
# in psql or your preferred client, once:
CREATE DATABASE rag_assistant;
```

Then apply the schema migrations:

```powershell
uv run alembic upgrade head
```

### Database migrations (Alembic)

The schema is managed by Alembic. When you change a model in `app/db/models.py`:

```powershell
# 1. generate a migration from the model changes
uv run alembic revision --autogenerate -m "describe your change"

# 2. review the generated file in alembic/versions/

# 3. apply it
uv run alembic upgrade head

# check current revision / history
uv run alembic current
uv run alembic history
```

Applied migration chain (newest last — run `uv run alembic upgrade head`
after pulling to bring a database up to it):

| Revision | Change |
|---|---|
| `2676f16ec739` | `documents` table |
| `bbfc9285daca` | embedding provenance columns |
| `d4f8a2c1e6b7` | `documents.content_hash` (idempotent dedup) |
| `f3a9c2d8e1b4` | `conversations` + `chat_messages` (chat history) |
| `a1c7e9f4b2d8` | `chat_messages.feedback` column |
| `c7d2e5a91f3b` | identity: `users`, `sessions`, `conversations.user_id` |
| `e5a8f3c1d9b6` | `audit_events` (admin action trail) |
| `a9d4e2b7c5f1` | `documents.owner_user_id` (per-user document scoping) |
| `b7f3d9a2c4e8` | `usage_events` + per-user budget overrides (cost governance) |

## Run

```powershell
# model server (separate terminal; see "Start the model server")
llama-server --model qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8081 --ctx-size 4096

# backend
uv run uvicorn app.main:app --reload

# shared demo documents (guests and every account can ask about them)
uv run python scripts/load_demo_corpus.py samples

# frontend (separate terminal, dev server with API proxy on :5173)
cd frontend
npm run dev
```

- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- UI (dev): http://localhost:5173 — Vite proxies `/chat`, `/documents`,
  `/retrieve`, and `/health` to the backend
- UI (production): `cd frontend && npm run build`. Caddy serves
  `frontend/dist` and proxies the API paths on the same origin, so no CORS
  configuration is needed (see [`deploy/README.md`](deploy/README.md))

### Logs

Application logs (requests with their `X-Request-ID`, ingestion progress,
errors with tracebacks) are written to `data/logs/oraculum.log` in addition
to stderr. The file rotates at 5 MB and keeps 3 backups. Tail it live:

```powershell
Get-Content data\logs\oraculum.log -Wait -Tail 50
```

Set `LOG_FILE=` (empty) in `.env` to disable file logging.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/auth/config` | Public auth configuration (`auth_required`, `registration_enabled`) — drives the SPA guard |
| `POST` | `/auth/register` | Create an account (`role: user`; admins are seeded only via `scripts/create_admin.py`) |
| `POST` | `/auth/login` | Sign in → session cookie |
| `POST` | `/auth/logout` | Revoke the current session and clear the cookie |
| `GET` | `/auth/me` | The signed-in account (`401` when signed out) |
| `POST` | `/documents` | Upload a PDF/TXT/Markdown file; validates synchronously, then extracts/chunks/embeds/indexes it in the background (`202` + `status: pending`; poll `GET /documents` for progress) |
| `GET` | `/documents` | List ingested documents and their statuses (scoped to your uploads + the shared corpus) |
| `GET` | `/documents/{id}` | Document record, ingestion status, and indexing config used |
| `GET` | `/documents/{id}/chunks` | Re-chunk a stored document with the current chunking configuration |
| `GET` | `/documents/{id}/content` | Extracted page-level text (drives the in-app document viewer) |
| `GET` | `/documents/{id}/file` | Raw stored file bytes (drives the PDF viewer iframe) |
| `DELETE` | `/documents/{id}` | Remove a document's vectors, stored file, and record |
| `POST` | `/retrieve` | Semantic search: question in, top-k chunks out with scores and citations |
| `POST` | `/chat` | Grounded Q&A: short answer with resolved source citations, or an explicit insufficient-evidence refusal; pass `conversation_id` to persist the exchange |
| `POST` | `/chat/stream` | Same request contract as `/chat`, answered as Server-Sent Events (`meta` → `delta`* → `done` \| `error`) |
| `PUT` | `/chat/messages/{id}/feedback` | Thumbs on one stored answer: `{"rating": "up" \| "down" \| null}` (`null` clears); 404 outside your chat scope |
| `GET` | `/conversations` | List saved chats (id, title, message count, activity) |
| `GET` | `/conversations/{id}` | Full thread: title plus every stored message in order |
| `PATCH` | `/conversations/{id}` | Rename a chat |
| `DELETE` | `/conversations/{id}` | Delete a chat and all of its messages |
| `GET` | `/usage/me` | Your own cost ledger: window/lifetime spend + tokens, effective budgets (404 for API-key/local callers) |
| `GET` | `/admin/overview` | *(admin)* Health, version, git commit, corpus/accounts counts, cache flag |
| `GET` | `/admin/users` · `PATCH /admin/users/{id}` | *(admin)* Accounts; role/activation changes and per-user budget overrides (audited) |
| `POST` | `/admin/users/{id}/password-reset` | *(admin)* Force a password reset; revokes all live sessions |
| `GET` | `/admin/usage` | *(admin)* Cost rollups: daily spend, totals, `by_user`, `by_model` over `?days=N` |
| `GET` | `/admin/feedback` | *(admin)* Recent 👍/👎 with answer context |
| `GET` | `/admin/audit` | *(admin)* Append-only audit trail of admin/security actions |
| `GET` | `/health` | Process liveness only |
| `GET` | `/health/ready` | Dependency readiness: Postgres + vector store must respond (503 if down) |

Try it with the committed sample documents:

```powershell
# via Swagger UI: open http://127.0.0.1:8000/docs and upload samples/employee_handbook.md
# or via curl (returns 202 immediately; ingestion continues in the background):
curl -F "file=@samples/employee_handbook.md" http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/documents   # poll until status is "completed" (or "failed")

# semantic retrieval (no LLM involved):
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" `
  -d "{\"question\": \"How many days of annual leave?\"}"

# grounded answer with citations:
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" `
  -d "{\"question\": \"How many days of annual leave?\"}"

# persistent thread: conversation_id: null creates a chat and returns its id;
# sending that id back appends to it. Omitting the key entirely stays stateless.
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" `
  -d "{\"question\": \"How many days of annual leave?\", \"conversation_id\": null}"
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" `
  -d "{\"question\": \"What about carry-over?\", \"conversation_id\": \"<id from previous reply>\"}"
curl http://127.0.0.1:8000/conversations

# streaming answer (SSE): meta first, then delta text, then the authoritative
# `done` payload - replace the streamed text with done.answer when it lands.
curl -N -X POST http://127.0.0.1:8000/chat/stream -H "Content-Type: application/json" `
  -d "{\"question\": \"How many days of annual leave?\"}"

# thumbs feedback on a stored answer (message_id comes back in the chat
# response; null clears a previous vote):
curl -X PUT http://127.0.0.1:8000/chat/messages/<message_id>/feedback `
  -H "Content-Type: application/json" -d "{\"rating\": \"up\"}"

# or the standalone smoke script:
uv run python scripts/verify_index.py "How many days of annual leave?"
```

### Hybrid retrieval (BM25 + vector)

Retrieval runs two rankings and fuses them with Reciprocal Rank Fusion:

- **Vector search** — cosine similarity over local bge-small embeddings (ChromaDB).
- **BM25 term search** — SQLite FTS5 with porter stemming, in `data/bm25.db`.
  No new dependencies; the row ids mirror Chroma's `{document_id}#{chunk_index}`
  so both rankings fuse on the same keys.

Each side ranks its top-k; every chunk votes `1 / (rrf_k + rank)` and chunks
found by both sides win. This fixes a flat-cosine failure mode where a question
literally containing a distinctive term ("principles") missed the chunk holding
it because the scores around the cutoff differed by ~0.001. **Scores are fused
rank support, not similarity** — they are for ordering, never confidence.

Consistency policy: a term-index write failure during ingestion **fails the
document and rolls back its vectors** (a `failed` document is never retrievable
through either path); a term-index *query* failure only degrades the request to
vector-only search (logged as a warning). Deleting a document removes its FTS
rows first, then its vectors — every step is idempotent, so retries are safe.

Upgrading an existing index (one-time; your documents are untouched):

```powershell
uv run python scripts/backfill_bm25.py   # rebuilds data/bm25.db from stored texts
```

Set `RETRIEVAL_HYBRID=false` to restore the exact legacy vector-only path.

### How grounding works

1. The question is retrieved through the hybrid pipeline above (same as `/retrieve`).
2. No evidence at all -> a deterministic refusal; the LLM is never called.
3. Evidence is passed to the chat model as clearly-labelled untrusted data; the trusted
   system prompt forbids following instructions found inside documents and requires
   structured JSON output (answer + cited evidence labels).
4. Citations are resolved in application code against the real retrieved chunks:
   invented labels are dropped, and an "answered" result with no resolvable citation
   is downgraded to `insufficient_evidence` so uncited claims are never presented as
   grounded facts.

### Streaming, follow-ups, and feedback (pre-Phase 7)

Three chat capabilities added before the Phase 7 hardening push. All of them are
additive: the plain `POST /chat` contract and its stateless path are unchanged,
so existing scripts, tests, and eval numbers stay exactly where they were.

**Streaming — `POST /chat/stream`.** Identical request body to `/chat`, answered
as Server-Sent Events:

| Event | Payload | Meaning |
|---|---|---|
| `meta` | `model`, `retrieval_ms`, `rewritten`, `expanded`, `has_evidence`, `cached` | Retrieval finished; the UI can react before any text exists |
| `delta` | `{"text": ...}` | Incremental answer text, decoded from the model's JSON `answer` field |
| `done` | full `ChatResponse` | **Authoritative** — validated citations, status, timings, `conversation_id`, `message_id` |
| `error` | `{"message": ...}` | Generation failed mid-stream (mirrors the plain endpoint's 503) |

`delta` text is cosmetic: the client renders it live but **must replace it with
`done.answer`** when `done` lands. Failures before the stream starts (422, 404,
rate limit, retrieval outage) are still normal HTTP status codes. Retries after
a transient error only happen before the first delta is emitted — replaying
text already on the wire would duplicate it, so a mid-stream failure surfaces
as `error` instead.

**Multi-turn context + query rewrite.** Requests carrying a `conversation_id`
load the last `CHAT_HISTORY_TURNS` exchanges (default 4) into the prompt as a
labelled transcript inside the single user message — every untrusted string
(earlier user turns *and* earlier model output) stays in one block, never in the
trusted system prompt. Because a follow-up like "what about part-timers?" is
not standalone, the same request rewrites it into a retrieval query first
(`CHAT_QUERY_REWRITE`, default `true`); the stored/displayed question stays
raw. The rewrite is best-effort: any failure falls back to the raw question,
and **stateless requests are never rewritten**, so single-shot Q&A behaves
byte-for-byte as before. A missing or foreign `conversation_id` 404s before
any LLM spend.

**Feedback — `PUT /chat/messages/{id}/feedback`.** Persisted answers carry a
`message_id` (in the response body, or in the stream's `done` event) that takes
`{"rating": "up" | "down"}`; sending `null` clears a vote, and clicking the
active thumb again does the same. Votes live in `chat_messages.feedback`
(nullable column, one Alembic migration) and are scoped to the owning API-key
fingerprint — a wrong scope gets 404, never a probe oracle. `GET
/conversations/{id}` returns `feedback` per message so threads reload with
their votes intact. `scripts/export_feedback.py` dumps votes for analysis.

### Semantic answer cache + multi-query expansion (post-freeze, opt-in)

Two flag-gated additions after the Phase 6 baseline freeze. **Both default
off**; with the defaults the chat path is byte-identical to the frozen
baseline and `evals/results/latest.json` reproduces exactly (verified —
r@3=1.0, r@5=1.0, MRR=0.984, no-answer max top-1=0.033 both with the
defaults and with `CHAT_QUERY_EXPANSION=true`, which was neutral on the
retrieval eval, so the flag stays off until a dataset shows a win).

**Semantic answer cache (`CHAT_CACHE_ENABLED`, default `false`).** A second
collection in the same on-disk Chroma instance (`chat_cache`) keys answered
responses by question embedding. A request whose embedding is at least
`CHAT_CACHE_MIN_SIMILARITY` (default 0.97) cosine-similar to a cached one —
within `CHAT_CACHE_MAX_AGE_HOURS` (default 24), same corpus fingerprint,
same document filter, same model — is answered with **no retrieval and no
LLM call** (`cached: true`, `cache_similarity` in the response; one `delta`
then `done` on the stream). Correctness rests on the corpus fingerprint
(sha256 over every document row's id, content hash, and status): any
ingest or delete makes every older entry unreachable, and the wipe on
those paths is hygiene. Refusals are never cached; cached exchanges are
still persisted to the conversation; every cache failure (probe, lookup,
store) degrades to the normal answer path, never fails the request. The
probe's embedding is reused as the retrieval vector when the query isn't
rewritten, so a miss pays one extra API call at most.

**Multi-query expansion (`CHAT_QUERY_EXPANSION`, default `false`).** The
standalone retrieval query (after any follow-up rewrite; stateless
questions included) is rephrased into `CHAT_QUERY_EXPANSION_VARIANTS`
(default 2) variants; original + variants are embedded in one batch,
searched on both the vector store and BM25, and every ranking is fused
with the existing RRF — so a chunk matched by several phrasings outranks a
single-phrasing hit. Like the rewrite, it's best-effort: any failure
returns no variants and retrieval proceeds with the original query only.
The stream's `meta` event reports the variant count as `expanded`.

Retrieval results are reranked by a cross-encoder (`RETRIEVAL_RERANK`); an LLM-as-judge eval scorer remains deferred.

### Reliability and security (Phase 5)

- **Shared model-server client** — one cached `httpx` client per (server,
  model, timeout) reuses connections to llama-server across requests;
  encoders are loaded once per process and warmed at startup.
- **Request tracing** — every request gets an `X-Request-ID` (incoming value is
  echoed, otherwise generated); log lines during the request carry the same ID.
- **Background ingestion** — uploads only validate and persist the file plus
  a `pending` row (`202 Accepted`); extraction, chunking, embedding, and
  vector indexing run as a FastAPI background task, so a large PDF never
  blocks the API or the chat. A process-wide semaphore ingests one document
  at a time. Any failure is stamped on the row (`failed` + `error_message`)
  instead of failing the upload. On startup, rows stuck in
  `pending`/`processing` (e.g. after a crash) are re-queued automatically.
- **Idempotent uploads** — SHA-256 of the file bytes is stored per document;
  re-uploading identical content returns the existing document with HTTP 200
  when it is already indexed, or HTTP 202 when it is still being indexed,
  never a duplicate record. Identical bytes that previously failed are
  retried in place (same row, error cleared).
- **Rate limiting** — per-IP fixed window on `/chat` and `/retrieve`
  (default 60/min, HTTP 429 + `Retry-After`); `/documents`, `/auth/login`
  (its own per-IP + per-email limit), and health have separate rules.
- **Dependency-aware readiness** — `/health` is process liveness only;
  `/health/ready` checks Postgres and the vector store and returns 503 when
  a critical dependency is down.
- **Chat history** — exchanges are saved server-side in Postgres when the
  `/chat` request includes a `conversation_id` (`null` creates a new chat,
  a UUID appends, omission keeps the endpoint stateless for scripts).
  Each chat is scoped to its owner (see identity below); API-key history
  is keyed on a SHA-256 fingerprint of the key, so changing `API_KEY`
  hides the previous key's history. Nothing is auto-purged; chats are
  deletable individually via `DELETE /conversations/{id}`.

### Identity and access

Browser users sign in; scripts and evals keep using the raw API key.

- **Session auth (default)** — `POST /auth/login` sets an HttpOnly `session`
  cookie (Argon2-hashed passwords in Postgres, 30-day TTL, revocable). The
  SPA checks `GET /auth/config` on load and redirects everything to `/login`
  (`auth_required` is `true` by default: `AUTH_ALLOW_ANONYMOUS=false` and it
  stays that way in any multi-user deployment — anonymous callers are pure
  token-spend risk).
- **API-key mode** — with `API_KEY` set, `X-API-Key` authenticates scripts
  exactly as before: key-auth requests touch **no database** (no budget
  check, no scoping DB reads on the chat path), so curl/evals keep their
  pre-identity behavior byte-for-byte.
- **Account creation** — `POST /auth/register` always creates `role: user`.
  Admins are seeded only from the terminal via
  `uv run python scripts/create_admin.py <email>` (promotes an existing
  user or creates one). Admin routes are role-gated and every mutation is
  written to the append-only `audit_events` trail.
- **Per-user scoping** — uploads are stamped with the owner; a document is
  visible to its owner plus the shared corpus, never to other users (reads,
  deletes, dedup, retrieval, and even the answer cache are all owner-bucketed —
  a foreign document id is indistinguishable from a missing one, 404 on
  every surface). Session admins see the whole corpus as an operational view.
- **Swagger** — `/docs` carries OpenAPI security schemes for the session
  cookie and `X-API-Key`, so the **Authorize** button works out of the box.

### Request hardening

- **CSRF** — cookie mutations must send `X-Requested-With` (the SPA adds it
  on every fetch); cross-origin `POST/PUT/PATCH/DELETE` without a matching
  `Host`/`Origin` gets `403`, including `Origin: null`. Bodies are capped
  (13 MB → `413`). API-key callers are exempt (they don't carry cookies).
- **Security headers** — `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: no-referrer`, and a
  lock-down CSP apply to every response.
- **Docs toggle** — `DOCS_ENABLED=false` removes `/docs`, `/redoc`, and
  `/openapi.json` from a public deployment.

### Cost governance

With local open models there is no per-token bill: the price table is empty,
so spend is reported as `null` (never a fabricated zero) and budgets only
apply if a priced model server is configured. The ledger still records
tokens and latency for every request, which is what the Usage pages chart.

Every token-spending path is metered into a Postgres ledger
(`usage_events`: user, endpoint, model, tokens, estimated spend, latency,
status), and chat has a budget brake:

- **Pricing** — one table (`app/services/pricing.py`) serves both the
  runtime ledger and the eval scripts, so a dashboard number and an eval
  number can never disagree. Unknown models report `null` cost — never a
  fabricated zero. Requests with unpricable models still log usage.
- **Budgets** — `USAGE_DAILY_BUDGET_USD` / `USAGE_MONTHLY_BUDGET_USD` are
  global defaults; admins set per-user overrides from the Users page
  (`null` = follow global, `0` = disable the window, audit-recorded).
- **Enforcement** — on `/chat` and `/chat/stream` the budget gate runs
  **before any model spend**: over the limit is a structured
  `403 {"code": "usage_limit_exceeded", window, limit_usd, spent_usd}`
  that costs nothing, and ≥ 80% of a budget adds an `X-Usage-Warn`
  response header. Both the gate and the ledger are fail-open by design:
  a database outage degrades governance, never availability.
- **Visibility** — `GET /usage/me` (own spend/tokens per day + effective
  budgets) and `GET /admin/usage` (spend ranked by account and model,
  daily bars) power the Usage pages in the UI.

## Frontend

A React + Vite + TypeScript single-page UI (Tailwind CSS v4, sidebar
navigation, React Router with `/` public landing, `/login`, `/register`,
guarded `/chat` + `/documents`, and an admin-only `/admin` dashboard) that
exercises the API above — nothing is served by the backend itself; the Vite
dev server (:5173) proxies `/api`-style paths to :8000:

- **Chat** — ask questions and watch the answer **stream in live** (SSE
  `meta`/`delta`/`done`; the streamed text is always replaced by the
  server's validated `done.answer`), with expandable source excerpts,
  status chip (grounded vs insufficient evidence), timing, and 👍/👎
  feedback per answer (optimistic toggle, `null` to clear, votes reload
  with the thread); 429 shows a retry countdown, a failed stream rolls
  back the bubble and restores your question. The sidebar lists saved
  chats with a **New chat** button, resume on click, and inline delete;
  threads reload from `GET /conversations/{id}` and the first message of
  a new thread creates the conversation server-side
- **Documents** — drag-and-drop upload (returns instantly; indexing runs in
  the background and the list auto-refreshes every 2 s while anything is
  pending/processing), duplicate uploads are detected and reported, list
  with status/chunks/size, full-document viewer (rendered Markdown / PDF
  iframe / text), delete
- **Admin dashboard** (`/admin`, admin session required) — Overview KPIs,
  Users (roles, activation, per-user budgets, password resets), Usage
  (spend charts + by-account/by-model rankings), Feedback inbox, Audit log
- **Health badge** — polls `/health/ready` every 30 s with dependency details
  on hover
- **Global UX** — toasts for app-wide events (session expiry → "Session
  expired" + login redirect; budget refusals stay visible as sticky
  warnings), loading skeletons for lists/threads, an onboarding checklist
  in Chat until the first document exists, favicon + meta/theme-color
  branding, and a dedicated 404 page. Theme (dark by default, light
  toggle) persists in `localStorage`.

The `/retrieve` endpoint stays API-only (no UI screen): it remains available
via `/docs` and curl as a retrieval debugging tool.

```powershell
cd frontend
npm install          # once
npm run dev          # dev server + API proxy → :5173 (UI only; backend on :8000)
npm run typecheck    # tsc --noEmit
npm test             # vitest (API client, SSE parser, streaming, auth guards, toasts)
npm run build        # typecheck + production build to frontend/dist
```

## Evaluation

A labelled dataset (`evals/dataset.jsonl`, 41 questions across four types —
factual, multi-document, distractor, and no-answer) drives two suites. One
command runs both over a shared corpus ingest and writes a single results
file with provenance (app version, git commit, dataset hash, run config):

```powershell
uv run python scripts/evaluate.py                    # retrieval + chat suites
uv run python scripts/evaluate.py --retrieval-only   # recall / MRR only
uv run python scripts/evaluate.py --chat-only        # answer quality only
uv run python scripts/evaluate.py --skip-ingest      # reuse the eval index
```

The suite ingests the sample documents into a dedicated `documents_eval`
ChromaDB collection and a dedicated `data/bm25_eval.db` term index (the live
indexes and PostgreSQL are never touched). Results are written to
`evals/results/latest.json` (`--save` to write elsewhere). A full run costs a
few cents; retrieval-only runs cost a fraction of a cent.

### Metric glossary

| Metric | Meaning |
|---|---|
| `retrieval.answerable.recall@3` / `recall@5` | Fraction of answerable questions with a relevant document in the top-3 / top-5 results. |
| `retrieval.answerable.mrr` | Mean reciprocal rank of the first relevant hit. |
| `retrieval.no_answer.max_top1_score` | Highest retrieval score given to an unanswerable question — a hallucination-risk guard. Scores are rank support, not confidence. |
| `chat.answerable.fact_score_mean` / `fact_all_rate` | Fraction of `expected_facts` present in the answer / fraction of questions with all facts present (case-insensitive substring; `\|`-separated phrasing alternates). |
| `chat.no_answer.abstention_accuracy` | No-answer questions correctly refused instead of hallucinated (the safety number). |
| `chat.answerable.citation_coverage` | Fraction of declared relevant documents actually cited by the answer. |
| `chat.answerable.grounded_sentence_pct` | Structural groundedness proxy: answer sentences sharing content words with the retrieved evidence. |
| `chat.errors.rate` | Questions that failed with an error (lower is better). |
| `chat.latency_ms.*` / `retrieval.latency_ms.*` | Mean stage latencies in milliseconds. |
| `chat.tokens.*` / `chat.cost_usd.*` | Usage totals; cost is estimated from the shared price table in `app/services/pricing.py` — the same single source the runtime usage ledger prices from, so eval and dashboard dollars agree by construction. |
| `chat.answerable.abstained` | Answerable questions the model declined to answer (over-caution counterweight). |

Every results file carries `generated_at`, `app_version`, `git_commit`, and
`dataset_sha256`, so any number can be traced to the code and dataset that
produced it.

### Comparing runs (regression gate)

```powershell
uv run python scripts/compare_eval.py evals/results/baseline.json evals/results/latest.json
```

Prints a per-metric `old → new` table with deltas, applies per-metric
tolerances (ε on scores, ±25% relative on latency), and exits `1` if a gated
metric regressed or is missing — `0` otherwise, so it drops into CI. Legacy
flat result files (`baseline.json`, `hybrid.json`) stay comparable: dotted
metric paths fall back to top-level keys automatically. The gate logic is
covered by `tests/test_eval_regression.py` (no API key needed).

## Test

```powershell
uv run pytest
```

## Project structure

```text
oraculum/
├── app/
│   ├── main.py                  # FastAPI application factory
│   ├── config.py                # Settings loaded from environment / .env
│   ├── observability.py         # X-Request-ID middleware + request-scoped log ID
│   ├── api/
│   │   ├── health.py            # GET /health + GET /health/ready
│   │   ├── auth.py              # Sessions, API key, guest actor, scoping
│   │   ├── documents.py         # Document upload/list/chunks/delete endpoints
│   │   ├── retrieve.py          # POST /retrieve semantic search endpoint
│   │   ├── chat.py              # POST /chat grounded answer endpoint (optional persistence)
│   │   ├── conversations.py      # Chat history list/get/rename/delete endpoints
│   │   └── deps.py               # Shared DB session dependencies (503 on DB down)
│   ├── db/
│   │   ├── base.py               # SQLAlchemy DeclarativeBase
│   │   ├── session.py            # Engine + session factory + URL normalization
│   │   ├── models.py             # SQLAlchemy Document, Conversation, ChatMessage models
│   │   └── repositories.py       # Document record CRUD + conversation/message persistence
│   ├── schemas/
│   │   ├── documents.py          # Pydantic models for pages, chunks, records
│   │   ├── retrieval.py          # Pydantic models for retrieval request/results
│   │   ├── chat.py               # Pydantic models for chat request/answer/sources
│   │   └── conversations.py       # Pydantic models for saved chats and messages
│   └── services/
│       ├── text_extraction.py    # PDF/TXT/Markdown → pages (magic-byte validation)
│       ├── tokenizer.py          # Token counting with the embedding model's own tokenizer
│       ├── chunking.py           # Pages → token-bounded overlapping chunks
│       ├── embeddings.py         # Local sentence-transformers embeddings (bge-small, PyTorch)
│       ├── reranker.py           # Cross-encoder reranking of fused candidates
│       ├── llm_client.py         # llama-server client: JSON-schema grammar, logprobs, SSE
│       ├── verifier.py           # NLI check: do the cited passages entail the answer?
│       ├── calibration.py        # Logistic calibrator + ECE/Brier/AUROC/risk-coverage (NumPy)
│       ├── confidence.py         # Signals → calibrated P(correct) → answer or withhold
│       ├── models.py             # Builds every model-backed service from Settings
│       ├── ingestion.py          # Background pipeline: extract→chunk→embed→index + startup recovery
│       ├── vector_store.py       # Persistent ChromaDB collection (upsert/query/delete)
│       ├── bm25_index.py          # SQLite FTS5 term index (BM25) + failure policy
│       ├── retrieval.py           # Hybrid retrieve_documents + RRF + recall/MRR metrics
│       ├── generation.py         # Grounded generation, citation resolution, token-level signals
│       ├── rate_limit.py         # In-memory per-IP rate limiting
│       ├── storage.py            # Safe UUID-based file storage under data/uploads
│       └── errors.py             # Ingestion, storage, and indexing error types
├── alembic/
│   ├── env.py                    # Alembic env wired to app settings
│   └── versions/                 # Generated migration files
├── alembic.ini                   # Alembic config (no secrets)
├── samples/                      # Fictional sample documents for manual testing
├── research/                     # The calibrated-abstention study
│   ├── __main__.py               # CLI: build-squad, build-acme, run, study
│   ├── datasets.py               # SQuAD 2.0 + in-domain benchmarks in one format
│   ├── scoring.py                # SQuAD EM/F1/containment + expected-facts scoring
│   ├── runner.py                 # Pipeline over a benchmark → one JSON row per question
│   └── study.py                  # Grouped CV, bootstrap CIs, gate evaluation, report
├── evals/
│   ├── dataset.jsonl             # Labelled eval questions (41, 4 types, expected_facts)
│   ├── benchmarks/               # Built benchmarks (corpus + questions + manifest)
│   ├── runs/                     # Per-question pipeline logs (study input)
│   ├── study/                    # REPORT.md, report.json, figures
│   └── results/                  # Retrieval/answer regression results
├── models/calibrator.json        # The fitted confidence gate the server loads
├── deploy/                       # systemd units, Caddy site, env template, guide
├── scripts/
│   ├── load_demo_corpus.py       # Index files into the shared (guest-visible) corpus
│   ├── verify_index.py           # Hybrid retrieval smoke test against the index
│   ├── backfill_bm25.py          # One-time: rebuild the FTS term index from stored texts
│   ├── evaluate.py               # Single entry point: retrieval + chat suites, one results file
│   ├── evaluate_retrieval.py     # Recall@k / MRR / latency suite (used by evaluate.py)
│   ├── evaluate_chat.py          # Answer quality / abstention / cost suite (used by evaluate.py)
│   └── compare_eval.py           # Regression gate: compare two result files, exit 1 on drop
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_text_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_bm25_index.py
│   ├── test_retrieval.py
│   ├── test_hybrid.py
│   ├── test_generation.py
│   ├── test_chat.py
│   ├── test_reliability.py
│   ├── test_documents_api.py
│   ├── test_ingestion.py
│   ├── test_conversations.py
│   ├── test_eval_dataset.py
│   ├── test_eval_regression.py
│   ├── test_open_models.py       # llama-server client, tokenizer, reranker, verifier
│   ├── test_confidence.py        # Calibrator maths, metrics, the confidence gate
│   ├── test_research.py          # Benchmarks, scoring, runner rows, end-to-end study
│   └── test_guest_mode.py        # Public demo access rules and rate limits
├── frontend/
│   ├── package.json
│   ├── vite.config.ts            # Vite config + dev proxy + vitest config
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx               # Tab shell + chat/document sidebar + API-key modal
│       ├── index.css               # Tailwind v4 entry: @theme tokens, dark palette
│       ├── api/                  # Typed fetch client + TS types + tests
│       ├── components/           # HealthBadge, SourceCard, ConfidenceNote, upload dropzone
│       ├── hooks/                # useDocuments: shared list + status polling
│       └── tabs/                 # Chat, Documents screens
├── .env.example
├── pyproject.toml
└── README.md
```