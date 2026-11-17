"""Local text embeddings with a Hugging Face sentence-transformers model.

Default model: BAAI/bge-small-en-v1.5 (33M parameters, 384 dimensions,
512-token input), run with PyTorch on CPU. Vectors are L2-normalized, so the
vector store's cosine distance and a plain dot product agree.

Queries and documents can be embedded differently: bge-style retrievers
were trained with an instruction prefix on the query side only
(EMBEDDING_QUERY_PREFIX; empty by default because bge v1.5 retrieves well
without it). Documents never get a prefix.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from app.services.errors import EmbeddingNotConfiguredError, EmbeddingServiceError

logger = logging.getLogger(__name__)


class SentenceEncoder(Protocol):
    """The slice of sentence_transformers.SentenceTransformer that we use."""

    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


@lru_cache(maxsize=2)
def load_sentence_encoder(model_name: str, device: str) -> SentenceEncoder:
    """Load (and cache) the encoder once per process; downloads on first use."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise EmbeddingNotConfiguredError(
            "sentence-transformers is not installed; run `uv sync`."
        ) from exc
    try:
        logger.info("Loading embedding model %s on %s", model_name, device)
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:  # noqa: BLE001 - surface any load failure as 503
        raise EmbeddingNotConfiguredError(
            f"Could not load embedding model {model_name!r}: {type(exc).__name__}."
        ) from exc


class Embedder:
    """Turns text into normalized vectors with a local encoder.

    The encoder is shared process-wide and guarded by a lock: PyTorch CPU
    inference is already multi-threaded internally, so serializing calls
    avoids oversubscribing cores when several requests arrive together.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str,
        batch_size: int = 32,
        device: str = "cpu",
        query_prefix: str = "",
        encoder: SentenceEncoder | None = None,
    ) -> None:
        self._model = model
        self._batch_size = max(1, batch_size)
        self._query_prefix = query_prefix
        self._encoder = encoder if encoder is not None else load_sentence_encoder(model, device)
        self._tokens_in = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def tokens_in(self) -> int:
        """Kept for the usage ledger's interface; local inference is not metered."""
        return self._tokens_in

    @property
    def retry_count(self) -> int:
        """Local inference never retries; kept for the logging interface."""
        return 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (no prefix), preserving input order."""
        return self._encode(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed search queries, with the model's query instruction if any."""
        return self._encode([f"{self._query_prefix}{text}" for text in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with self._lock:
                matrix = self._encoder.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:  # noqa: BLE001 - one error type for callers
            raise EmbeddingServiceError(
                f"Embedding failed: {type(exc).__name__}."
            ) from exc
        vectors = [list(map(float, row)) for row in matrix]
        if len(vectors) != len(texts):
            raise EmbeddingServiceError(
                f"Encoder returned {len(vectors)} vectors for {len(texts)} inputs."
            )
        if any(not vector for vector in vectors):
            raise EmbeddingServiceError("Encoder returned an empty vector.")
        return vectors


def embed_queries(embedder: Any, texts: Sequence[str]) -> list[list[float]]:
    """Query-side embedding for any embedder (fakes may only define embed)."""
    method = getattr(embedder, "embed_queries", None)
    return method(texts) if callable(method) else embedder.embed(texts)
