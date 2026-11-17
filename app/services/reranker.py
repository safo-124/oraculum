"""Cross-encoder reranking of retrieved candidates.

Hybrid retrieval (BM25 + vectors, fused with RRF) is good at recall but its
fused scores are rank votes, not relevance. A cross-encoder reads the
question and each candidate together, so it ranks precisely - and its raw
score for the best candidate is one of the confidence model's strongest
signals ("does any passage really answer this?").

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M parameters,
PyTorch on CPU, ~10 ms per candidate). Scores are logits; higher is more
relevant, and the scale is model-specific.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class PairScorer(Protocol):
    """The slice of sentence_transformers.CrossEncoder that we use."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


@lru_cache(maxsize=2)
def load_cross_encoder(model_name: str, device: str, max_length: int = 512) -> PairScorer:
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder %s on %s", model_name, device)
    return CrossEncoder(model_name, device=device, max_length=max_length)


class Reranker:
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        batch_size: int = 16,
        scorer: PairScorer | None = None,
    ) -> None:
        self.model = model
        self._batch_size = batch_size
        self._scorer = scorer if scorer is not None else load_cross_encoder(model, device)

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            raw = self._scorer.predict(
                [(question, passage) for passage in passages],
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        return [float(value) for value in raw]

    def rerank(
        self, question: str, results: Sequence[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Return the top_k results by cross-encoder score, scores attached."""
        scores = self.score(question, [result.text for result in results])
        ranked = sorted(
            zip(results, scores), key=lambda pair: (-pair[1], pair[0].source_id)
        )[:top_k]
        return [
            result.model_copy(update={"rerank_score": score}) for result, score in ranked
        ]
