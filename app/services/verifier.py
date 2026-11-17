"""Answer-support check with a natural language inference (NLI) cross-encoder.

Question: do the passages the model cited actually entail its answer? An
answer can carry a citation and still not be supported by it, which a
citation-format check cannot see. An NLI model reads (premise = cited
passage, hypothesis = answer) and returns P(entailment / neutral /
contradiction).

Two hypotheses are scored, because short extractive answers ("20 days")
are not sentences an NLI model can judge on their own:
- the answer alone;
- the question joined with the answer ("How many days of leave? 20 days").
Each takes the maximum over cited passages (support needs one passage).

Default model: cross-encoder/nli-deberta-v3-xsmall (22M parameters, PyTorch
on CPU). The label order is read from the model config, not assumed.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.services.reranker import PairScorer, load_cross_encoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupportScores:
    entail: float
    entail_qa: float
    contradiction: float


def _softmax(row: Sequence[float]) -> list[float]:
    peak = max(row)
    exps = [math.exp(value - peak) for value in row]
    total = sum(exps)
    return [value / total for value in exps]


def _label_index(scorer: Any, name: str, default: int) -> int:
    config = getattr(getattr(scorer, "model", None), "config", None)
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if str(label).lower().startswith(name):
            return int(index)
    return default


class SupportVerifier:
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        scorer: PairScorer | None = None,
        max_passages: int = 4,
    ) -> None:
        self.model = model
        self._scorer = scorer if scorer is not None else load_cross_encoder(model, device)
        self._max_passages = max_passages
        # cross-encoder/nli-* models: contradiction=0, entailment=1, neutral=2.
        self._entail = _label_index(self._scorer, "entail", 1)
        self._contra = _label_index(self._scorer, "contradict", 0)

    def score(
        self, question: str, answer: str, passages: Sequence[str]
    ) -> SupportScores | None:
        passages = [p for p in passages if p.strip()][: self._max_passages]
        if not passages or not answer.strip():
            return None
        qa = f"{question.strip()} {answer.strip()}"
        pairs = [(p, answer) for p in passages] + [(p, qa) for p in passages]
        with self._lock:
            logits = self._scorer.predict(pairs, show_progress_bar=False)
        probs = [_softmax([float(v) for v in row]) for row in logits]
        n = len(passages)
        return SupportScores(
            entail=max(row[self._entail] for row in probs[:n]),
            entail_qa=max(row[self._entail] for row in probs[n:]),
            contradiction=max(row[self._contra] for row in probs[:n]),
        )
