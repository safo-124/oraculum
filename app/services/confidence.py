"""Calibrated abstention: should Oraculum show this answer or withhold it?

A grounded-looking answer can still be wrong. Each answer gets a feature
vector from three independent views of the same question:

- retrieval (reranker): is there a passage that really matches the question?
    rerank_top1, rerank_margin
- generation (token log-probabilities): how sure was the model while
  deciding to answer and while writing the answer?
    status_p_answered, answer_mean_logprob, answer_min_logprob
- verification (NLI): do the cited passages entail the answer?
    nli_entail, nli_entail_qa, nli_contradiction
- plus cited_fraction (cited passages / passages shown).

A logistic calibrator fitted on a held-out benchmark (scripts/
fit_calibrator.py) maps the vector to P(answer is correct). Answers below
the calibrated threshold are withheld with abstain_reason="low_confidence";
the threshold is chosen on calibration data for a target risk (e.g. at most
10% wrong answers among those shown).

Without a calibrator file the gate still computes and returns the features
(useful for collecting training data) but never withholds an answer.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.schemas.retrieval import RetrievalResult
from app.services.calibration import Calibrator
from app.services.generation import GenerationResult
from app.services.verifier import SupportVerifier

logger = logging.getLogger(__name__)

FEATURES = [
    "rerank_top1",
    "rerank_margin",
    "status_p_answered",
    "answer_mean_logprob",
    "answer_min_logprob",
    "nli_entail",
    "nli_entail_qa",
    "nli_contradiction",
    "cited_fraction",
]

FEATURE_GROUPS = {
    "retrieval": ["rerank_top1", "rerank_margin"],
    "generation": ["status_p_answered", "answer_mean_logprob", "answer_min_logprob"],
    "verification": ["nli_entail", "nli_entail_qa", "nli_contradiction"],
    "citations": ["cited_fraction"],
}

LOW_CONFIDENCE_ANSWER = (
    "I found related passages, but I am not confident enough that they "
    "answer this question, so I am not giving an answer."
)


def retrieval_features(evidence: Sequence[RetrievalResult]) -> dict[str, float]:
    scores = sorted(
        (hit.rerank_score for hit in evidence if hit.rerank_score is not None),
        reverse=True,
    )
    features: dict[str, float] = {}
    if scores:
        features["rerank_top1"] = scores[0]
        features["rerank_margin"] = scores[0] - scores[1] if len(scores) > 1 else 0.0
    return features


def collect_features(
    question: str,
    evidence: Sequence[RetrievalResult],
    result: GenerationResult,
    verifier: SupportVerifier | None,
) -> dict[str, float]:
    """All features for one generated result (missing ones are simply absent)."""
    signals = dict(result.signals or {})
    features: dict[str, float] = retrieval_features(evidence)
    for name in ("status_p_answered", "answer_mean_logprob", "answer_min_logprob"):
        if name in signals:
            features[name] = float(signals[name])
    n_evidence = signals.get("n_evidence") or float(len(evidence)) or 1.0
    features["cited_fraction"] = float(signals.get("n_cited", 0.0)) / n_evidence

    draft = result.draft_answer or (result.answer if result.status == "answered" else "")
    if verifier is not None and draft:
        cited_ids = {source.source_id for source in result.sources}
        passages = [hit.text for hit in evidence if hit.source_id in cited_ids]
        if not passages:  # model declined or cited nothing: check the top passages
            passages = [hit.text for hit in evidence[:2]]
        try:
            support = verifier.score(question, draft, passages)
        except Exception:
            logger.warning("Support verification failed", exc_info=True)
            support = None
        if support is not None:
            features["nli_entail"] = support.entail
            features["nli_entail_qa"] = support.entail_qa
            features["nli_contradiction"] = support.contradiction
    return features


class ConfidenceGate:
    def __init__(
        self,
        *,
        verifier: SupportVerifier | None,
        calibrator: Calibrator | None,
        threshold: float | None = None,
    ) -> None:
        self.verifier = verifier
        self.calibrator = calibrator
        if threshold is not None:
            self.threshold = threshold
        elif calibrator is not None:
            self.threshold = calibrator.threshold
        else:
            self.threshold = None

    def assess(
        self,
        question: str,
        evidence: Sequence[RetrievalResult],
        result: GenerationResult,
    ) -> GenerationResult:
        if result.abstain_reason == "no_evidence" or not evidence:
            return result
        # Model-declined answers skip verification at serving time (nothing
        # will be shown); the benchmark scores them through collect_features.
        verifier = self.verifier if result.status == "answered" else None
        features = collect_features(question, evidence, result, verifier)
        signals = {**(result.signals or {}), **features}
        if result.status != "answered" or self.calibrator is None:
            return dataclasses.replace(result, signals=signals)

        confidence = self.calibrator.predict(features)
        if self.threshold is not None and confidence < self.threshold:
            logger.info(
                "Withholding answer: confidence %.3f < threshold %.3f",
                confidence,
                self.threshold,
            )
            return dataclasses.replace(
                result,
                answer=LOW_CONFIDENCE_ANSWER,
                status="insufficient_evidence",
                sources=[],
                signals=signals,
                confidence=confidence,
                abstain_reason="low_confidence",
            )
        return dataclasses.replace(result, signals=signals, confidence=confidence)


class GatedGenerator:
    """A Generator whose final results pass through the ConfidenceGate.

    Streaming is unchanged for the client: answer deltas are shown as they
    arrive, and the authoritative `done` event carries the gated result (a
    withheld answer replaces the streamed text, as the protocol requires).
    Every other attribute is delegated to the wrapped generator.
    """

    def __init__(self, inner: Any, gate: ConfidenceGate) -> None:
        self._inner = inner
        self.gate = gate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def generate(self, question, evidence, history=None) -> GenerationResult:
        result = self._inner.generate(question, evidence, history)
        return self.gate.assess(question, evidence, result)

    def generate_stream(self, question, evidence, history=None) -> Iterator:
        for kind, payload in self._inner.generate_stream(question, evidence, history):
            if kind == "result":
                yield kind, self.gate.assess(question, evidence, payload)
            else:
                yield kind, payload


@lru_cache(maxsize=4)
def load_calibrator(path: str) -> Calibrator | None:
    file = Path(path)
    if not file.is_file():
        logger.warning("No calibrator at %s; answers will not be gated", file)
        return None
    try:
        calibrator = Calibrator.load(file)
    except Exception:
        logger.exception("Calibrator at %s is unreadable; answers will not be gated", file)
        return None
    logger.info(
        "Loaded calibrator %s (threshold %.3f, features %s)",
        file,
        calibrator.threshold,
        ",".join(calibrator.features),
    )
    return calibrator
