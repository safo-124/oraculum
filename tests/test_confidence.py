"""Calibrated abstention: calibrator fitting, metrics, and the confidence gate."""

import math
import random

import numpy as np
import pytest

from app.schemas.chat import Source
from app.schemas.retrieval import RetrievalResult
from app.services import calibration as cal
from app.services.confidence import (
    LOW_CONFIDENCE_ANSWER,
    ConfidenceGate,
    GatedGenerator,
    collect_features,
    load_calibrator,
)
from app.services.generation import GenerationResult
from app.services.verifier import SupportScores


# ------------------------------------------------------------ calibrator


def _synthetic(n: int = 600, seed: int = 0):
    """y ~ Bernoulli(sigmoid(2*a - 1*b)); feature c is pure noise."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        a, b, c = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
        p = 1 / (1 + math.exp(-(2 * a - b)))
        rows.append({"a": a, "b": b, "c": c})
        labels.append(1 if rng.random() < p else 0)
    return rows, labels


class TestFitLogistic:
    def test_recovers_signs_and_ignores_noise(self) -> None:
        rows, labels = _synthetic()
        model = cal.fit_logistic(rows, labels, ["a", "b", "c"], l2=0.1)
        wa, wb, wc = model.weights
        assert wa > 1.0 and wb < -0.4
        assert abs(wc) < 0.3

    def test_predictions_are_calibrated_on_training_distribution(self) -> None:
        rows, labels = _synthetic(n=3000, seed=1)
        model = cal.fit_logistic(rows, labels, ["a", "b", "c"])
        probs = model.predict_matrix(cal.impute_matrix(rows, model.features)[0])
        assert cal.expected_calibration_error(probs, labels) < 0.05

    def test_missing_values_are_imputed_with_training_mean(self) -> None:
        rows, labels = _synthetic(n=200)
        rows[0] = {"a": None, "b": float("nan")}
        model = cal.fit_logistic(rows, labels, ["a", "b", "c"])
        at_mean = model.predict({"a": model.mean[0], "b": model.mean[1], "c": model.mean[2]})
        assert model.predict({}) == pytest.approx(at_mean)

    def test_json_round_trip(self, tmp_path) -> None:
        rows, labels = _synthetic(n=100)
        model = cal.fit_logistic(rows, labels, ["a", "b"])
        model.threshold = 0.42
        model.meta = {"dataset": "unit"}
        path = tmp_path / "c.json"
        model.save(path)
        loaded = cal.Calibrator.load(path)
        assert loaded.threshold == 0.42 and loaded.meta == {"dataset": "unit"}
        assert loaded.predict({"a": 1.0, "b": 0.0}) == pytest.approx(
            model.predict({"a": 1.0, "b": 0.0})
        )

    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        rows = [{"a": 1.0, "k": 3.0}, {"a": -1.0, "k": 3.0}] * 20
        model = cal.fit_logistic(rows, [1, 0] * 20, ["a", "k"])
        assert all(math.isfinite(w) for w in model.weights)


class TestMetrics:
    def test_perfect_calibration_has_zero_ece(self) -> None:
        conf = [0.0] * 10 + [1.0] * 10
        hit = [0] * 10 + [1] * 10
        assert cal.expected_calibration_error(conf, hit) == pytest.approx(0.0)

    def test_overconfidence_shows_in_ece_and_brier(self) -> None:
        conf = [0.9] * 10
        hit = [1] * 5 + [0] * 5
        assert cal.expected_calibration_error(conf, hit) == pytest.approx(0.4)
        assert cal.brier_score(conf, hit) == pytest.approx(0.41)

    def test_auroc_known_values(self) -> None:
        assert cal.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
        assert cal.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
        assert cal.auroc([0.5, 0.5], [1, 0]) == 0.5
        assert math.isnan(cal.auroc([0.3], [1]))

    def test_risk_coverage_counts_unanswered_outside_coverage(self) -> None:
        score = [0.9, 0.8, 0.7, 0.1]
        correct = [1, 0, 1, 0]
        answered = [True, True, True, False]
        curve = cal.risk_coverage(score, correct, answered)
        assert curve["coverage"] == [0.25, 0.5, 0.75]
        assert curve["risk"] == pytest.approx([0.0, 0.5, 1 / 3])
        assert curve["max_coverage"] == 0.75
        assert cal.selective_accuracy_at(curve, 0.5) == pytest.approx(0.5)

    def test_threshold_for_risk_picks_largest_safe_set(self) -> None:
        score = [0.95, 0.9, 0.8, 0.6, 0.4]
        correct = [1, 1, 1, 0, 0]
        assert cal.threshold_for_risk(score, correct, 0.0) == 0.8
        assert cal.threshold_for_risk(score, correct, 0.25) == 0.6
        assert cal.threshold_for_risk([0.9], [0], 0.1) == 1.0

    def test_reliability_table_sums_to_n(self) -> None:
        table = cal.reliability_table(np.linspace(0, 1, 25), [1] * 25, bins=5)
        assert sum(row["n"] for row in table) == 25


# ------------------------------------------------------------ the gate


def _evidence() -> list[RetrievalResult]:
    return [
        RetrievalResult(source_id="d#0", document_id="d", filename="f.md", page=1,
                        chunk_index=0, text="Staff get 20 days.", score=0.1, rerank_score=6.0),
        RetrievalResult(source_id="d#1", document_id="d", filename="f.md", page=1,
                        chunk_index=1, text="VPN guide.", score=0.1, rerank_score=1.5),
    ]


def _answered(**overrides) -> GenerationResult:
    base = dict(
        answer="20 days",
        status="answered",
        sources=[Source(source_id="d#0", document_id="d", filename="f.md", page=1,
                        chunk_index=0, excerpt="Staff get 20 days.")],
        model="m",
        elapsed_ms=1.0,
        signals={"status_p_answered": 0.97, "answer_mean_logprob": -0.2,
                 "answer_min_logprob": -0.9, "n_cited": 1.0, "n_evidence": 2.0},
        draft_answer="20 days",
    )
    base.update(overrides)
    return GenerationResult(**base)


class StubVerifier:
    def __init__(self, entail: float = 0.95) -> None:
        self.entail = entail
        self.calls: list = []

    def score(self, question, answer, passages):
        self.calls.append((question, answer, list(passages)))
        return SupportScores(entail=self.entail, entail_qa=self.entail, contradiction=0.01)


def _calibrator(threshold: float) -> cal.Calibrator:
    """Confidence driven by nli_entail only: p = sigmoid(10*(entail - 0.5))."""
    return cal.Calibrator(
        features=["nli_entail"], mean=[0.5], std=[1.0], weights=[10.0], bias=0.0,
        threshold=threshold,
    )


class TestCollectFeatures:
    def test_all_three_views_are_collected(self) -> None:
        verifier = StubVerifier()
        features = collect_features("How much leave?", _evidence(), _answered(), verifier)
        assert features["rerank_top1"] == 6.0
        assert features["rerank_margin"] == pytest.approx(4.5)
        assert features["status_p_answered"] == 0.97
        assert features["nli_entail"] == 0.95
        assert features["cited_fraction"] == 0.5
        # only the cited passage is checked
        assert verifier.calls[0][2] == ["Staff get 20 days."]

    def test_without_rerank_or_verifier_features_are_absent(self) -> None:
        evidence = [e.model_copy(update={"rerank_score": None}) for e in _evidence()]
        features = collect_features("q", evidence, _answered(), None)
        assert "rerank_top1" not in features and "nli_entail" not in features


class TestConfidenceGate:
    def test_confident_answer_passes_with_confidence(self) -> None:
        gate = ConfidenceGate(verifier=StubVerifier(0.95), calibrator=_calibrator(0.5))
        result = gate.assess("q", _evidence(), _answered())
        assert result.status == "answered"
        assert result.confidence == pytest.approx(1 / (1 + math.exp(-4.5)))
        assert result.signals["nli_entail"] == 0.95

    def test_low_confidence_answer_is_withheld(self) -> None:
        gate = ConfidenceGate(verifier=StubVerifier(0.2), calibrator=_calibrator(0.5))
        result = gate.assess("q", _evidence(), _answered())
        assert result.status == "insufficient_evidence"
        assert result.abstain_reason == "low_confidence"
        assert result.answer == LOW_CONFIDENCE_ANSWER
        assert result.sources == []
        assert result.draft_answer == "20 days"
        assert result.confidence < 0.5

    def test_threshold_override_beats_calibrator_file(self) -> None:
        gate = ConfidenceGate(verifier=StubVerifier(0.7), calibrator=_calibrator(0.99), threshold=0.5)
        assert gate.assess("q", _evidence(), _answered()).status == "answered"

    def test_without_calibrator_nothing_is_withheld(self) -> None:
        gate = ConfidenceGate(verifier=StubVerifier(0.0), calibrator=None)
        result = gate.assess("q", _evidence(), _answered())
        assert result.status == "answered" and result.confidence is None
        assert result.signals["nli_entail"] == 0.0  # features still reported

    def test_model_refusal_is_not_verified_at_serving_time(self) -> None:
        verifier = StubVerifier()
        gate = ConfidenceGate(verifier=verifier, calibrator=_calibrator(0.5))
        refusal = _answered(status="insufficient_evidence", sources=[], abstain_reason="model")
        result = gate.assess("q", _evidence(), refusal)
        assert result.abstain_reason == "model"
        assert verifier.calls == []

    def test_no_evidence_passes_through_untouched(self) -> None:
        gate = ConfidenceGate(verifier=StubVerifier(), calibrator=_calibrator(0.5))
        refusal = GenerationResult(answer="x", status="insufficient_evidence", sources=[],
                                   model="m", elapsed_ms=0, abstain_reason="no_evidence")
        assert gate.assess("q", [], refusal) is refusal


class TestGatedGenerator:
    class Inner:
        model = "inner-model"

        def __init__(self, result):
            self.result = result

        def generate(self, question, evidence, history=None):
            return self.result

        def generate_stream(self, question, evidence, history=None):
            yield ("delta", "20 ")
            yield ("delta", "days")
            yield ("result", self.result)

        def rewrite_query(self, question, history):
            return "rewritten"

    def test_stream_keeps_deltas_and_gates_the_final_result(self) -> None:
        gated = GatedGenerator(
            self.Inner(_answered()),
            ConfidenceGate(verifier=StubVerifier(0.1), calibrator=_calibrator(0.5)),
        )
        events = list(gated.generate_stream("q", _evidence()))
        assert [k for k, _ in events] == ["delta", "delta", "result"]
        assert events[-1][1].abstain_reason == "low_confidence"

    def test_delegates_other_attributes(self) -> None:
        gated = GatedGenerator(self.Inner(_answered()), ConfidenceGate(verifier=None, calibrator=None))
        assert gated.model == "inner-model"
        assert gated.rewrite_query("q", []) == "rewritten"


class TestCalibratorLoading:
    def test_missing_file_means_no_gating(self, tmp_path) -> None:
        load_calibrator.cache_clear()
        assert load_calibrator(str(tmp_path / "absent.json")) is None

    def test_corrupt_file_means_no_gating(self, tmp_path) -> None:
        load_calibrator.cache_clear()
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        assert load_calibrator(str(path)) is None


class TestChatResponseFields:
    def test_low_confidence_surfaces_in_chat_response(self, client, fake_vector_store, fake_generator) -> None:
        fake_vector_store.query_result = [
            {"id": "d#0", "score": 0.9, "text": "Staff get 20 days.",
             "metadata": {"document_id": "d", "filename": "f.md", "page": 1, "chunk_index": 0}}
        ]
        fake_generator.result = GenerationResult(
            answer=LOW_CONFIDENCE_ANSWER, status="insufficient_evidence", sources=[],
            model="m", elapsed_ms=3.0, confidence=0.31234, abstain_reason="low_confidence",
            draft_answer="21 days", signals={"nli_entail": 0.12345},
        )
        body = client.post("/chat", json={"question": "How much leave?"}).json()
        assert body["abstain_reason"] == "low_confidence"
        assert body["confidence"] == 0.3123
        assert body["withheld_answer"] == "21 days"
        assert body["signals"] == {"nli_entail": 0.1235}

    def test_no_evidence_reason_in_chat_response(self, client) -> None:
        body = client.post("/chat", json={"question": "anything"}).json()
        assert body["abstain_reason"] == "no_evidence"
        assert body["withheld_answer"] is None
