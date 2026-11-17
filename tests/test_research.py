"""Benchmark building, answer scoring, the runner's row format, and the study."""

import json
import math
import random

import pytest

from app.schemas.chat import Source
from app.services.generation import GenerationResult
from research.datasets import build_acme, build_squad, load_questions, slugify
from research.runner import gold_retrieved, run_one
from research.scoring import facts_scores, normalize, squad_scores, token_f1
from research.study import load_rows, render_markdown, run_study, write_outputs

from conftest import RecordingBm25, RecordingEmbedder, RecordingVectorStore
from test_confidence import StubVerifier


class TestScoring:
    def test_normalize_drops_case_punctuation_articles(self) -> None:
        assert normalize("The  Normans, in France!") == "normans in france"

    def test_token_f1(self) -> None:
        assert token_f1("in France", "France") == pytest.approx(2 / 3)
        assert token_f1("Spain", "France") == 0.0

    def test_long_generated_answer_containing_gold_is_correct(self) -> None:
        s = squad_scores("The Normans came from Normandy in France.", ["France"])
        assert s["contains"] and s["correct"] and not s["em"]

    def test_contains_rule_needs_whole_words(self) -> None:
        assert not squad_scores("Frances Smith", ["France"])["contains"]

    def test_very_long_answer_cannot_pass_by_containment(self) -> None:
        answer = "It could be " + ", ".join(f"place{i}" for i in range(40)) + " or France"
        assert not squad_scores(answer, ["France"])["correct"]

    def test_facts_need_every_fact_with_alternatives(self) -> None:
        assert facts_scores("20 days, renewed every 6 months", ["20 days", "6 months|half a year"])["correct"]
        assert not facts_scores("20 days", ["20 days", "6 months"])["correct"]


def _squad_source(tmp_path):
    def qa(qid, text, impossible):
        return {"id": qid, "question": f"Q {qid}?", "is_impossible": impossible,
                "answers": [] if impossible else [{"text": text, "answer_start": 0}]}

    data = {"data": [
        {"title": f"Article_{a}", "paragraphs": [
            {"context": f"Paragraph {a}-{p} mentions answer{a}{p}.",
             "qas": [qa(f"{a}{p}{i}", f"answer{a}{p}", i % 2 == 1) for i in range(6)]}
            for p in range(3)
        ]}
        for a in range(5)
    ]}
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(data))
    return path


class TestDatasets:
    def test_build_squad_balanced_grouped_and_reproducible(self, tmp_path) -> None:
        source = _squad_source(tmp_path)
        manifest = build_squad(source, tmp_path / "squad", articles=3, questions_per_article=6, seed=1)
        questions = load_questions(tmp_path / "squad")
        assert manifest["articles"] == 3 and len(questions) == 18
        assert manifest["answerable"] == manifest["unanswerable"] == 9
        docs = sorted(p.stem for p in (tmp_path / "squad" / "corpus").glob("*.md"))
        assert {q["group"] for q in questions} == set(docs)
        assert all(q["answers"] for q in questions if q["answerable"])
        assert all(not q["answers"] for q in questions if not q["answerable"])
        again = build_squad(source, tmp_path / "squad2", articles=3, questions_per_article=6, seed=1)
        assert again["source_sha256"] == manifest["source_sha256"]
        assert [q["id"] for q in load_questions(tmp_path / "squad2")] == [q["id"] for q in questions]

    def test_corpus_document_holds_title_and_paragraphs(self, tmp_path) -> None:
        build_squad(_squad_source(tmp_path), tmp_path / "s", articles=1, questions_per_article=2)
        text = next((tmp_path / "s" / "corpus").glob("*.md")).read_text()
        assert text.startswith("# Article ") and text.count("Paragraph") == 3

    def test_build_acme_from_repository_files(self, tmp_path) -> None:
        from app.config import PROJECT_ROOT

        manifest = build_acme(PROJECT_ROOT / "evals" / "dataset.jsonl", PROJECT_ROOT / "samples", tmp_path / "acme")
        questions = load_questions(tmp_path / "acme")
        assert manifest["questions"] == len(questions) == 40  # one unscorable skipped
        assert manifest["unanswerable"] == 9
        assert all(q["scheme"] == "facts" for q in questions)
        assert len(manifest["skipped_unscorable"]) == 1

    def test_slugify(self) -> None:
        assert slugify("Normans (people)") == "Normans_people"


DOC = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class TestRunner:
    def _deps(self, result: GenerationResult):
        store = RecordingVectorStore()
        store.query_result = [{"id": f"{DOC}#0", "score": 0.9, "text": "The Normans lived in France.",
                               "metadata": {"document_id": DOC, "filename": "n.md", "page": 1, "chunk_index": 0}}]

        class Gen:
            def generate(self, question, evidence, history=None):
                return result

        from app.config import Settings

        return dict(settings=Settings(retrieval_hybrid=False), embedder=RecordingEmbedder(), store=store,
                    bm25=RecordingBm25(), reranker=None, generator=Gen(), verifier=StubVerifier(0.8))

    def test_row_for_correct_answer(self) -> None:
        result = GenerationResult(
            answer="France", status="answered", model="m", elapsed_ms=4.0, draft_answer="France",
            sources=[Source(source_id=f"{DOC}#0", document_id=DOC, filename="n.md", page=1, chunk_index=0, excerpt="x")],
            signals={"status_p_answered": 0.9, "n_cited": 1.0, "n_evidence": 1.0},
        )
        item = {"id": "q1", "group": "n", "answerable": True, "question": "Where?", "answers": ["France"], "scheme": "squad"}
        row = run_one(item, **self._deps(result))
        assert row["model_status"] == "answered" and row["draft_correct"] is True
        assert row["gold_retrieved"] is True
        assert row["features"]["nli_entail"] == 0.8
        assert row["features"]["status_p_answered"] == 0.9

    def test_unanswerable_answer_is_never_correct(self) -> None:
        result = GenerationResult(answer="France", status="answered", model="m", elapsed_ms=1.0,
                                  draft_answer="France", sources=[], signals={})
        item = {"id": "q2", "group": "n", "answerable": False, "question": "When?", "answers": [], "scheme": "squad"}
        row = run_one(item, **self._deps(result))
        assert row["draft_correct"] is False and row["gold_retrieved"] is None

    def test_gold_retrieved_handles_fact_alternatives(self) -> None:
        item = {"answerable": True, "answers": ["reset portal|reset.acme.example"], "scheme": "facts"}
        assert gold_retrieved(item, ["Use the reset portal today."]) is True
        assert gold_retrieved(item, ["Call the desk."]) is False


def _synthetic_runs(path, n_docs: int = 10, per_doc: int = 30, seed: int = 0, shift: float = 0.0):
    """Correctness is driven by nli_entail_qa and status_p_answered; rerank is noise."""
    rng = random.Random(seed)
    with path.open("w") as handle:
        for d in range(n_docs):
            for q in range(per_doc):
                answerable = q % 2 == 0
                entail = min(1.0, max(0.0, rng.gauss(0.75 if answerable else 0.35, 0.2) + shift))
                status_p = min(1.0, max(0.0, rng.gauss(0.8 if answerable else 0.55, 0.15)))
                logit = 6 * (entail - 0.55) + 3 * (status_p - 0.6)
                correct = answerable and rng.random() < 1 / (1 + math.exp(-logit))
                answered = rng.random() < 0.85
                handle.write(json.dumps({
                    "id": f"{d}-{q}", "group": f"doc{d}", "answerable": answerable, "question": "q",
                    "model_status": "answered" if answered else "insufficient_evidence",
                    "draft_correct": correct, "gold_retrieved": answerable or None, "model": "m",
                    "features": {"nli_entail_qa": entail, "nli_entail": entail, "status_p_answered": status_p,
                                 "rerank_top1": rng.gauss(0, 1), "rerank_margin": rng.gauss(0, 1),
                                 "cited_fraction": 0.4},
                }) + "\n")
        handle.write(json.dumps({"id": "err", "group": "doc0", "error": "timeout"}) + "\n")


class TestStudy:
    def test_load_rows_drops_errors(self, tmp_path) -> None:
        path = tmp_path / "runs.jsonl"
        _synthetic_runs(path, n_docs=2, per_doc=4)
        assert len(load_rows(path).rows) == 8

    def test_end_to_end_report_and_calibrator(self, tmp_path) -> None:
        runs = tmp_path / "squad.jsonl"
        _synthetic_runs(runs)
        transfer = tmp_path / "acme.jsonl"
        _synthetic_runs(transfer, n_docs=3, per_doc=12, seed=5, shift=-0.05)

        report, calibrator = run_study(runs, target_risk=0.15, k=5, n_boot=50, transfer_path=transfer)

        methods = report["methods"]
        assert methods["nli_entail_qa"]["auroc"] > 0.7
        assert abs(methods["rerank_top1"]["auroc"] - 0.5) < 0.15  # noise stays near chance
        assert methods["all"]["auroc"] >= methods["rerank_top1"]["auroc"]
        gate = methods["all"]["gate_at_target"]
        assert gate["coverage"] < report["no_gate"]["coverage"]
        assert gate["risk"] < report["no_gate"]["risk"]
        assert gate["answers_to_unanswerable"] < report["no_gate"]["answers_to_unanswerable"]
        assert "auroc" in report["bootstrap"]["all"]
        assert report["bootstrap"]["all"]["auroc"][0] <= methods["all"]["auroc"] <= report["bootstrap"]["all"]["auroc"][1]
        assert 0.0 < calibrator.threshold < 1.0
        assert calibrator.meta["n"] == report["dataset"]["model_answered"]
        assert report["transfer"]["model_answered"] > 0

        out = tmp_path / "study"
        write_outputs(report, calibrator, out, tmp_path / "models" / "calibrator.json")
        md = (out / "REPORT.md").read_text()
        assert "Which signals separate" in md and "Transfer to the in-domain set" in md
        assert json.loads((out / "report.json").read_text())["final_calibrator"]["threshold"] == calibrator.threshold
        from app.services.calibration import Calibrator

        assert Calibrator.load(tmp_path / "models" / "calibrator.json").features == calibrator.features

    def test_too_few_answers_is_a_clear_error(self, tmp_path) -> None:
        runs = tmp_path / "tiny.jsonl"
        _synthetic_runs(runs, n_docs=1, per_doc=5)
        with pytest.raises(SystemExit, match="run more of the benchmark"):
            run_study(runs, n_boot=5)

    def test_markdown_tolerates_missing_values(self, tmp_path) -> None:
        runs = tmp_path / "r.jsonl"
        _synthetic_runs(runs, n_docs=6, per_doc=10)
        report, _ = run_study(runs, n_boot=5)
        report["dataset"]["gold_in_retrieved_passages"] = None
        assert "–" in render_markdown(report)
