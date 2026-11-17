"""The calibration study: which signals predict a correct answer, and how well?

Input: rows logged by research/runner.py. Output: metrics, a markdown report,
figures, and the calibrator file the server loads.

Protocol
- Unit of analysis: questions the model chose to answer (with at least one
  valid citation). Label = the answer is correct. An answer to an
  unanswerable question is always wrong.
- Grouped K-fold cross-validation by document (`group`): every probability
  in the evaluation comes from a calibrator that never saw that document.
- Methods: single signals, signal groups, and all signals together, each a
  logistic calibrator fitted inside the training folds.
- The gate threshold is also chosen inside the training folds (largest
  coverage whose in-sample risk <= target) and applied to the held-out fold,
  so the reported "realized risk" is an honest out-of-sample number.
- Uncertainty: 95% bootstrap intervals over documents (resampling groups,
  not questions, because questions about one document are correlated).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.services import calibration as cal
from app.services.confidence import FEATURE_GROUPS, FEATURES

METHODS: dict[str, list[str]] = {
    "status_prob": ["status_p_answered"],
    "answer_logprob": ["answer_mean_logprob"],
    "rerank_top1": ["rerank_top1"],
    "nli_entail_qa": ["nli_entail_qa"],
    "retrieval": FEATURE_GROUPS["retrieval"],
    "generation": FEATURE_GROUPS["generation"],
    "verification": FEATURE_GROUPS["verification"],
    "all": FEATURES,
}


@dataclass
class StudyData:
    rows: list[dict]  # every question with a result (errors dropped)
    answered: np.ndarray  # bool per row: the model answered with a citation
    correct: np.ndarray  # int per row: answered AND correct
    groups: list[str]


def load_rows(path: Path) -> StudyData:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row or "model_status" not in row:
            continue
        rows.append(row)
    answered = np.array([r["model_status"] == "answered" for r in rows], dtype=bool)
    correct = np.array(
        [int(r["model_status"] == "answered" and bool(r.get("draft_correct"))) for r in rows]
    )
    return StudyData(rows, answered, correct, [r["group"] for r in rows])


def group_folds(groups: Sequence[str], k: int, seed: int) -> list[set[str]]:
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    k = max(2, min(k, len(unique)))
    return [set(unique[i::k]) for i in range(k)]


def cross_validate(
    data: StudyData, features: list[str], *, k: int = 5, seed: int = 0,
    target_risk: float = 0.1, l2: float = 1.0,
) -> dict:
    """Out-of-fold P(correct) for answered rows + fold-wise threshold decisions."""
    idx = np.where(data.answered)[0]
    oof = np.full(len(data.rows), np.nan)
    admitted = np.zeros(len(data.rows), dtype=bool)
    for test_groups in group_folds([data.groups[i] for i in idx], k, seed):
        test = [i for i in idx if data.groups[i] in test_groups]
        train = [i for i in idx if data.groups[i] not in test_groups]
        if not test or len({int(data.correct[i]) for i in train}) < 2:
            continue
        model = cal.fit_logistic(
            [data.rows[i]["features"] for i in train], data.correct[train], features, l2=l2
        )
        train_p = model.predict_matrix(cal.impute_matrix([data.rows[i]["features"] for i in train], features)[0])
        tau = cal.threshold_for_risk(train_p, data.correct[train], target_risk)
        test_x = np.vstack([model.vector(data.rows[i]["features"]) for i in test])
        test_p = model.predict_matrix(test_x)
        oof[test] = test_p
        admitted[test] = test_p >= tau
    return {"oof": oof, "admitted": admitted}


def evaluate_scores(data: StudyData, scores: np.ndarray) -> dict:
    mask = data.answered & np.isfinite(scores)
    y = data.correct[mask]
    p = scores[mask]
    curve = cal.risk_coverage(
        np.where(np.isfinite(scores), scores, -np.inf), data.correct, mask
    )
    return {
        "n_answered": int(mask.sum()),
        "auroc": cal.auroc(p, y),
        "aurc": curve["aurc"],
        "ece": cal.expected_calibration_error(p, y),
        "brier": cal.brier_score(p, y),
        "sel_acc@50": cal.selective_accuracy_at(curve, 0.5),
        "sel_acc@70": cal.selective_accuracy_at(curve, 0.7),
    }


def gate_outcome(data: StudyData, admitted: np.ndarray) -> dict:
    """Coverage/risk over all questions when only admitted answers are shown."""
    shown = data.answered & admitted
    unanswerable = np.array([not r["answerable"] for r in data.rows])
    n = len(data.rows)
    return {
        "coverage": float(shown.sum() / n) if n else float("nan"),
        "risk": float(1 - data.correct[shown].mean()) if shown.any() else float("nan"),
        "answers_to_unanswerable": float(shown[unanswerable].mean()) if unanswerable.any() else float("nan"),
    }


def bootstrap(
    data: StudyData, scores_by_method: dict[str, np.ndarray], *, n_boot: int = 1000, seed: int = 0
) -> dict:
    """95% intervals for AUROC/AURC per method, and all-vs-best-single deltas."""
    groups = sorted(set(data.groups))
    by_group = {g: [i for i, gg in enumerate(data.groups) if gg == g] for g in groups}
    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {m: {"auroc": [], "aurc": []} for m in scores_by_method}
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        rows = [i for g in pick for i in by_group[groups[g]]]
        sub = StudyData(
            [data.rows[i] for i in rows], data.answered[rows], data.correct[rows],
            [data.groups[i] for i in rows],
        )
        for method, scores in scores_by_method.items():
            metrics = evaluate_scores(sub, scores[rows])
            samples[method]["auroc"].append(metrics["auroc"])
            samples[method]["aurc"].append(metrics["aurc"])

    def interval(values: list[float]) -> list[float]:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return [float("nan"), float("nan")]
        return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]

    result = {m: {k: interval(v) for k, v in s.items()} for m, s in samples.items()}
    singles = [m for m in scores_by_method if len(METHODS.get(m, [])) == 1]
    if "all" in samples and singles:
        point = {m: np.nanmean(samples[m]["aurc"]) for m in singles}
        best = min(point, key=point.get)
        delta = np.array(samples["all"]["aurc"]) - np.array(samples[best]["aurc"])
        result["all_minus_best_single_aurc"] = {"best_single": best, "ci": interval(list(delta))}
    return result


def run_study(
    runs_path: Path,
    *,
    target_risk: float = 0.1,
    k: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
    transfer_path: Path | None = None,
) -> tuple[dict, cal.Calibrator]:
    data = load_rows(runs_path)
    if data.answered.sum() < 20:
        raise SystemExit(
            f"only {int(data.answered.sum())} answered questions in {runs_path}; run more of the benchmark first"
        )
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": str(runs_path),
        "protocol": {"folds": k, "seed": seed, "target_risk": target_risk, "bootstrap": n_boot,
                     "split_unit": "document (group)"},
    }
    unanswerable = np.array([not r["answerable"] for r in data.rows])
    gold = [r["gold_retrieved"] for r in data.rows if r.get("gold_retrieved") is not None]
    report["dataset"] = {
        "questions": len(data.rows),
        "documents": len(set(data.groups)),
        "answerable": int((~unanswerable).sum()),
        "unanswerable": int(unanswerable.sum()),
        "model_answered": int(data.answered.sum()),
        "accuracy_when_answered": float(data.correct[data.answered].mean()),
        "gold_in_retrieved_passages": float(np.mean(gold)) if gold else None,
    }
    report["no_gate"] = gate_outcome(data, np.ones(len(data.rows), dtype=bool))

    scores_by_method: dict[str, np.ndarray] = {}
    report["methods"] = {}
    for method, features in METHODS.items():
        present = [f for f in features if any(f in r.get("features", {}) for r in data.rows)]
        if not present:
            continue
        cv = cross_validate(data, present, k=k, seed=seed, target_risk=target_risk)
        scores_by_method[method] = cv["oof"]
        report["methods"][method] = {
            "features": present,
            **evaluate_scores(data, cv["oof"]),
            "gate_at_target": gate_outcome(data, cv["admitted"]),
        }
    report["bootstrap"] = bootstrap(data, scores_by_method, n_boot=n_boot, seed=seed)

    # Final calibrator: all signals, fitted on every answered row; threshold
    # from the out-of-fold probabilities (the honest view of its behaviour).
    features = report["methods"]["all"]["features"]
    idx = np.where(data.answered)[0]
    final = cal.fit_logistic([data.rows[i]["features"] for i in idx], data.correct[idx], features)
    oof_all = scores_by_method["all"]
    valid = data.answered & np.isfinite(oof_all)
    final.threshold = cal.threshold_for_risk(oof_all[valid], data.correct[valid], target_risk)
    final.meta = {
        "trained_on": str(runs_path),
        "generated_at": report["generated_at"],
        "n": int(len(idx)),
        "target_risk": target_risk,
        "model": data.rows[0].get("model"),
        "weights_standardized": dict(zip(features, [round(w, 3) for w in final.weights])),
    }
    report["final_calibrator"] = {"threshold": final.threshold, **final.meta}
    report["reliability_all"] = cal.reliability_table(oof_all[valid], data.correct[valid])
    report["curves"] = {
        m: {key: v for key, v in cal.risk_coverage(
            np.where(np.isfinite(s), s, -np.inf), data.correct, data.answered & np.isfinite(s)
        ).items() if key in ("coverage", "risk")}
        for m, s in scores_by_method.items()
    }
    if transfer_path is not None:
        report["transfer"] = transfer(final, transfer_path)
    return report, final


def transfer(calibrator: cal.Calibrator, runs_path: Path) -> dict:
    """Apply a frozen calibrator and threshold to another dataset (domain shift)."""
    data = load_rows(runs_path)
    scores = np.full(len(data.rows), np.nan)
    for i, row in enumerate(data.rows):
        if data.answered[i]:
            scores[i] = calibrator.predict(row.get("features", {}))
    return {
        "runs": str(runs_path),
        "questions": len(data.rows),
        "model_answered": int(data.answered.sum()),
        **evaluate_scores(data, scores),
        "no_gate": gate_outcome(data, np.ones(len(data.rows), dtype=bool)),
        "gate_at_threshold": gate_outcome(data, np.nan_to_num(scores, nan=-1.0) >= calibrator.threshold),
    }


# ---------------------------------------------------------------- reporting


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{100 * value:.1f}%"


def render_markdown(report: dict) -> str:
    d, p = report["dataset"], report["protocol"]
    lines = [
        "# Calibrated abstention: study results",
        "",
        f"Generated {report['generated_at']} from `{report['runs']}`.",
        "",
        f"{d['questions']} questions over {d['documents']} documents "
        f"({d['answerable']} answerable, {d['unanswerable']} unanswerable). "
        f"The model answered {d['model_answered']} of them; "
        f"{_pct(d['accuracy_when_answered'])} of those answers were correct. "
        f"A gold answer appeared in the retrieved passages for "
        f"{_pct(d['gold_in_retrieved_passages'])} of answerable questions.",
        "",
        f"Protocol: {p['folds']}-fold cross-validation grouped by document, "
        f"threshold chosen inside training folds for a target risk of {_pct(p['target_risk'])}, "
        f"{p['bootstrap']} bootstrap resamples over documents.",
        "",
        "## Which signals separate right from wrong answers?",
        "",
        "Out-of-fold scores. AUROC: higher is better (0.5 = chance). AURC: area under the "
        "risk-coverage curve, lower is better. ECE and Brier: calibration, lower is better.",
        "",
        "| Method | Features | AUROC [95% CI] | AURC [95% CI] | ECE | Brier | Acc @50% cov. |",
        "|---|---|---|---|---|---|---|",
    ]
    boot = report["bootstrap"]
    for method, m in report["methods"].items():
        ci = boot.get(method, {})
        auroc_ci = ci.get("auroc", [float("nan")] * 2)
        aurc_ci = ci.get("aurc", [float("nan")] * 2)
        lines.append(
            f"| {method} | {len(m['features'])} | {_fmt(m['auroc'])} [{_fmt(auroc_ci[0])}, {_fmt(auroc_ci[1])}] "
            f"| {_fmt(m['aurc'])} [{_fmt(aurc_ci[0])}, {_fmt(aurc_ci[1])}] | {_fmt(m['ece'])} "
            f"| {_fmt(m['brier'])} | {_pct(m['sel_acc@50'])} |"
        )
    if "all_minus_best_single_aurc" in boot:
        delta = boot["all_minus_best_single_aurc"]
        lines += [
            "",
            f"All signals vs the best single signal ({delta['best_single']}): AURC difference "
            f"95% CI [{_fmt(delta['ci'][0])}, {_fmt(delta['ci'][1])}] (negative favours combining).",
        ]
    ng = report["no_gate"]
    lines += [
        "",
        "## What the gate does to the answers users see",
        "",
        "Realized out-of-sample numbers over all questions (coverage = share of questions "
        "answered; risk = share of shown answers that are wrong).",
        "",
        "| Policy | Coverage | Risk | Answers to unanswerable questions |",
        "|---|---|---|---|",
        f"| Model alone (no gate) | {_pct(ng['coverage'])} | {_pct(ng['risk'])} | {_pct(ng['answers_to_unanswerable'])} |",
    ]
    for method, m in report["methods"].items():
        g = m["gate_at_target"]
        lines.append(
            f"| Gate: {method} | {_pct(g['coverage'])} | {_pct(g['risk'])} | {_pct(g['answers_to_unanswerable'])} |"
        )
    fc = report["final_calibrator"]
    lines += [
        "",
        "## Shipped calibrator",
        "",
        f"All signals, fitted on {fc['n']} answered questions; threshold {_fmt(fc['threshold'])} "
        f"(target risk {_pct(fc['target_risk'])}). Standardized weights:",
        "",
        "| Feature | Weight |",
        "|---|---|",
    ]
    for name, weight in sorted(fc["weights_standardized"].items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"| {name} | {weight:+.3f} |")
    if "transfer" in report:
        t = report["transfer"]
        lines += [
            "",
            "## Transfer to the in-domain set (frozen calibrator and threshold)",
            "",
            f"`{t['runs']}`: {t['questions']} questions, model answered {t['model_answered']}. "
            f"AUROC {_fmt(t['auroc'])}, ECE {_fmt(t['ece'])}.",
            "",
            "| Policy | Coverage | Risk | Answers to unanswerable questions |",
            "|---|---|---|---|",
            f"| Model alone | {_pct(t['no_gate']['coverage'])} | {_pct(t['no_gate']['risk'])} | {_pct(t['no_gate']['answers_to_unanswerable'])} |",
            f"| With gate | {_pct(t['gate_at_threshold']['coverage'])} | {_pct(t['gate_at_threshold']['risk'])} | {_pct(t['gate_at_threshold']['answers_to_unanswerable'])} |",
        ]
    lines += [
        "",
        "## Figures",
        "",
        "![Reliability](reliability.png)",
        "",
        "![Risk-coverage](risk_coverage.png)",
        "",
    ]
    return "\n".join(lines)


def save_figures(report: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (uv sync --extra research); skipping figures")
        return []
    written = []
    table = report["reliability_all"]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#9aa0a6", linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot([r["mean_confidence"] for r in table], [r["accuracy"] for r in table],
            marker="o", color="#2a6fdb", linewidth=2, label="all signals (out-of-fold)")
    ax.set_xlabel("Predicted P(correct)")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out_dir / "reliability.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    palette = {"all": "#2a6fdb", "verification": "#1a9e77", "generation": "#d95f02", "retrieval": "#7570b3"}
    for method, curve in report["curves"].items():
        if method not in palette:
            continue
        ax.plot(curve["coverage"], curve["risk"], label=method, color=palette[method], linewidth=2)
    ng = report["no_gate"]
    ax.scatter([ng["coverage"]], [ng["risk"]], color="#202124", zorder=5, label="model alone")
    ax.axhline(report["protocol"]["target_risk"], color="#9aa0a6", linestyle=":", linewidth=1)
    ax.set_xlabel("Coverage (share of questions answered)")
    ax.set_ylabel("Risk (share of shown answers that are wrong)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = out_dir / "risk_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written


def write_outputs(report: dict, calibrator: cal.Calibrator, out_dir: Path, calibrator_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in report.items() if k != "curves"}
    (out_dir / "report.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    save_figures(report, out_dir)
    calibrator.save(calibrator_path)
