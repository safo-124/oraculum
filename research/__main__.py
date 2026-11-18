"""Calibrated-abstention study: build datasets, run the pipeline, fit the gate.

    uv run python -m research build-squad           # download + sample SQuAD 2.0 dev
    uv run python -m research build-acme            # in-domain set from evals/ + samples/
    uv run python -m research run squad             # pipeline over the benchmark (resumable)
    uv run python -m research run acme
    uv run python -m research study --transfer acme # metrics, report, models/calibrator.json

`run` needs the model server (llama-server) up; see deploy/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "evals" / "benchmarks"
RUNS = ROOT / "evals" / "runs"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    squad = sub.add_parser("build-squad", help="build the SQuAD 2.0 benchmark")
    squad.add_argument("--articles", type=int, default=12)
    squad.add_argument("--per-article", type=int, default=40)
    squad.add_argument("--seed", type=int, default=13)
    squad.add_argument("--source", type=Path, default=ROOT / "data" / "downloads" / "dev-v2.0.json")

    sub.add_parser("build-acme", help="build the in-domain benchmark")

    run = sub.add_parser("run", help="run a benchmark through the pipeline")
    run.add_argument("dataset", help="benchmark name under evals/benchmarks (squad, acme)")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--reindex", action="store_true", help="rebuild the benchmark index")

    study = sub.add_parser("study", help="fit and evaluate the calibrator")
    study.add_argument("--train", default="squad")
    study.add_argument("--transfer", default=None, help="benchmark to test transfer on")
    study.add_argument("--target-risk", type=float, default=0.1)
    study.add_argument("--folds", type=int, default=5)
    study.add_argument("--bootstrap", type=int, default=1000)
    study.add_argument("--calibrator", type=Path, default=ROOT / "models" / "calibrator.json")

    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))

    if args.command == "build-squad":
        from research.datasets import SQUAD_DEV_URL, build_squad, download

        source = download(SQUAD_DEV_URL, args.source)
        manifest = build_squad(source, BENCH / "squad", articles=args.articles,
                               questions_per_article=args.per_article, seed=args.seed)
        print(manifest)
    elif args.command == "build-acme":
        from research.datasets import build_acme

        print(build_acme(ROOT / "evals" / "dataset.jsonl", ROOT / "samples", BENCH / "acme"))
    elif args.command == "run":
        from app.config import get_settings
        from research.runner import run_benchmark

        run_benchmark(BENCH / args.dataset, RUNS / f"{args.dataset}.jsonl", get_settings(),
                      limit=args.limit, reindex=args.reindex)
    elif args.command == "study":
        from research.datasets import load_questions
        from research.study import run_study, write_outputs

        def questions_of(name):
            path = BENCH / name
            return {q["id"]: q for q in load_questions(path)} if (path / "questions.jsonl").exists() else None

        transfer = RUNS / f"{args.transfer}.jsonl" if args.transfer else None
        report, calibrator = run_study(
            RUNS / f"{args.train}.jsonl", target_risk=args.target_risk, k=args.folds,
            n_boot=args.bootstrap, transfer_path=transfer,
            questions=questions_of(args.train),
            transfer_questions=questions_of(args.transfer) if args.transfer else None,
        )
        out_dir = ROOT / "evals" / "study"
        write_outputs(report, calibrator, out_dir, args.calibrator)
        print(f"report: {out_dir / 'REPORT.md'}\ncalibrator: {args.calibrator} "
              f"(threshold {calibrator.threshold:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
