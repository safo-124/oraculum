"""Benchmark datasets in one shared on-disk format.

A dataset is a directory:

    <dir>/corpus/*.md        documents to index (one file per source)
    <dir>/questions.jsonl    {"id", "question", "group", "answerable",
                              "answers": [...], "scheme": "squad"|"facts"}
    <dir>/manifest.json      provenance: source, sha256, seed, counts, license

`group` is the unit for grouped cross-validation (a Wikipedia article for
SQuAD): questions about the same document never land on both sides of a
calibration/test split, so the calibrator is always tested on unseen
documents.

Builders:
- build_squad: SQuAD 2.0 dev (Rajpurkar et al., 2018; CC BY-SA 4.0). Its
  unanswerable questions were written adversarially against the same
  paragraphs, so retrieval finds plausible evidence that does not contain
  the answer - exactly the case abstention must handle.
- build_acme: the repository's 41-question in-domain set (HR/IT policies),
  used to test whether a calibrator fitted on Wikipedia transfers.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
_SLUG = re.compile(r"[^A-Za-z0-9]+")


def slugify(title: str) -> str:
    return _SLUG.sub("_", title).strip("_")[:80] or "doc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed URL
            target.write_bytes(response.read())
    return target


def _write_dataset(out_dir: Path, documents: dict[str, str], questions: list[dict], manifest: dict) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    corpus = out_dir / "corpus"
    corpus.mkdir(parents=True)
    for name, text in documents.items():
        (corpus / f"{name}.md").write_text(text, encoding="utf-8")
    with (out_dir / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for item in questions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_squad(
    source: Path,
    out_dir: Path,
    *,
    articles: int = 12,
    questions_per_article: int = 40,
    seed: int = 13,
) -> dict:
    """Sample articles and balanced answerable/unanswerable questions."""
    data = json.loads(source.read_text(encoding="utf-8"))["data"]
    rng = random.Random(seed)
    chosen = rng.sample(data, k=min(articles, len(data)))
    documents: dict[str, str] = {}
    questions: list[dict] = []
    for article in sorted(chosen, key=lambda a: a["title"]):
        name = slugify(article["title"])
        paragraphs = [p["context"].strip() for p in article["paragraphs"]]
        title = article["title"].replace("_", " ")
        documents[name] = f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"
        pools: dict[bool, list[dict]] = defaultdict(list)
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answerable = not qa.get("is_impossible", False)
                golds = sorted({a["text"].strip() for a in qa.get("answers", []) if a["text"].strip()})
                if answerable and not golds:
                    continue
                pools[answerable].append(
                    {
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "group": name,
                        "answerable": answerable,
                        "answers": golds,
                        "scheme": "squad",
                    }
                )
        half = questions_per_article // 2
        for answerable in (True, False):
            pool = pools[answerable]
            rng.shuffle(pool)
            questions.extend(pool[:half])
    rng.shuffle(questions)
    manifest = {
        "name": "squad2-dev-subset",
        "source": SQUAD_DEV_URL,
        "source_sha256": sha256_file(source),
        "license": "CC BY-SA 4.0 (SQuAD 2.0, Rajpurkar et al. 2018)",
        "seed": seed,
        "articles": len(documents),
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "unanswerable": sum(not q["answerable"] for q in questions),
    }
    _write_dataset(out_dir, documents, questions, manifest)
    return manifest


def build_acme(dataset_path: Path, samples_dir: Path, out_dir: Path) -> dict:
    """Convert evals/dataset.jsonl + samples/ into the shared format."""
    documents: dict[str, str] = {}
    for path in sorted(samples_dir.glob("*.md")):
        documents[path.stem] = path.read_text(encoding="utf-8")
    questions = []
    skipped: list[str] = []
    for index, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        answerable = item["type"] != "no_answer"
        if answerable and not item.get("expected_facts"):
            skipped.append(item["question"])
            continue
        files = sorted({r["filename"] for r in item.get("relevant", [])})
        questions.append(
            {
                "id": f"acme-{index:03d}",
                "question": item["question"],
                "group": item["type"],
                "answerable": answerable,
                "answers": item.get("expected_facts", []) if answerable else [],
                "scheme": "facts",
                "relevant_files": files,
            }
        )
    manifest = {
        "name": "acme-in-domain",
        "source": str(dataset_path),
        "source_sha256": sha256_file(dataset_path),
        "documents": len(documents),
        "questions": len(questions),
        "answerable": sum(q["answerable"] for q in questions),
        "unanswerable": sum(not q["answerable"] for q in questions),
        "skipped_unscorable": skipped,
        "note": "PDF sample excluded (no question refers to it)",
    }
    _write_dataset(out_dir, documents, questions, manifest)
    return manifest


def load_questions(dataset_dir: Path) -> list[dict]:
    lines = (dataset_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
