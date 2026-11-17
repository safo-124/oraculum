"""Run a benchmark through the served pipeline and log one JSON row per question.

The pipeline is exactly what the API serves - hybrid retrieval, cross-encoder
reranking, the open model behind llama-server - except that the confidence
gate is NOT applied: the runner records every model answer together with its
features, so the calibrator can be fitted and evaluated afterwards. NLI
features are computed for model refusals too (on their draft answer), which
the serving path skips.

Runs are resumable: rows already in the output file are skipped, so a long
CPU run can be stopped and restarted at any time.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.services.bm25_index import Bm25Index
from app.services.chunking import chunk_pages
from app.services.confidence import collect_features
from app.services.errors import GenerationServiceError
from app.services.models import (
    build_base_generator,
    build_embedder,
    build_reranker,
    build_verifier,
)
from app.services.retrieval import retrieve_documents
from app.services.text_extraction import extract_pages
from app.services.vector_store import VectorStore
from research.datasets import load_questions
from research.scoring import normalize, score_answer


def collection_name(dataset_dir: Path) -> str:
    return f"bench_{dataset_dir.name}"[:60]


def reset_index(settings: Settings, dataset_dir: Path) -> None:
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection(collection_name(dataset_dir))
    except Exception:
        pass
    bm25_path = settings.chroma_path.parent / f"bm25_{dataset_dir.name}.db"
    if bm25_path.exists():
        bm25_path.unlink()


def open_index(settings: Settings, dataset_dir: Path) -> tuple[VectorStore, Bm25Index]:
    store = VectorStore(settings.chroma_path, collection_name(dataset_dir))
    bm25 = Bm25Index(settings.chroma_path.parent / f"bm25_{dataset_dir.name}.db")
    return store, bm25


def ingest(dataset_dir: Path, store: VectorStore, embedder, bm25: Bm25Index) -> int:
    total = 0
    for path in sorted((dataset_dir / "corpus").glob("*.md")):
        pages = extract_pages(path.name, path.read_bytes())
        chunks = chunk_pages(pages, document_id=str(uuid.uuid4()), filename=path.name)
        store.upsert_chunks(chunks, embedder.embed([c.text for c in chunks]))
        bm25.index_chunks(chunks)
        total += len(chunks)
        print(f"  indexed {path.name}: {len(chunks)} chunks")
    return total


def gold_retrieved(item: dict, texts: Iterable[str]) -> bool | None:
    """Answerable questions: does any retrieved passage contain a gold answer?"""
    if not item["answerable"] or not item.get("answers"):
        return None
    passages = [f" {normalize(t)} " for t in texts]
    for gold in item["answers"]:
        for alternative in gold.split("|") if item.get("scheme") == "facts" else [gold]:
            needle = normalize(alternative)
            if needle and any(f" {needle} " in p for p in passages):
                return True
    return False


def run_one(item: dict, *, settings: Settings, embedder, store, bm25, reranker, generator, verifier) -> dict:
    row: dict = {
        "id": item["id"],
        "group": item["group"],
        "answerable": item["answerable"],
        "question": item["question"],
    }
    started = time.perf_counter()
    retrieval = retrieve_documents(
        item["question"],
        embedder=embedder,
        store=store,
        bm25=bm25,
        top_k=settings.retrieval_top_k,
        hybrid=settings.retrieval_hybrid,
        rrf_k=settings.retrieval_rrf_k,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )
    row["retrieval_ms"] = round(retrieval.elapsed_ms, 1)
    row["gold_retrieved"] = gold_retrieved(item, (h.text for h in retrieval.results))
    try:
        result = generator.generate(item["question"], retrieval.results)
    except GenerationServiceError as exc:
        row["error"] = str(exc)
        return row
    row["model_status"] = (
        "no_evidence" if result.abstain_reason == "no_evidence" else result.status
    )
    row["draft_answer"] = result.draft_answer
    row["cited"] = [s.source_id for s in result.sources]
    row["generation_ms"] = round(result.elapsed_ms, 1)
    row["tokens_in"], row["tokens_out"] = result.tokens_in, result.tokens_out
    row["features"] = (
        collect_features(item["question"], retrieval.results, result, verifier)
        if retrieval.results
        else {}
    )
    if item["answerable"] and result.draft_answer:
        row["score"] = score_answer(result.draft_answer, item["answers"], item["scheme"])
        row["draft_correct"] = row["score"]["correct"]
    else:
        row["draft_correct"] = False
    row["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return row


def run_benchmark(
    dataset_dir: Path,
    out_path: Path,
    settings: Settings,
    *,
    limit: int | None = None,
    reindex: bool = False,
) -> int:
    questions = load_questions(dataset_dir)[:limit] if limit else load_questions(dataset_dir)
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if "error" not in row:
                    done.add(row["id"])
    todo = [q for q in questions if q["id"] not in done]
    print(f"{dataset_dir.name}: {len(done)} done, {len(todo)} to run")
    if not todo:
        return 0

    embedder = build_embedder(settings)
    if reindex:
        reset_index(settings, dataset_dir)
    store, bm25 = open_index(settings, dataset_dir)
    if store.count() == 0:
        print("indexing corpus")
        ingest(dataset_dir, store, embedder, bm25)
    reranker = build_reranker(settings)
    verifier = build_verifier(settings)
    generator = build_base_generator(settings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(todo, start=1):
            row = run_one(
                item,
                settings=settings,
                embedder=embedder,
                store=store,
                bm25=bm25,
                reranker=reranker,
                generator=generator,
                verifier=verifier,
            )
            row["model"] = settings.chat_model
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (len(todo) - index)
            status = row.get("model_status", "error")
            print(
                f"[{index}/{len(todo)}] {status:<22} correct={row.get('draft_correct')!s:<5} "
                f"{row.get('total_ms', 0):>7.0f} ms  eta {eta / 60:5.1f} min  {item['question'][:50]}"
            )
    return len(todo)
