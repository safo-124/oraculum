"""Index documents into the SHARED corpus (the one guests can ask about).

    uv run python scripts/load_demo_corpus.py                  # samples/
    uv run python scripts/load_demo_corpus.py path/a.md dir/   # files or folders

Each file becomes a shared document (no owner), exactly like an upload by
an API-key client, and is ingested synchronously. Files whose bytes are
already in the shared corpus are skipped, so re-running is safe.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.db import repositories  # noqa: E402
from app.services.ingestion import _open_session, build_embedder, ingest_document  # noqa: E402
from app.services.storage import DocumentStorage  # noqa: E402
from app.services.vector_store import get_vector_store  # noqa: E402

SUPPORTED = {".md", ".txt", ".pdf"}


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths or [str(ROOT / "samples")]:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED))
        elif path.suffix.lower() in SUPPORTED:
            files.append(path)
        else:
            print(f"skip {path} (unsupported type)")
    return files


def main(argv: list[str]) -> int:
    settings = get_settings()
    embedder = build_embedder(settings)
    store = get_vector_store()
    storage = DocumentStorage(settings.uploads_path)
    loaded = 0
    for path in collect(argv):
        data = path.read_bytes()
        content_hash = hashlib.sha256(data).hexdigest()
        session = _open_session()
        try:
            if repositories.find_by_content_hash(session, content_hash, owner=None):
                print(f"already indexed  {path.name}")
                continue
            extension = path.suffix.lower()
            document_id = storage.save(data, extension)
            repositories.create_document(
                session,
                document_id=document_id,
                original_filename=path.name,
                file_type=extension.lstrip("."),
                size_bytes=len(data),
                content_hash=content_hash,
                owner_user_id=None,
            )
        finally:
            session.close()
        ingest_document(document_id, embedder=embedder, store=store, storage=storage)
        loaded += 1
        print(f"indexed          {path.name}")
    print(f"done: {loaded} new document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
