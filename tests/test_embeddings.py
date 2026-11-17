import numpy as np
import pytest
from fastapi import HTTPException

from app.api.documents import get_embedder
from app.config import Settings
from app.services.embeddings import Embedder, embed_queries
from app.services.errors import EmbeddingNotConfiguredError, EmbeddingServiceError


class FakeEncoder:
    """Stands in for SentenceTransformer: vector = [len(text), 1.0]."""

    def __init__(self, fail: Exception | None = None, drop_last: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail
        self.drop_last = drop_last

    def encode(self, sentences, **kwargs):
        self.calls.append({"sentences": list(sentences), **kwargs})
        if self.fail is not None:
            raise self.fail
        rows = [[float(len(text)), 1.0] for text in sentences]
        if self.drop_last:
            rows = rows[:-1]
        return np.asarray(rows)


def make_embedder(**overrides) -> tuple[Embedder, FakeEncoder]:
    encoder = overrides.pop("encoder", None) or FakeEncoder()
    defaults = dict(model="test-model", batch_size=8, encoder=encoder)
    defaults.update(overrides)
    return Embedder(**defaults), encoder


class TestEmbedder:
    def test_empty_input_skips_model(self) -> None:
        embedder, encoder = make_embedder()
        assert embedder.embed([]) == []
        assert encoder.calls == []

    def test_vectors_preserve_input_order(self) -> None:
        embedder, _ = make_embedder()
        vectors = embedder.embed(["a", "bbb", "cc"])
        assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]
        assert all(isinstance(x, float) for v in vectors for x in v)

    def test_encoder_called_with_normalization_and_batch_size(self) -> None:
        embedder, encoder = make_embedder(batch_size=4)
        embedder.embed(["x"])
        call = encoder.calls[0]
        assert call["normalize_embeddings"] is True
        assert call["batch_size"] == 4

    def test_query_prefix_applies_to_queries_only(self) -> None:
        embedder, encoder = make_embedder(query_prefix="query: ")
        embedder.embed(["doc text"])
        embedder.embed_queries(["question"])
        assert encoder.calls[0]["sentences"] == ["doc text"]
        assert encoder.calls[1]["sentences"] == ["query: question"]

    def test_encoder_failure_maps_to_service_error(self) -> None:
        embedder, _ = make_embedder(encoder=FakeEncoder(fail=RuntimeError("oom")))
        with pytest.raises(EmbeddingServiceError, match="RuntimeError"):
            embedder.embed(["x"])

    def test_wrong_vector_count_raises(self) -> None:
        embedder, _ = make_embedder(encoder=FakeEncoder(drop_last=True))
        with pytest.raises(EmbeddingServiceError, match="1 vectors for 2"):
            embedder.embed(["a", "b"])

    def test_model_name_exposed(self) -> None:
        embedder, _ = make_embedder(model="BAAI/bge-small-en-v1.5")
        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.retry_count == 0


class TestEmbedQueriesHelper:
    def test_falls_back_to_embed_for_minimal_fakes(self) -> None:
        class Minimal:
            def embed(self, texts):
                return [[9.0] for _ in texts]

        assert embed_queries(Minimal(), ["q"]) == [[9.0]]


class TestEmbedderDependency:
    def test_unloadable_model_maps_to_503(self, monkeypatch) -> None:
        import app.services.models as models

        def boom(settings):
            raise EmbeddingNotConfiguredError("Could not load embedding model 'x'.")

        monkeypatch.setattr(models, "build_embedder", boom)
        with pytest.raises(HTTPException) as excinfo:
            get_embedder(Settings())
        assert excinfo.value.status_code == 503
        assert "Could not load" in str(excinfo.value.detail)
