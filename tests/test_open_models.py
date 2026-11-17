"""Open-model plumbing: llama-server client, tokenizer, reranker, verifier."""

import json

import httpx
import pytest

from app.schemas.documents import ExtractedPage
from app.schemas.retrieval import RetrievalResult
from app.services.chunking import chunk_pages
from app.services.llm_client import LlamaCppClient, LLMRequestError, LLMTransientError
from app.services.reranker import Reranker
from app.services.retrieval import retrieve_documents
from app.services.tokenizer import RegexTokenCounter
from app.services.verifier import SupportVerifier

from conftest import RecordingBm25, RecordingEmbedder, RecordingVectorStore


# ------------------------------------------------------------ llama-server


def _client(handler) -> LlamaCppClient:
    return LlamaCppClient(
        base_url="http://llm.test", model="qwen", transport=httpx.MockTransport(handler)
    )


def _completion_body(content: str) -> dict:
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {
                    "content": [
                        {"token": content[:4], "logprob": -0.5, "top_logprobs": [{"token": content[:4], "logprob": -0.5}, {"token": "x", "logprob": -2.0}]},
                        {"token": content[4:], "logprob": -0.1, "top_logprobs": []},
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 9},
    }


class TestLlamaCppClient:
    def test_complete_sends_schema_and_logprobs(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_completion_body('{"a": 1}'))

        result = _client(handler).complete(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            temperature=0.0,
            max_tokens=50,
            logprobs=True,
        )
        assert seen["path"] == "/v1/chat/completions"
        body = seen["body"]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
        assert body["logprobs"] is True and body["top_logprobs"] == 5
        assert body["stream"] is False
        assert result.content == '{"a": 1}'
        assert (result.tokens_in, result.tokens_out) == (30, 9)
        assert [t.token for t in result.logprobs] == ['{"a"', ": 1}"]
        assert result.logprobs[0].top[1] == ("x", -2.0)

    def test_no_schema_no_logprobs_keeps_body_minimal(self) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        result = _client(handler).complete(
            [], json_schema=None, temperature=0.2, max_tokens=5
        )
        assert "response_format" not in seen["body"]
        assert "logprobs" not in seen["body"]
        assert result.content == "ok" and result.logprobs == ()

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_overload_and_server_errors_are_transient(self, status) -> None:
        client = _client(lambda request: httpx.Response(status, text="busy"))
        with pytest.raises(LLMTransientError):
            client.complete([], json_schema=None, temperature=0, max_tokens=5)

    def test_client_errors_are_not_retryable(self) -> None:
        client = _client(lambda request: httpx.Response(400, text="bad grammar"))
        with pytest.raises(LLMRequestError, match="400"):
            client.complete([], json_schema=None, temperature=0, max_tokens=5)

    def test_unreachable_server_is_transient(self) -> None:
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(LLMTransientError, match="unreachable"):
            _client(handler).complete([], json_schema=None, temperature=0, max_tokens=5)

    def test_garbage_body_is_a_request_error(self) -> None:
        client = _client(lambda request: httpx.Response(200, json={"nope": 1}))
        with pytest.raises(LLMRequestError, match="unexpected body"):
            client.complete([], json_schema=None, temperature=0, max_tokens=5)

    def test_stream_parses_sse_chunks_usage_and_logprobs(self) -> None:
        frames = [
            {"choices": [{"delta": {"content": '{"st'}, "logprobs": {"content": [{"token": '{"st', "logprob": -0.2}]}}]},
            {"choices": [{"delta": {"content": 'atus"'}}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
        ]
        body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        chunks = list(
            _client(handler).stream([], json_schema=None, temperature=0, max_tokens=5, logprobs=True)
        )
        assert "".join(c.text for c in chunks) == '{"status"'
        assert chunks[0].logprobs[0].logprob == -0.2
        assert (chunks[-1].tokens_in, chunks[-1].tokens_out) == (12, 3)

    def test_stream_error_status_raises(self) -> None:
        client = _client(lambda request: httpx.Response(503, text="loading model"))
        with pytest.raises(LLMTransientError):
            list(client.stream([], json_schema=None, temperature=0, max_tokens=5))

    def test_health(self) -> None:
        assert _client(lambda r: httpx.Response(200, json={"status": "ok"})).health()
        assert not _client(lambda r: httpx.Response(503)).health()


# ------------------------------------------------------------ tokenizer


class TestTokenCounting:
    def test_regex_counts_words_and_punctuation(self) -> None:
        assert RegexTokenCounter().count("Hello, world!") == 4

    def test_windows_are_exact_slices_with_overlap(self) -> None:
        text = "one two three four five six seven"
        windows = RegexTokenCounter().windows(text, size=3, overlap=1)
        assert [w for w, _ in windows] == [
            "one two three",
            "three four five",
            "five six seven",
        ]
        assert all(text.find(w) >= 0 for w, _ in windows)

    def test_long_sentence_split_keeps_original_case(self) -> None:
        sentence = " ".join(f"Word{i}" for i in range(40))
        chunks = chunk_pages(
            [ExtractedPage(page_number=1, text=sentence)],
            document_id="d",
            filename="f.md",
            chunk_size_tokens=10,
            chunk_overlap_tokens=2,
        )
        assert len(chunks) > 1
        assert chunks[0].text.startswith("Word0")
        assert all(c.token_count <= 10 for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ------------------------------------------------------------ reranker


class FakePairScorer:
    def __init__(self, scores=None, logits=None, id2label=None) -> None:
        self.scores = scores
        self.logits = logits
        self.calls: list = []
        if id2label is not None:
            config = type("C", (), {"id2label": id2label})()
            self.model = type("M", (), {"config": config})()

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        if self.logits is not None:
            return [self.logits(p) for p in pairs]
        return [self.scores(p) for p in pairs]


def _hit(i: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        source_id=f"d#{i}", document_id="d", filename="f.md", page=1,
        chunk_index=i, text=text, score=0.1,
    )


class TestReranker:
    def test_orders_by_cross_encoder_and_attaches_scores(self) -> None:
        scorer = FakePairScorer(scores=lambda pair: 5.0 if "leave" in pair[1] else -3.0)
        reranker = Reranker(model="ce", scorer=scorer)
        hits = [_hit(0, "VPN setup"), _hit(1, "annual leave is 20 days"), _hit(2, "printers")]
        ranked = reranker.rerank("leave?", hits, top_k=2)
        assert [h.source_id for h in ranked] == ["d#1", "d#0"]
        assert ranked[0].rerank_score == 5.0
        assert scorer.calls[0][0] == ("leave?", "VPN setup")

    def test_retrieval_widens_candidates_then_cuts_to_top_k(self) -> None:
        store = RecordingVectorStore()
        store.query_result = [
            {"id": f"d#{i}", "score": 0.9 - i / 100, "text": f"passage {i}",
             "metadata": {"document_id": "d", "filename": "f.md", "page": 1, "chunk_index": i}}
            for i in range(10)
        ]
        scorer = FakePairScorer(scores=lambda pair: float(pair[1].split()[-1]))
        outcome = retrieve_documents(
            "q", embedder=RecordingEmbedder(), store=store, bm25=RecordingBm25(),
            top_k=3, hybrid=True, reranker=Reranker(model="ce", scorer=scorer),
            rerank_candidates=10,
        )
        assert store.queries[0]["top_k"] == 10  # candidates, not top_k
        assert [r.chunk_index for r in outcome.results] == [9, 8, 7]
        assert outcome.results[0].rerank_score == 9.0

    def test_reranker_failure_keeps_fused_order(self) -> None:
        class Broken:
            def predict(self, pairs, **kwargs):
                raise RuntimeError("oom")

        store = RecordingVectorStore()
        store.query_result = [
            {"id": f"d#{i}", "score": 0.9, "text": "t",
             "metadata": {"document_id": "d", "filename": "f.md", "page": 1, "chunk_index": i}}
            for i in range(5)
        ]
        outcome = retrieve_documents(
            "q", embedder=RecordingEmbedder(), store=store, top_k=2,
            reranker=Reranker(model="ce", scorer=Broken()),
        )
        assert len(outcome.results) == 2
        assert outcome.results[0].rerank_score is None


# ------------------------------------------------------------ verifier


class TestSupportVerifier:
    def test_max_entailment_over_passages_and_label_order_from_config(self) -> None:
        # Label order deliberately non-default: entailment=0, neutral=1, contradiction=2.
        def logits(pair):
            passage, hypothesis = pair
            if "20 days" in passage:
                return [4.0, 0.0, -4.0]  # entailed
            return [-4.0, 4.0, 0.0]      # neutral

        scorer = FakePairScorer(
            logits=logits, id2label={0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        )
        verifier = SupportVerifier(model="nli", scorer=scorer)
        scores = verifier.score("How much leave?", "20 days", ["VPN text", "Staff get 20 days."])
        assert scores.entail > 0.9 and scores.entail_qa > 0.9
        assert scores.contradiction < 0.1
        assert len(scorer.calls[0]) == 4  # 2 passages x 2 hypotheses

    def test_nothing_to_verify_returns_none(self) -> None:
        verifier = SupportVerifier(model="nli", scorer=FakePairScorer(scores=lambda p: 0))
        assert verifier.score("q", "", ["p"]) is None
        assert verifier.score("q", "a", ["  "]) is None
