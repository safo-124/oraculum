"""HTTP client for a locally served open-weight chat model.

Oraculum generates answers with an open model (default: Qwen2.5 Instruct,
GGUF-quantized) served by llama.cpp's `llama-server`. The server speaks the
widely used `/v1/chat/completions` wire format, so the same client also
works with Ollama, vLLM or Hugging Face TGI - no vendor SDK is involved,
only `httpx`.

What this client adds over a bare POST:
- JSON-schema constrained decoding (`response_format: json_schema`): the
  server compiles the schema to a grammar, so a small model cannot emit
  malformed JSON.
- Token log-probabilities (`logprobs` + `top_logprobs`): the raw material
  for the confidence model in app/services/confidence.py.
- Streaming over Server-Sent Events, with usage and log-probabilities
  collected from the chunks.
- A two-class error contract: LLMTransientError (connection problems,
  timeouts, 429, 5xx - worth a retry) and LLMRequestError (everything
  else - retrying cannot help).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for model-server failures."""


class LLMTransientError(LLMError):
    """A failure worth retrying: network, timeout, overload (429) or 5xx."""


class LLMRequestError(LLMError):
    """A failure retrying cannot fix: bad request, missing model, bad body."""


@dataclass(frozen=True)
class TokenLogprob:
    """One generated token with its log-probability and the top alternatives."""

    token: str
    logprob: float
    top: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Completion:
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    # Empty when the server does not return log-probabilities.
    logprobs: tuple[TokenLogprob, ...] = ()


@dataclass
class StreamChunk:
    """One streamed piece: text plus any log-probabilities and usage it carried."""

    text: str = ""
    logprobs: list[TokenLogprob] = field(default_factory=list)
    tokens_in: int | None = None
    tokens_out: int | None = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
    ) -> Completion: ...

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
    ) -> Iterator[StreamChunk]: ...


def _parse_logprobs(choice: dict) -> list[TokenLogprob]:
    block = choice.get("logprobs") or {}
    items = block.get("content") or []
    parsed: list[TokenLogprob] = []
    for item in items:
        try:
            top = tuple(
                (str(alt.get("token", "")), float(alt["logprob"]))
                for alt in item.get("top_logprobs") or []
                if "logprob" in alt
            )
            parsed.append(
                TokenLogprob(
                    token=str(item.get("token", "")),
                    logprob=float(item["logprob"]),
                    top=top,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return parsed


def _usage(payload: dict) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    return (
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
    )


class LlamaCppClient:
    """Talks to llama-server (or any server with the same chat wire format)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        top_logprobs: int = 5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._model = model
        self._top_logprobs = top_logprobs
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._model

    def _body(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
        logprobs: bool,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema, "strict": True},
            }
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = self._top_logprobs
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = response.text[:300] if not response.is_stream_consumed else ""
        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(
                f"model server returned {response.status_code} {detail}".strip()
            )
        raise LLMRequestError(
            f"model server returned {response.status_code} {detail}".strip()
        )

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
    ) -> Completion:
        body = self._body(
            messages,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            stream=False,
        )
        try:
            response = self._http.post("/v1/chat/completions", json=body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMTransientError(f"model server unreachable: {type(exc).__name__}") from exc
        self._raise_for_status(response)
        try:
            payload = response.json()
            choice = payload["choices"][0]
            content = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("model server returned an unexpected body") from exc
        tokens_in, tokens_out = _usage(payload)
        return Completion(
            content=content,
            tokens_in=tokens_in or 0,
            tokens_out=tokens_out or 0,
            logprobs=tuple(_parse_logprobs(choice)),
        )

    def stream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        json_schema: dict | None,
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
    ) -> Iterator[StreamChunk]:
        body = self._body(
            messages,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            stream=True,
        )
        try:
            with self._http.stream("POST", "/v1/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    response.read()
                    self._raise_for_status(response)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except ValueError:
                        logger.debug("Skipping unparsable stream line")
                        continue
                    chunk = StreamChunk()
                    chunk.tokens_in, chunk.tokens_out = _usage(payload)
                    choices = payload.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        chunk.text = delta.get("content") or ""
                        chunk.logprobs = _parse_logprobs(choices[0])
                    yield chunk
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LLMTransientError(f"model server unreachable: {type(exc).__name__}") from exc

    def health(self) -> bool:
        """True when the server answers its health endpoint (llama-server: /health)."""
        try:
            response = self._http.get("/health", timeout=3.0)
        except httpx.HTTPError:
            return False
        return response.status_code == 200


@lru_cache(maxsize=4)
def get_llm_client(base_url: str, model: str, timeout: float) -> LlamaCppClient:
    """One shared client (and connection pool) per server configuration."""
    return LlamaCppClient(base_url=base_url, model=model, timeout=timeout)
