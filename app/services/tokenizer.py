"""Token counting for chunking, matched to the embedding model.

Chunks are sized in the embedding model's own tokens: an encoder such as
bge-small reads at most 512 tokens and silently truncates the rest, so
counting with any other tokenizer would let chunk tails go unembedded.

The Hugging Face tokenizer is loaded with `tokenizers` (Rust, no PyTorch).
Windows of a long sentence are cut using character offsets, never by
decoding token ids: uncased WordPiece decoding would lowercase the text and
mangle whitespace. When the tokenizer cannot be loaded (offline test runs,
or CHUNK_TOKENIZER=regex) a deterministic word-piece fallback is used and
the choice is logged.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def windows(self, text: str, size: int, overlap: int) -> list[tuple[str, int]]:
        """Split text into overlapping windows of at most `size` tokens.

        Returns (window_text, token_count) pairs; window text is an exact
        slice of the input.
        """
        ...


def _windows_from_spans(
    text: str, spans: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    start = 0
    while start < len(spans):
        window = spans[start : start + size]
        piece = text[window[0][0] : window[-1][1]].strip()
        if piece:
            results.append((piece, len(window)))
        if start + size >= len(spans):
            break
        start += size - overlap
    return results


class RegexTokenCounter:
    """Fallback: words and punctuation marks each count as one token.

    Real subword tokenizers produce roughly 1.3 tokens per English word, so
    this undercounts; chunk_size_tokens leaves headroom below the encoder
    limit for exactly that reason.
    """

    name = "regex"
    _PIECE = re.compile(r"\w+|[^\w\s]")

    def count(self, text: str) -> int:
        return len(self._PIECE.findall(text))

    def windows(self, text: str, size: int, overlap: int) -> list[tuple[str, int]]:
        spans = [match.span() for match in self._PIECE.finditer(text)]
        return _windows_from_spans(text, spans, size, overlap)


class HuggingFaceTokenCounter:
    def __init__(self, model_name: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_pretrained(model_name)
        # Counting must not be capped by the model's truncation setting.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self.name = f"hf:{model_name}"

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def windows(self, text: str, size: int, overlap: int) -> list[tuple[str, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        spans = [span for span in encoding.offsets if span[1] > span[0]]
        return _windows_from_spans(text, spans, size, overlap)


@lru_cache(maxsize=4)
def get_token_counter(mode: str, model_name: str) -> TokenCounter:
    """`mode` is "embedding" (the embedding model's tokenizer) or "regex"."""
    if mode == "regex":
        return RegexTokenCounter()
    try:
        counter = HuggingFaceTokenCounter(model_name)
        logger.info("Chunk token counting uses %s", counter.name)
        return counter
    except Exception as exc:  # noqa: BLE001 - offline or unknown model
        logger.warning(
            "Could not load the %s tokenizer (%s); falling back to regex counting",
            model_name,
            type(exc).__name__,
        )
        return RegexTokenCounter()
