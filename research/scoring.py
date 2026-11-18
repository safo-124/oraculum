"""Answer correctness: SQuAD-style matching and expected-facts matching.

A generated answer is usually longer than the gold span ("The Normans came
from Normandy, in France." vs "France"), so strict exact match would mark
good answers wrong. An answer counts as correct when any of these holds
against any gold answer:
- exact match after SQuAD normalization;
- token F1 >= 0.5;
- the normalized gold is contained in the whole answer or in its FIRST
  SENTENCE, whichever of the two is short (<= 30 tokens). A model that states the answer and then
  adds context ("...curriculum standards. In addition, ...") is right; a long
  answer listing many candidates does not pass by luck.
For the in-domain "facts" datasets every expected fact (with `|`
alternatives) must appear in the answer, matching scripts/evaluate_chat.py.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
# A sentence ends at . ! or ? followed by whitespace and an upper-case letter
# or digit; decimals ("11.7") and initials ("J. I.") rarely match this.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PUNCT = set(string.punctuation)
MAX_CONTAINS_TOKENS = 30


def normalize(text: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation and articles, squash spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize(prediction).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def first_sentence(text: str) -> str:
    return _SENTENCE_END.split(text.strip(), maxsplit=1)[0]


def squad_scores(prediction: str, golds: Sequence[str]) -> dict:
    norm_pred = normalize(prediction)
    em = any(norm_pred == normalize(g) for g in golds)
    f1 = max((token_f1(prediction, g) for g in golds), default=0.0)
    def holds(text: str) -> bool:
        return len(text.split()) <= MAX_CONTAINS_TOKENS and any(
            normalize(g) and f" {normalize(g)} " in f" {text} " for g in golds
        )

    # The whole answer is short, or its first sentence is (answer, then context).
    contains = holds(norm_pred) or holds(normalize(first_sentence(prediction)))
    return {
        "em": bool(em),
        "f1": round(f1, 4),
        "contains": bool(contains),
        "correct": bool(em or f1 >= 0.5 or contains),
    }


def facts_scores(prediction: str, facts: Sequence[str]) -> dict:
    lowered = prediction.lower()
    hits = [
        any(alt.strip().lower() in lowered for alt in fact.split("|") if alt.strip())
        for fact in facts
    ]
    share = sum(hits) / len(hits) if hits else 0.0
    return {"fact_share": round(share, 4), "correct": bool(hits) and all(hits)}


def score_answer(prediction: str, golds: Sequence[str], scheme: str) -> dict:
    if scheme == "squad":
        return squad_scores(prediction, golds)
    if scheme == "facts":
        return facts_scores(prediction, golds)
    raise ValueError(f"unknown scoring scheme {scheme!r}")
