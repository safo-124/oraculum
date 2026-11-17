"""Builds the model-backed services from Settings.

Encoders are loaded once per process (the loaders are cached) and shared by
every request; the chat model lives in a separate llama-server process.
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.services.confidence import ConfidenceGate, GatedGenerator, load_calibrator
from app.services.embeddings import Embedder
from app.services.generation import Generator
from app.services.llm_client import get_llm_client
from app.services.reranker import Reranker
from app.services.verifier import SupportVerifier

logger = logging.getLogger(__name__)


def build_embedder(settings: Settings) -> Embedder:
    return Embedder(
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        device=settings.model_device,
        query_prefix=settings.embedding_query_prefix,
    )


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.retrieval_rerank:
        return None
    return Reranker(model=settings.rerank_model, device=settings.model_device)


def build_verifier(settings: Settings) -> SupportVerifier | None:
    if not settings.verifier_enabled:
        return None
    return SupportVerifier(model=settings.verifier_model, device=settings.model_device)


def build_base_generator(settings: Settings) -> Generator:
    client = get_llm_client(
        settings.llm_base_url, settings.chat_model, settings.chat_timeout_seconds
    )
    return Generator(
        client=client,
        model=settings.chat_model,
        max_tokens=settings.chat_max_tokens,
        temperature=settings.chat_temperature,
        max_retries=settings.chat_max_retries,
        logprobs=settings.chat_logprobs,
    )


def build_gate(settings: Settings) -> ConfidenceGate:
    return ConfidenceGate(
        verifier=build_verifier(settings),
        calibrator=load_calibrator(str(settings.calibrator_path)),
        threshold=settings.confidence_threshold,
    )


def build_generator(settings: Settings) -> GatedGenerator:
    return GatedGenerator(build_base_generator(settings), build_gate(settings))


def preload_models(settings: Settings) -> None:
    """Warm every encoder so the first visitor does not wait for downloads."""
    for name, build in (
        ("embedder", build_embedder),
        ("reranker", build_reranker),
        ("verifier", build_verifier),
    ):
        try:
            build(settings)
        except Exception:
            logger.exception("Could not preload the %s", name)
    load_calibrator(str(settings.calibrator_path))
