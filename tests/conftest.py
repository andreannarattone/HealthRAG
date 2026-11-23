"""Fixtures compartidas: corpus sintético, configuración aislada y servicio."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import load_config
from app.ingestion.pdf_catalog import build_corpus_catalog
from app.ingestion.pdf_processor import process_corpus
from app.llm.base import LLMResponse
from tests.sample_corpus import build_sample_corpus


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("corpus")
    raw, meta = build_sample_corpus(d)
    build_corpus_catalog(raw, d / "catalog.csv", metadata_csv=meta)
    process_corpus(d / "catalog.csv", d / "chunks.jsonl", tokenizer_name=None)
    return d


@pytest.fixture()
def cfg(corpus_dir, tmp_path) -> dict:
    """Configuración offline: embeddings por hashing, sin reranker, sin LLM."""
    return load_config(overrides={
        "corpus": {"chunks_path": str(corpus_dir / "chunks.jsonl")},
        "retrieval": {"index_dir": str(tmp_path / "index"), "embedding_model": "hashing",
                      "reranker_model": None, "min_vector_score": 0.05},
        "llm": {"provider": "extractive"},
        "security": {"audit_log": str(tmp_path / "audit.jsonl"), "require_api_key": False},
        "evaluation": {"results_dir": str(tmp_path / "reports")},
        "logging": {"file": str(tmp_path / "test.log")},
    })


@pytest.fixture()
def retriever(cfg):
    from app.retrieval.retriever import HealthRetriever
    return HealthRetriever(cfg)


@pytest.fixture()
def service(cfg, retriever):
    from app.rag.pipeline import HealthRAGService
    return HealthRAGService(cfg, retriever=retriever, llm=None)


class ScriptedLLM:
    """LLM falso que devuelve respuestas predefinidas en orden."""

    provider = "scripted"

    def __init__(self, responses: list[dict | str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system, user, *, model=None, max_tokens=None, temperature=None,
                 json_mode=False) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "model": model})
        item = self.responses.pop(0)
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LLMResponse(text=text, model=model or "fake", input_tokens=100,
                           output_tokens=50, latency_ms=1.0)


@pytest.fixture()
def scripted_llm():
    return ScriptedLLM
