"""Esquemas de entrada/salida de la API (validación con Pydantic)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          examples=["¿Cuánta sal se recomienda consumir por día?"])
    mode: Literal["basic", "advanced", "multiagent"] = "multiagent"
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool | None = None
    include_trace: bool = False


class Source(BaseModel):
    eid: str
    chunk_id: str
    doc_id: str
    title: str | None = None
    organization: str | None = None
    publication_year: str | None = None
    version: str | None = None
    section: str | None = None
    page: int
    source_url: str | None = None
    citation: str | None = None
    text: str


class QueryResponse(BaseModel):
    request_id: str
    query: str
    mode: str
    provider: str
    answer: str
    abstained: bool
    blocked: bool = False
    verdict: str | None = None
    supported_ratio: float | None = None
    iterations: int | None = None
    claims: list[dict[str, Any]] = []
    sources: list[Source] = []
    warnings: list[str] = []
    flags: list[str] = []
    disclaimer: str
    latency_ms: float
    tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    trace: list[dict[str, Any]] | None = None
    timings: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str
    chunks: int
    documents: int
    embedding_model: str
    reranker: str | None
    llm_provider: str
    cache: dict[str, Any]
