"""API FastAPI de HealthRAG.

Ejecutar:  uvicorn app.api.main:app --reload
Docs:      http://localhost:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.api.schemas import HealthResponse, QueryRequest, QueryResponse
from app.config import get_config
from app.rag.pipeline import HealthRAGService
from app.security.auth import Principal, RateLimiter, authenticate, load_api_keys
from app.utils.logging_config import configure_logging

logger = logging.getLogger("healthrag.api")
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    configure_logging(cfg)
    if "service" not in _state:          # los tests pueden inyectar un servicio
        _state["service"] = HealthRAGService(cfg)
    _state["limiter"] = RateLimiter(int(cfg["security"].get("rate_limit_per_minute", 30)))
    logger.info("API HealthRAG iniciada")
    yield
    _state.clear()


app = FastAPI(title="HealthRAG API", version="1.0.0", lifespan=lifespan,
              description="Consulta documental trazable sobre guías de hipertensión "
                          "arterial y salud cardiovascular (RAG híbrido + agentes).")


def get_service() -> HealthRAGService:
    service = _state.get("service")
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Servicio no inicializado")
    return service


def get_principal(x_api_key: str | None = Header(default=None)) -> Principal:
    service = get_service()
    require = bool(service.cfg["security"].get("require_api_key", False))
    principal = authenticate(x_api_key, require, load_api_keys())
    if principal is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key inválida o ausente",
                            headers={"WWW-Authenticate": "X-API-Key"})
    limiter: RateLimiter | None = _state.get("limiter")
    if limiter and not limiter.allow(principal.user_id):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Límite de consultas por minuto")
    return principal


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Error no controlado en %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno. Ver logs del servidor."})


@app.get("/health", response_model=HealthResponse)
def health(service: HealthRAGService = Depends(get_service)) -> HealthResponse:
    r = service.retriever
    return HealthResponse(
        status="ok", chunks=len(r.chunks), documents=len({c.doc_id for c in r.chunks}),
        embedding_model=r.embedder.name,
        reranker=getattr(r.reranker, "name", None), llm_provider=service.provider,
        cache=service.cache.stats(),
    )


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(req: QueryRequest, principal: Principal = Depends(get_principal),
          service: HealthRAGService = Depends(get_service)) -> dict[str, Any]:
    result = service.answer(req.question, mode=req.mode, top_k=req.top_k,
                            user_id=principal.user_id, use_cache=req.use_cache)
    if not req.include_trace:
        result = {k: v for k, v in result.items() if k not in {"trace", "timings"}}
    return result


@app.get("/sources/{chunk_id}")
def source(chunk_id: str, _: Principal = Depends(get_principal),
           service: HealthRAGService = Depends(get_service)) -> dict[str, Any]:
    chunk = service.retriever.by_id.get(chunk_id)
    if chunk is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Fragmento inexistente")
    return {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "title": chunk.title,
            "section": chunk.section, "page": chunk.page, "source_url": chunk.source_url,
            "citation": chunk.citation(), "text": chunk.text}


@app.get("/audit")
def audit(n: int = 50, principal: Principal = Depends(get_principal),
          service: HealthRAGService = Depends(get_service)) -> list[dict[str, Any]]:
    if service.cfg["security"].get("require_api_key") and principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Requiere rol admin")
    return service.audit.tail(min(max(n, 1), 500))
