"""Servicio HealthRAG: punto de entrada único usado por la API, la UI y la evaluación.

Flujo: validación de entrada -> caché -> grafo de agentes (recuperación,
analista, verificador) -> validación de salida -> auditoría -> acción final.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from app.actions.notifier import WebhookNotifier
from app.agents.analyst import AnalystAgent
from app.agents.graph import Mode, build_graph
from app.agents.verifier import VerifierAgent
from app.config import load_config, resolve_path
from app.llm.base import LLMClient, estimate_cost
from app.llm.factory import get_llm
from app.prompts.prompts import DISCLAIMER, PROMPT_VERSION
from app.retrieval.retriever import HealthRetriever
from app.security.audit import AuditLogger
from app.security.guardrails import redact_pii, validate_query
from app.utils.cache import TTLCache

logger = logging.getLogger("healthrag.pipeline")

MODES: tuple[Mode, ...] = ("basic", "advanced", "multiagent")


class HealthRAGService:
    def __init__(self, cfg: dict[str, Any] | None = None,
                 retriever: HealthRetriever | None = None,
                 llm: LLMClient | None | str = "auto") -> None:
        self.cfg = cfg or load_config()
        self.retriever = retriever or HealthRetriever(self.cfg)
        self.llm = get_llm(self.cfg["llm"]) if llm == "auto" else llm
        self.analyst = AnalystAgent(self.llm, self.cfg["llm"],
                                    term_weight=self.retriever.bm25.idf)
        self.verifier = VerifierAgent(self.llm, self.cfg["llm"], self.cfg["agents"])
        self.graphs = {
            mode: build_graph(mode, self.retriever, self.analyst,
                              self.verifier if mode == "multiagent" else None, self.cfg)
            for mode in MODES
        }
        c = self.cfg.get("cache", {})
        self.cache = TTLCache(int(c.get("max_items", 512)), float(c.get("ttl_seconds", 3600)))
        self.cache_enabled = bool(c.get("enabled", True))
        self.audit = AuditLogger(resolve_path(self.cfg["security"].get("audit_log", "logs/audit.jsonl")))
        self.notifier = WebhookNotifier(self.cfg.get("actions", {}))

    @property
    def provider(self) -> str:
        return self.llm.provider if self.llm else "extractive"

    def answer(self, query: str, mode: Mode = "multiagent", top_k: int | None = None,
               user_id: str = "anonymous", use_cache: bool | None = None) -> dict[str, Any]:
        if mode not in self.graphs:
            raise ValueError(f"Modo inválido: {mode}. Opciones: {MODES}")
        request_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        guard = validate_query(query, self.cfg["security"])

        base = {"request_id": request_id, "query": guard.query, "mode": mode,
                "provider": self.provider, "prompt_version": PROMPT_VERSION}
        if not guard.allowed:
            response = {**base, "blocked": True, "answer": " ".join(guard.reasons),
                        "abstained": True, "sources": [], "claims": [], "retrieved": [],
                        "warnings": guard.warnings, "flags": guard.flags,
                        "disclaimer": DISCLAIMER, "latency_ms": (time.perf_counter() - t0) * 1000,
                        "usage": [], "tokens": 0, "cost_usd": 0.0, "cache_hit": False}
            self._audit(response, user_id)
            logger.warning("Consulta bloqueada (%s): %s", request_id, guard.flags or guard.reasons)
            return response

        use_cache = self.cache_enabled if use_cache is None else use_cache
        key = TTLCache.make_key(q=guard.query.lower(), mode=mode, k=top_k,
                                provider=self.provider, pv=PROMPT_VERSION)
        if use_cache and (cached := self.cache.get(key)) is not None:
            response = {**cached, **base, "cache_hit": True,
                        "latency_ms": (time.perf_counter() - t0) * 1000,
                        "tokens": 0, "cost_usd": 0.0, "usage": []}
            self._audit(response, user_id)
            return response

        try:
            state = self.graphs[mode].invoke({
                "question": guard.query, "mode": mode,
                "top_k": top_k or int(self.cfg["retrieval"].get("top_k", 5)),
                "iteration": 0, "trace": [], "usage": [], "timings": [],
            }, {"recursion_limit": 25})
        except Exception:
            logger.exception("Error en el grafo de agentes (%s)", request_id)
            raise

        usage = state.get("usage", [])
        final = state["final"]
        response = {
            **base, **final, "blocked": False, "flags": guard.flags,
            "warnings": guard.warnings + final.get("warnings", []),
            "trace": state.get("trace", []), "timings": state.get("timings", []),
            "usage": usage,
            "tokens": sum(u["input_tokens"] + u["output_tokens"] for u in usage),
            "cost_usd": estimate_cost(usage, self.cfg["llm"]),
            "latency_ms": (time.perf_counter() - t0) * 1000, "cache_hit": False,
        }
        if use_cache:
            self.cache.set(key, response)
        self._audit(response, user_id)
        self.notifier.notify(response)
        logger.info("Consulta %s [%s] %.0f ms, abstención=%s, tokens=%d", request_id, mode,
                    response["latency_ms"], response["abstained"], response["tokens"])
        return response

    def _audit(self, r: dict[str, Any], user_id: str) -> None:
        self.audit.log({
            "request_id": r["request_id"], "user": user_id, "mode": r["mode"],
            "provider": r["provider"], "query": redact_pii(r["query"]), "blocked": r.get("blocked"),
            "flags": r.get("flags"), "abstained": r.get("abstained"),
            "verdict": r.get("verdict"), "supported_ratio": r.get("supported_ratio"),
            "sources": [s["chunk_id"] for s in r.get("sources", [])],
            "latency_ms": round(r.get("latency_ms", 0), 1), "tokens": r.get("tokens"),
            "cost_usd": r.get("cost_usd"), "cache_hit": r.get("cache_hit"),
        })
