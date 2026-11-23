"""Selección del proveedor generativo según configuración."""
from __future__ import annotations

import logging
from typing import Any

from app.llm.base import LLMClient

logger = logging.getLogger("healthrag.llm")


def get_llm(llm_cfg: dict[str, Any]) -> LLMClient | None:
    """Devuelve el cliente LLM o ``None`` para el modo extractivo (offline)."""
    provider = (llm_cfg.get("provider") or "extractive").lower()
    if provider in {"extractive", "none", "offline"}:
        logger.info("Proveedor LLM: extractivo (sin modelo generativo)")
        return None
    if provider in {"openai", "openai_compatible", "ollama", "groq"}:
        from app.llm.openai_compat import OpenAICompatibleClient
        client = OpenAICompatibleClient(llm_cfg)
        logger.info("Proveedor LLM: %s (%s)", provider, llm_cfg.get("base_url") or "api.openai.com")
        return client
    raise ValueError(f"Proveedor LLM desconocido: {provider}")
