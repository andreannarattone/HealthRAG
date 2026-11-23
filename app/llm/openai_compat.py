"""Cliente para cualquier API compatible con OpenAI (OpenAI, Groq, Ollama...)."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from app.llm.base import LLMError, LLMResponse

logger = logging.getLogger("healthrag.llm.openai")


class OpenAICompatibleClient:
    provider = "openai_compatible"

    def __init__(self, llm_cfg: dict[str, Any]) -> None:
        from openai import OpenAI  # import diferido

        api_key = os.getenv("OPENAI_API_KEY") or ("ollama" if llm_cfg.get("base_url") else None)
        if not api_key:
            raise LLMError("Falta OPENAI_API_KEY en el entorno (.env).")
        self.cfg = llm_cfg
        self._client = OpenAI(api_key=api_key, base_url=llm_cfg.get("base_url") or None,
                              timeout=float(llm_cfg.get("timeout_s", 60)),
                              # Reintentos ante 429/5xx (el SDK respeta retry-after).
                              # Subirlo en planes con límite de tokens por minuto (p. ej. Groq gratis).
                              max_retries=int(llm_cfg.get("max_retries", 2)))

    def complete(self, system: str, user: str, *, model: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None,
                 json_mode: bool = False) -> LLMResponse:
        model = model or self.cfg["analyst_model"]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.cfg.get("temperature", 0.0) if temperature is None else temperature,
            "max_tokens": max_tokens or int(self.cfg.get("max_tokens", 700)),
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # Reintento sin response_format si: (a) el servidor no lo soporta, o
            # (b) el modelo no logró producir JSON válido (Groq: "failed_generation",
            # típico cuando los tokens de razonamiento agotan max_tokens y el JSON
            # queda cortado). El texto libre se parsea luego con parse_json.
            if json_mode and ("response_format" in msg or "failed_generation" in msg
                              or "JSON" in msg):
                logger.warning("JSON estricto falló (%s); reintento sin response_format", model)
                kwargs.pop("response_format", None)
                kwargs["max_tokens"] = int(kwargs["max_tokens"]) * 2
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:  # noqa: BLE001
                    logger.error("Error LLM (%s) en reintento: %s", model, exc2)
                    raise LLMError(str(exc2)) from exc2
            else:
                logger.error("Error LLM (%s): %s", model, exc)
                raise LLMError(msg) from exc
        latency = (time.perf_counter() - t0) * 1000
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency,
        )
