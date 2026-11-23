"""Abstracción de modelos generativos y registro de consumo de tokens."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

logger = logging.getLogger("healthrag.llm")


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def usage(self, agent: str) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text")
        d["agent"] = agent
        return d


class LLMClient(Protocol):
    provider: str

    def complete(self, system: str, user: str, *, model: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None,
                 json_mode: bool = False) -> LLMResponse: ...


class LLMError(RuntimeError):
    """Error al invocar el proveedor generativo."""


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict[str, Any]:
    """Parsea JSON tolerando ```json fences``` o texto alrededor."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Respuesta no es JSON válido: {text[:200]!r}")


def estimate_cost(usages: list[dict[str, Any]], llm_cfg: dict[str, Any]) -> float:
    """Costo estimado en USD a partir de precios por millón de tokens."""
    pin = float(llm_cfg.get("price_input_per_1m", 0) or 0)
    pout = float(llm_cfg.get("price_output_per_1m", 0) or 0)
    tin = sum(u.get("input_tokens", 0) for u in usages)
    tout = sum(u.get("output_tokens", 0) for u in usages)
    return round(tin / 1e6 * pin + tout / 1e6 * pout, 6)
