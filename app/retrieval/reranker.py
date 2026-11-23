"""Reranking con Cross-Encoder."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol, Sequence

logger = logging.getLogger("healthrag.retrieval.reranker")


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Cross-encoder (query, pasaje) -> score de relevancia.

    Por defecto ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``: multilingüe,
    entrenado en mMARCO (incluye español) y liviano para CPU.
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder  # import diferido

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        return [float(s) for s in self._model.predict([(query, p) for p in passages])]


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None) -> Reranker | None:
    """Devuelve el reranker o ``None`` si no se puede cargar (se loguea)."""
    if not model_name:
        return None
    try:
        reranker = CrossEncoderReranker(model_name)
        logger.info("Reranker cargado: %s", model_name)
        return reranker
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reranker '%s' no disponible (%s). Se usa el orden RRF.",
                       model_name, type(exc).__name__)
        return None
