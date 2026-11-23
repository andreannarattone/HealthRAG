"""Modelos de embeddings (Sentence Transformers) con fallback offline."""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Any, Protocol, Sequence

import numpy as np

from app.utils.text import analyze, strip_accents

logger = logging.getLogger("healthrag.retrieval.embedder")


class Embedder(Protocol):
    name: str
    dim: int

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wrapper de sentence-transformers con prefijos (E5) y normalización L2."""

    def __init__(self, model_name: str, query_prefix: str = "",
                 passage_prefix: str = "", batch_size: int = 32,
                 device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # import diferido

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.batch_size = batch_size

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        vectors = self._model.encode(
            [f"{prefix}{t}" for t in texts], batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)


class HashingEmbedder:
    """Embedding léxico por *feature hashing* (sin descargas).

    No captura semántica real: solo se usa para pruebas automáticas o
    cuando no hay acceso a Hugging Face. Queda registrado en el log y en el
    manifiesto del índice para no confundirlo con el modelo real.
    """

    def __init__(self, dim: int = 768) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float32")
        words = analyze(text)
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        plain = strip_accents(text.lower())
        grams += [plain[i:i + 4] for i in range(0, max(len(plain) - 3, 0), 2)]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")

    encode_passages = encode_queries


@lru_cache(maxsize=2)
def _cached_st(model_name: str, qp: str, pp: str, bs: int) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name, qp, pp, bs)


def get_embedder(retrieval_cfg: dict[str, Any], allow_fallback: bool = True) -> Embedder:
    name = retrieval_cfg.get("embedding_model")
    if name and not str(name).startswith("hashing"):
        try:
            emb = _cached_st(name, retrieval_cfg.get("query_prefix", ""),
                             retrieval_cfg.get("passage_prefix", ""),
                             int(retrieval_cfg.get("embedding_batch_size", 32)))
            logger.info("Embeddings: %s (dim=%d)", name, emb.dim)
            return emb
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            logger.warning("No se pudo cargar '%s' (%s: %s). Fallback HashingEmbedder.",
                           name, type(exc).__name__, exc)
    return HashingEmbedder()
