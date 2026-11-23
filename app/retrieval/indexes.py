"""Índices de recuperación: léxico (BM25) y vectorial (FAISS)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from app.utils.text import analyze

logger = logging.getLogger("healthrag.retrieval.indexes")

try:
    import faiss
except ImportError:  # pragma: no cover - se usa búsqueda exacta con numpy
    faiss = None


class BM25Index:
    """BM25 Okapi sobre texto analizado (sin tildes, stopwords, stemming).

    Se reconstruye en memoria a partir de chunks.jsonl: con corpus de
    algunos miles de fragmentos tarda menos de un segundo.
    """

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        tokenized = [analyze(t) or ["_vacio_"] for t in texts]
        self._bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    def idf(self, term: str) -> float:
        """IDF del término analizado; los términos ausentes reciben el máximo."""
        idf = self._bm25.idf
        if not hasattr(self, "_max_idf"):
            self._max_idf = max(idf.values()) if idf else 1.0
        return max(idf.get(term, self._max_idf), 0.05)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = analyze(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class VectorIndex:
    """Producto interno sobre vectores normalizados (= similitud coseno)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim) if faiss else None
        self._matrix = np.zeros((0, dim), dtype="float32")

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else len(self._matrix)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = np.ascontiguousarray(query_vec.reshape(1, -1), dtype="float32")
        k = min(k, self.size)
        if k == 0:
            return []
        if self._index is not None:
            scores, ids = self._index.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self._matrix @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, directory: Path, manifest: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(directory / "vectors.faiss"))
        else:
            np.save(directory / "vectors.npy", self._matrix)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> tuple["VectorIndex", dict]:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        index = cls(manifest["dim"])
        if (directory / "vectors.faiss").exists() and faiss is not None:
            index._index = faiss.read_index(str(directory / "vectors.faiss"))
        elif (directory / "vectors.npy").exists():
            index._index = None
            index._matrix = np.load(directory / "vectors.npy")
        else:
            raise FileNotFoundError(f"No hay vectores en {directory}")
        return index, manifest
