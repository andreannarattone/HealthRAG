"""Recuperación: básica (vectorial), híbrida (BM25 + vectorial + RRF) y
avanzada (híbrida + reranking con Cross-Encoder).
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.config import resolve_path
from app.ingestion.pdf_processor import Chunk, read_chunks
from app.retrieval.embedder import Embedder, get_embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.indexes import BM25Index, VectorIndex
from app.retrieval.reranker import Reranker, get_reranker

logger = logging.getLogger("healthrag.retrieval")

RetrievalMode = Literal["basic", "hybrid", "advanced"]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        c = self.chunk
        return {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": c.title,
            "organization": c.organization, "publication_year": c.publication_year,
            "version": c.version, "section": c.section, "page": c.page,
            "source_url": c.source_url, "text": c.text, "score": round(self.score, 4),
            "rank": self.rank, "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "citation": c.citation(),
        }


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float]
    weak_evidence: bool

    @property
    def chunk_ids(self) -> list[str]:
        return [r.chunk.chunk_id for r in self.chunks]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_vector_index(chunks: list[Chunk], embedder: Embedder, index_dir: Path,
                       chunks_path: Path | None = None) -> VectorIndex:
    """Calcula embeddings de todos los chunks y persiste el índice."""
    t0 = time.perf_counter()
    vectors = embedder.encode_passages([c.text for c in chunks])
    index = VectorIndex(embedder.dim)
    index.add(vectors)
    manifest = {
        "embedding_model": embedder.name,
        "dim": embedder.dim,
        "n_chunks": len(chunks),
        "chunk_ids": [c.chunk_id for c in chunks],
        "chunks_sha256": _sha256(chunks_path) if chunks_path and chunks_path.exists() else None,
        "build_seconds": round(time.perf_counter() - t0, 2),
    }
    index.save(index_dir, manifest)
    logger.info("Índice vectorial: %d chunks, %s, %.1fs", len(chunks), embedder.name,
                manifest["build_seconds"])
    return index


class HealthRetriever:
    """Punto único de acceso a la recuperación de evidencia."""

    def __init__(self, cfg: dict[str, Any], chunks: list[Chunk] | None = None,
                 embedder: Embedder | None = None, reranker: Reranker | None | str = "auto",
                 rebuild: bool = False) -> None:
        self.cfg = cfg["retrieval"]
        chunks_path = resolve_path(cfg["corpus"]["chunks_path"])
        if chunks is None:
            if not chunks_path.exists():
                raise FileNotFoundError(
                    f"No existe {chunks_path}. Ejecutá primero la Etapa 2 "
                    "(python -m app.ingestion.pdf_processor).")
            chunks = read_chunks(chunks_path)
        if not chunks:
            raise ValueError("El corpus no tiene chunks.")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.embedder = embedder or get_embedder(self.cfg)
        self.reranker = get_reranker(self.cfg.get("reranker_model")) if reranker == "auto" else reranker

        t0 = time.perf_counter()
        self.bm25 = BM25Index([c.text for c in chunks])
        index_dir = resolve_path(self.cfg.get("index_dir", "data/index"))
        self.vector_index = self._load_or_build(index_dir, chunks_path, rebuild)
        logger.info("Retriever listo: %d chunks (%.0f ms)", len(chunks),
                    (time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------ índices
    def _load_or_build(self, index_dir: Path, chunks_path: Path, rebuild: bool) -> VectorIndex:
        if not rebuild and (index_dir / "manifest.json").exists():
            try:
                index, manifest = VectorIndex.load(index_dir)
                valid = (manifest.get("embedding_model") == self.embedder.name
                         and manifest.get("chunk_ids") == [c.chunk_id for c in self.chunks])
                if valid:
                    return index
                logger.warning("El índice no coincide con el corpus/modelo actual: se reconstruye.")
            except Exception:  # noqa: BLE001
                logger.exception("No se pudo leer el índice; se reconstruye.")
        return build_vector_index(self.chunks, self.embedder, index_dir, chunks_path)

    # ---------------------------------------------------------------- búsquedas
    @lru_cache(maxsize=1024)
    def _query_vector(self, query: str) -> np.ndarray:
        return self.embedder.encode_queries([query])[0]

    def vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.vector_index.search(self._query_vector(query), k)

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, k)

    def retrieve(self, query: str, mode: RetrievalMode = "advanced", top_k: int | None = None,
                 exclude_ids: set[str] | None = None) -> RetrievalResult:
        top_k = top_k or int(self.cfg.get("top_k", 5))
        n_cand = int(self.cfg.get("candidates_per_retriever", 30))
        exclude_ids = exclude_ids or set()
        timings: dict[str, float] = {}

        t = time.perf_counter()
        vec_hits = self.vector_search(query, n_cand + len(exclude_ids))
        timings["vector_ms"] = (time.perf_counter() - t) * 1000
        vec_scores = {i: s for i, s in vec_hits}

        if mode == "basic":
            ranked = [(i, s) for i, s in vec_hits if self.chunks[i].chunk_id not in exclude_ids]
            results = [RetrievedChunk(self.chunks[i], s, r, {"vector": s})
                       for r, (i, s) in enumerate(ranked[:top_k], start=1)]
            weak = not results or results[0].score < float(self.cfg.get("min_vector_score", 0.3))
            timings["total_ms"] = sum(timings.values())
            return RetrievalResult(query, mode, results, timings, weak)

        t = time.perf_counter()
        bm_hits = self.bm25_search(query, n_cand + len(exclude_ids))
        timings["bm25_ms"] = (time.perf_counter() - t) * 1000
        bm_scores = {i: s for i, s in bm_hits}

        t = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [[i for i, _ in vec_hits], [i for i, _ in bm_hits]],
            k=int(self.cfg.get("rrf_k", 60)))
        fused = [(i, s) for i, s in fused if self.chunks[i].chunk_id not in exclude_ids]
        timings["fusion_ms"] = (time.perf_counter() - t) * 1000

        def scores_for(i: int, rrf: float) -> dict[str, float]:
            d = {"rrf": rrf}
            if i in vec_scores:
                d["vector"] = vec_scores[i]
            if i in bm_scores:
                d["bm25"] = bm_scores[i]
            return d

        weak = False
        if mode == "advanced" and self.reranker is not None:
            candidates = fused[: int(self.cfg.get("rerank_candidates", 20))]
            t = time.perf_counter()
            rr = self.reranker.score(query, [self.chunks[i].text for i, _ in candidates])
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000
            ordered = sorted(zip(candidates, rr), key=lambda x: -x[1])
            results = [RetrievedChunk(self.chunks[i], s, r, {**scores_for(i, f), "rerank": s})
                       for r, ((i, f), s) in enumerate(ordered[:top_k], start=1)]
            weak = not results or results[0].score < float(self.cfg.get("min_rerank_score", -2.0))
        else:
            results = [RetrievedChunk(self.chunks[i], f, r, scores_for(i, f))
                       for r, (i, f) in enumerate(fused[:top_k], start=1)]
            # Sin reranker: evidencia débil si ninguno de los dos retrievers coincide
            # y la similitud vectorial del primero es baja.
            if results:
                top = results[0].scores
                weak = top.get("vector", 0.0) < float(self.cfg.get("min_vector_score", 0.3)) \
                    and "bm25" not in top
            else:
                weak = True

        timings["total_ms"] = sum(timings.values())
        return RetrievalResult(query, mode, results, timings, weak)


def main() -> None:
    """CLI: python -m app.retrieval.retriever [--rebuild] -> construye el índice."""
    import argparse

    from app.config import load_config
    from app.utils.logging_config import configure_logging

    parser = argparse.ArgumentParser(description="Construye/valida los índices de HealthRAG")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--query", help="Consulta de prueba")
    parser.add_argument("--mode", default="advanced", choices=["basic", "hybrid", "advanced"])
    args = parser.parse_args()

    cfg = load_config()
    configure_logging(cfg)
    retriever = HealthRetriever(cfg, rebuild=args.rebuild)
    if args.query:
        result = retriever.retrieve(args.query, mode=args.mode)
        for r in result.chunks:
            print(f"[{r.rank}] {r.score:.3f} {r.chunk.chunk_id} | {r.chunk.text[:100]}")
        print("timings:", {k: round(v, 1) for k, v in result.timings_ms.items()},
              "weak:", result.weak_evidence)


if __name__ == "__main__":
    main()
