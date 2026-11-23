"""Métricas de recuperación, generación, agentes y rendimiento."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np

from app.utils.text import analyze

# --------------------------------------------------------------------------
# Recuperación
# --------------------------------------------------------------------------
Unit = tuple[str, int | None]   # (doc_id, página) ; página None = cualquier página


def is_relevant(doc_id: str, page: int, qrels: Sequence[Unit]) -> Unit | None:
    """Devuelve la unidad de relevancia que cubre el chunk, si existe."""
    for d, p in qrels:
        if d == doc_id and (p is None or p == page):
            return (d, p)
    return None


def retrieval_metrics(retrieved: Sequence[tuple[str, int]], qrels: Sequence[Unit],
                      ks: Iterable[int]) -> dict[str, float]:
    """``retrieved``: lista ordenada de (doc_id, página) de cada chunk.

    * hit@k: al menos un fragmento relevante en el top-k.
    * recall@k: unidades relevantes (doc/página) cubiertas en el top-k.
    * precision@k: proporción de fragmentos relevantes en el top-k.
    * mrr: inversa del rango del primer relevante.
    * ndcg@k: ganancia binaria descontada normalizada.
    """
    units = set(qrels)
    rel_flags = [is_relevant(d, p, qrels) for d, p in retrieved]
    out: dict[str, float] = {}
    first = next((i for i, r in enumerate(rel_flags, 1) if r), None)
    out["mrr"] = 1.0 / first if first else 0.0
    for k in ks:
        top = rel_flags[:k]
        covered = {r for r in top if r}
        out[f"hit@{k}"] = float(bool(covered))
        out[f"recall@{k}"] = len(covered) / len(units) if units else 0.0
        out[f"precision@{k}"] = sum(bool(r) for r in top) / k
        dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(top, 1) if r)
        n_ideal = min(k, max(len(units), sum(bool(r) for r in rel_flags)) or 1)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, n_ideal + 1))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out


def useful_context_ratio(retrieved: Sequence[tuple[str, int]], qrels: Sequence[Unit]) -> float:
    """Cobertura de contexto útil: fracción del contexto entregado que es relevante."""
    if not retrieved:
        return 0.0
    return sum(bool(is_relevant(d, p, qrels)) for d, p in retrieved) / len(retrieved)


# --------------------------------------------------------------------------
# Similitud con respuesta de referencia
# --------------------------------------------------------------------------
def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 sobre tokens analizados (LCS)."""
    c, r = analyze(candidate, stem=True), analyze(reference, stem=True)
    if not c or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
    for i, ct in enumerate(c, 1):
        for j, rt in enumerate(r, 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ct == rt else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    if lcs == 0:
        return 0.0
    p, rec = lcs / len(c), lcs / len(r)
    return 2 * p * rec / (p + rec)


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """BLEU de oración con suavizado +1 (Lin & Och, 2004)."""
    c, r = analyze(candidate, stem=False, remove_stopwords=False), \
        analyze(reference, stem=False, remove_stopwords=False)
    if not c or not r:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cn = Counter(tuple(c[i:i + n]) for i in range(len(c) - n + 1))
        rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        overlap = sum(min(v, rn[g]) for g, v in cn.items())
        total = max(sum(cn.values()), 0)
        log_p += math.log((overlap + 1) / (total + 1))
    bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
    return bp * math.exp(log_p / max_n)


def semantic_similarity(a: str, b: str, embedder: Any) -> float:
    if not a or not b:
        return 0.0
    va, vb = embedder.encode_passages([a, b])
    return float(np.dot(va, vb))


# --------------------------------------------------------------------------
# Fidelidad y citas (heurísticas, mismas para todas las arquitecturas)
# --------------------------------------------------------------------------
def faithfulness_and_citations(claims: list[dict[str, Any]], sources: list[dict[str, Any]],
                               checker: Any) -> dict[str, float | None]:
    """Respaldo documental de las afirmaciones finales.

    ``checker`` es un VerifierAgent; se usa su chequeo heurístico (léxico +
    numérico) para que la métrica sea independiente del LLM evaluado.
    """
    if not claims:
        return {"faithfulness": None, "citation_precision": None, "n_claims": 0}
    by_id = {s["eid"]: s for s in sources}
    supported = 0.0
    cit_ok = cit_total = 0
    for claim in claims:
        res = checker.check_claim(claim, by_id)
        supported += {"yes": 1.0, "partial": 0.5}.get(res["supported"], 0.0)
        for eid in claim.get("citations", []):
            cit_total += 1
            single = checker.check_claim({**claim, "citations": [eid]}, by_id)
            cit_ok += single["citation_ok"]
    return {"faithfulness": supported / len(claims),
            "citation_precision": cit_ok / cit_total if cit_total else 0.0,
            "n_claims": len(claims)}


def redundancy(claims: list[dict[str, Any]], threshold: float = 0.8) -> float:
    """Fracción de afirmaciones casi duplicadas (Jaccard de términos)."""
    sets = [set(analyze(c["text"])) for c in claims]
    if len(sets) < 2:
        return 0.0
    dup = 0
    for i in range(1, len(sets)):
        if any(sets[i] and len(sets[i] & sets[j]) / len(sets[i] | sets[j]) >= threshold
               for j in range(i)):
            dup += 1
    return dup / len(sets)


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")
