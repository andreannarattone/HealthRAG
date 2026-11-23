"""Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = 60,
                           weights: Sequence[float] | None = None
                           ) -> list[tuple[Hashable, float]]:
    """Fusiona listas ordenadas: score(d) = sum_i w_i / (k + rank_i(d)).

    ``rankings`` contiene listas de ids ordenadas de mayor a menor
    relevancia. Devuelve (id, score) ordenado por score descendente; ante
    empate se conserva el orden de primera aparición (estable).
    """
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights debe tener la misma longitud que rankings")
    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += w / (k + rank)
            if doc not in first_seen:
                first_seen[doc] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
