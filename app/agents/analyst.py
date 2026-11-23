"""Agente 1 — Analista de evidencia.

Selecciona la evidencia pertinente, sintetiza una respuesta y la descompone
en afirmaciones atómicas con citas explícitas, para que el verificador
pueda auditarlas una por una.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from app.llm.base import LLMClient, LLMError, parse_json
from app.prompts import prompts as P
from app.utils.text import analyze

logger = logging.getLogger("healthrag.agents.analyst")

_SENT = re.compile(r"(?<=[.!?;])\s+")


class AnalystAgent:
    name = "analista"

    def __init__(self, llm: LLMClient | None, llm_cfg: dict[str, Any],
                 term_weight: Callable[[str], float] | None = None) -> None:
        self.llm = llm
        self.cfg = llm_cfg
        # Peso de cada término (IDF del corpus) para el modo extractivo.
        self.term_weight = term_weight

    def run(self, question: str, evidence: list[dict[str, Any]],
            feedback: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict]]:
        if not evidence:
            return {"answer": "", "claims": [], "abstain": True,
                    "abstain_reason": "No se recuperó evidencia."}, []
        if self.llm is None:
            return self._extractive(question, evidence, feedback, weight=self.term_weight), []
        try:
            return self._llm(question, evidence, feedback)
        except LLMError as exc:
            logger.warning("Analista LLM falló (%s); se usa modo extractivo.", exc)
            draft = self._extractive(question, evidence, feedback, weight=self.term_weight)
            draft["fallback"] = "extractive"
            return draft, []

    # ------------------------------------------------------------------ LLM
    def _llm(self, question: str, evidence: list[dict[str, Any]],
             feedback: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict]]:
        feedback_block = ""
        if feedback and feedback.get("text"):
            feedback_block = P.ANALYST_FEEDBACK.format(feedback=feedback["text"])
        user = P.ANALYST_USER.format(question=question, feedback_block=feedback_block,
                                     evidence=P.format_evidence(evidence))
        usages = []
        resp = self.llm.complete(P.ANALYST_SYSTEM, user, model=self.cfg.get("analyst_model"),
                                 json_mode=True)
        usages.append(resp.usage(self.name))
        try:
            draft = parse_json(resp.text)
        except LLMError:
            # Mecanismo de autocorrección: se pide de nuevo solo el JSON.
            resp = self.llm.complete(P.ANALYST_SYSTEM, user + "\n\nDevolvé SOLO JSON válido.",
                                     model=self.cfg.get("analyst_model"), json_mode=True)
            usages.append(resp.usage(self.name))
            draft = parse_json(resp.text)
        return draft, usages

    # ----------------------------------------------------------- extractivo
    @staticmethod
    def _extractive(question: str, evidence: list[dict[str, Any]],
                    feedback: dict[str, Any] | None, max_claims: int = 3,
                    min_score: float = 0.4,
                    weight: Callable[[str], float] | None = None) -> dict[str, Any]:
        """Selecciona oraciones de la evidencia con mayor cobertura de la consulta.

        Línea de base sin LLM: no puede alucinar contenido (copia oraciones
        de la fuente), pero tampoco sintetiza. Útil para tests y comparación.
        """
        q_terms = set(analyze(question))
        rejected = {r.strip() for r in (feedback or {}).get("rejected_claims", [])}
        if not q_terms:
            return {"answer": "", "claims": [], "abstain": True,
                    "abstain_reason": "La consulta no tiene términos de contenido."}
        candidates: list[tuple[float, str, str]] = []
        for rank, ev in enumerate(evidence):
            for sent in _SENT.split(ev["text"]):
                sent = sent.strip()
                if len(sent) < 25 or sent in rejected:
                    continue
                terms = set(analyze(sent))
                # Cobertura ponderada por especificidad: un término raro o
                # ausente del corpus ("migraña") pesa más que "tratamiento".
                w = weight or (lambda _t: 1.0)
                total = sum(w(t) for t in q_terms)
                score = sum(w(t) for t in q_terms & terms) / total - 0.01 * rank
                candidates.append((score, sent, ev["eid"]))
        candidates.sort(key=lambda c: -c[0])

        claims, seen = [], set()
        best = candidates[0][0] if candidates else 0.0
        for score, sent, eid in candidates:
            # Solo oraciones cercanas a la mejor: evita "relleno" poco pertinente.
            if score < max(min_score, 0.6 * best) or len(claims) >= max_claims:
                break
            key = " ".join(analyze(sent))
            if key in seen:
                continue
            seen.add(key)
            claims.append({"text": sent, "citations": [eid]})

        if not claims:
            return {"answer": "", "claims": [], "abstain": True,
                    "abstain_reason": "Ningún fragmento cubre los términos de la consulta."}
        answer = " ".join(f"{c['text']} [{c['citations'][0]}]" for c in claims)
        return {"answer": answer, "claims": claims, "abstain": False, "abstain_reason": None}


def draft_to_json(draft: dict[str, Any]) -> str:
    return json.dumps({k: draft.get(k) for k in ("answer", "claims", "abstain", "abstain_reason")},
                      ensure_ascii=False, indent=1)
