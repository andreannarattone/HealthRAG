"""Agente 2 — Verificador.

Audita cada afirmación del analista contra la evidencia citada y decide si
aprobar, pedir una revisión (retroalimentación), pedir más evidencia o
abstenerse. La decisión final combina el juicio del verificador con una
política determinística, para que el flujo sea coherente y reproducible.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agents.analyst import draft_to_json
from app.llm.base import LLMClient, LLMError, parse_json
from app.prompts import prompts as P
from app.utils.text import content_overlap, numbers_in

logger = logging.getLogger("healthrag.agents.verifier")

SUPPORT_VALUE = {"yes": 1.0, "partial": 0.5, "no": 0.0}
VERDICTS = {"approve", "revise", "retrieve_more", "abstain"}


class VerifierAgent:
    name = "verificador"

    def __init__(self, llm: LLMClient | None, llm_cfg: dict[str, Any],
                 agents_cfg: dict[str, Any]) -> None:
        self.llm = llm
        self.llm_cfg = llm_cfg
        self.cfg = agents_cfg
        self.threshold = float(agents_cfg.get("heuristic_support_threshold", 0.35))

    def run(self, question: str, draft: dict[str, Any], evidence: list[dict[str, Any]],
            iteration: int, expanded: bool) -> tuple[dict[str, Any], list[dict]]:
        usages: list[dict] = []
        if draft.get("abstain") or not draft.get("claims"):
            raw = {"claims": [], "verdict": "abstain", "feedback": draft.get("abstain_reason"),
                   "suggested_query": None}
        elif self.llm is None:
            raw = self._heuristic(draft, evidence)
        else:
            try:
                raw, usages = self._llm(question, draft, evidence)
            except LLMError as exc:
                logger.warning("Verificador LLM falló (%s); se usa verificación heurística.", exc)
                raw = self._heuristic(draft, evidence)
                raw["fallback"] = "heuristic"
        return self._policy(raw, draft, iteration, expanded), usages

    # ------------------------------------------------------------- heurística
    def check_claim(self, claim: dict[str, Any], evidence_by_id: dict[str, dict]) -> dict[str, Any]:
        """Respaldo léxico + coincidencia de valores numéricos (dosis, umbrales)."""
        def support(eid: str) -> float:
            ev = evidence_by_id.get(eid)
            if not ev:
                return 0.0
            score = content_overlap(claim["text"], ev["text"])
            missing_numbers = numbers_in(claim["text"]) - numbers_in(ev["text"])
            return score * (0.5 if missing_numbers else 1.0)

        cited = claim.get("citations") or []
        cited_score = max((support(e) for e in cited), default=0.0)
        better = [eid for eid in evidence_by_id if support(eid) >= self.threshold]
        if cited_score >= self.threshold:
            status = "yes"
        elif cited_score >= self.threshold / 2 or better:
            status = "partial"
        else:
            status = "no"
        return {"supported": status, "citation_ok": cited_score >= self.threshold,
                "better_citations": [b for b in better if b not in cited],
                "score": round(cited_score, 3)}

    def _heuristic(self, draft: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        by_id = {e["eid"]: e for e in evidence}
        claims = []
        for i, claim in enumerate(draft["claims"]):
            res = self.check_claim(claim, by_id)
            note = "" if res["supported"] == "yes" else "La evidencia citada no respalda claramente la afirmación."
            claims.append({"index": i, **res, "note": note})
        bad = [c for c in claims if c["supported"] != "yes"]
        verdict = "approve" if not bad else "revise"
        feedback = "; ".join(f"Afirmación {c['index']}: {c['note']}" for c in bad)
        return {"claims": claims, "contradictions": [], "verdict": verdict,
                "feedback": feedback, "suggested_query": None}

    # ------------------------------------------------------------------- LLM
    def _llm(self, question: str, draft: dict[str, Any],
             evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict]]:
        user = P.VERIFIER_USER.format(question=question, draft=draft_to_json(draft),
                                      evidence=P.format_evidence(evidence))
        resp = self.llm.complete(P.VERIFIER_SYSTEM, user,
                                 model=self.llm_cfg.get("verifier_model"), json_mode=True)
        raw = parse_json(resp.text)
        # Normalización defensiva de la salida del modelo.
        claims = []
        for i, c in enumerate(raw.get("claims") or []):
            if not isinstance(c, dict):
                continue
            sup = str(c.get("supported", "no")).lower()
            claims.append({
                "index": int(c.get("index", i)) if str(c.get("index", i)).isdigit() else i,
                "supported": sup if sup in SUPPORT_VALUE else "no",
                "citation_ok": bool(c.get("citation_ok", False)),
                "better_citations": [b for b in c.get("better_citations") or [] if isinstance(b, str)],
                "note": str(c.get("note", "")),
            })
        raw["claims"] = claims
        if raw.get("verdict") not in VERDICTS:
            raw["verdict"] = "revise"
        return raw, [resp.usage(self.name)]

    # ---------------------------------------------------------------- política
    def _policy(self, raw: dict[str, Any], draft: dict[str, Any], iteration: int,
                expanded: bool) -> dict[str, Any]:
        max_iter = int(self.cfg.get("max_iterations", 2))
        min_ratio = float(self.cfg.get("min_supported_ratio", 0.8))
        n_claims = len(draft.get("claims") or [])
        # Afirmaciones no evaluadas por el verificador se consideran no respaldadas.
        status = {c["index"]: c["supported"] for c in raw.get("claims", [])}
        values = [SUPPORT_VALUE.get(status.get(i, "no"), 0.0) for i in range(n_claims)]
        ratio = sum(values) / n_claims if n_claims else 0.0
        can_iterate = iteration < max_iter

        llm_verdict = raw.get("verdict", "revise")
        if n_claims == 0 or llm_verdict == "abstain":
            verdict = "retrieve_more" if (can_iterate and not expanded) else "abstain"
        elif ratio >= min_ratio and llm_verdict in {"approve", "revise"}:
            verdict = "approve"
        elif can_iterate:
            verdict = "retrieve_more" if (llm_verdict == "retrieve_more" and not expanded) else "revise"
        else:
            verdict = "finalize_partial"

        rejected = [draft["claims"][i]["text"] for i in range(n_claims)
                    if status.get(i, "no") == "no"]
        return {**raw, "verdict": verdict, "llm_verdict": llm_verdict,
                "supported_ratio": round(ratio, 3), "rejected_claims": rejected,
                "claim_status": [status.get(i, "no") for i in range(n_claims)]}
