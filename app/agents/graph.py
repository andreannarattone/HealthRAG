"""Orquestación con LangGraph: recuperación -> analista <-> verificador -> salida.

Modos (corresponden a los experimentos):
* ``basic``      EXP1: recuperación vectorial + analista.
* ``advanced``   EXP2: híbrida (BM25 + embeddings + RRF + Cross-Encoder) + analista.
* ``multiagent`` EXP3: advanced + verificador con ciclos de revisión y
  re-recuperación (comunicación dinámica entre agentes).
"""
from __future__ import annotations

import logging
import operator
import time
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.analyst import AnalystAgent
from app.agents.verifier import VerifierAgent
from app.ingestion.tokenizer import RegexTokenCounter
from app.prompts import prompts as P
from app.retrieval.retriever import HealthRetriever
from app.security.guardrails import sanitize_evidence, validate_output

logger = logging.getLogger("healthrag.agents.graph")

Mode = Literal["basic", "advanced", "multiagent"]
RETRIEVAL_MODE = {"basic": "basic", "advanced": "advanced", "multiagent": "advanced"}
_counter = RegexTokenCounter()


class AgentState(TypedDict, total=False):
    question: str
    mode: str
    top_k: int
    evidence: list[dict[str, Any]]
    seen_ids: list[str]
    weak_evidence: bool
    expanded: bool
    suggested_query: str | None
    iteration: int
    draft: dict[str, Any]
    verification: dict[str, Any] | None
    feedback: dict[str, Any] | None
    final: dict[str, Any]
    # Canales acumulativos (reducer = concatenación)
    trace: Annotated[list[dict[str, Any]], operator.add]
    usage: Annotated[list[dict[str, Any]], operator.add]
    timings: Annotated[list[dict[str, Any]], operator.add]


def _msg(sender: str, receiver: str, kind: str, content: Any) -> dict[str, Any]:
    """Mensaje estructurado entre agentes (queda en la traza para auditoría)."""
    return {"from": sender, "to": receiver, "type": kind, "content": content,
            "t": round(time.time(), 3)}


def build_graph(mode: Mode, retriever: HealthRetriever, analyst: AnalystAgent,
                verifier: VerifierAgent | None, cfg: dict[str, Any]):
    max_ctx = int(cfg["agents"].get("max_context_tokens", 1800))

    # ----------------------------------------------------------------- nodos
    def retrieve(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        seen = set(state.get("seen_ids", []))
        query = state.get("suggested_query") or state["question"]
        result = retriever.retrieve(query, mode=RETRIEVAL_MODE[state["mode"]],
                                    top_k=state.get("top_k"), exclude_ids=seen)
        evidence = list(state.get("evidence", []))
        used = sum(_counter.count(e["text"]) for e in evidence)
        flagged = []
        for r in result.chunks:
            text, was_flagged = sanitize_evidence(r.chunk.text)
            if was_flagged:
                flagged.append(r.chunk.chunk_id)
            n_tok = _counter.count(text)
            # Control de longitud de contexto (costo/latencia): se respeta el
            # presupuesto de tokens manteniendo siempre al menos un fragmento.
            if evidence and used + n_tok > max_ctx:
                continue
            used += n_tok
            evidence.append({**r.to_dict(), "text": text, "eid": f"E{len(evidence) + 1}",
                             "sanitized": was_flagged})
        content = {"query": query, "retrieved": result.chunk_ids,
                   "weak_evidence": result.weak_evidence, "context_tokens": used}
        if flagged:
            content["sanitized_chunks"] = flagged
        return {
            "evidence": evidence,
            "seen_ids": list(seen | set(result.chunk_ids)),
            "weak_evidence": result.weak_evidence if not state.get("evidence") else state.get("weak_evidence", False),
            "trace": [_msg("recuperador", "analista", "evidencia", content)],
            "timings": [{"step": "retrieval", **result.timings_ms,
                         "node_ms": (time.perf_counter() - t0) * 1000}],
        }

    def analyst_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        draft, usage = analyst.run(state["question"], state.get("evidence", []),
                                   state.get("feedback"))
        valid_ids = {e["eid"] for e in state.get("evidence", [])}
        draft, issues = validate_output(draft, valid_ids)
        if issues:
            draft["validation_issues"] = issues
        iteration = state.get("iteration", 0) + 1
        return {
            "draft": draft, "iteration": iteration,
            "trace": [_msg("analista", "verificador" if verifier else "salida", "borrador",
                           {"iteration": iteration, "abstain": draft.get("abstain"),
                            "n_claims": len(draft.get("claims", [])), "issues": issues})],
            "usage": usage,
            "timings": [{"step": f"analyst_{iteration}", "node_ms": (time.perf_counter() - t0) * 1000}],
        }

    def verifier_node(state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        verification, usage = verifier.run(state["question"], state["draft"],
                                           state.get("evidence", []), state.get("iteration", 1),
                                           state.get("expanded", False))
        update: dict[str, Any] = {
            "verification": verification, "usage": usage,
            "trace": [_msg("verificador", "analista", verification["verdict"],
                           {"supported_ratio": verification.get("supported_ratio"),
                            "feedback": verification.get("feedback"),
                            "llm_verdict": verification.get("llm_verdict")})],
            "timings": [{"step": f"verifier_{state.get('iteration', 1)}",
                         "node_ms": (time.perf_counter() - t0) * 1000}],
        }
        if verification["verdict"] == "revise":
            update["feedback"] = {"text": verification.get("feedback") or "",
                                  "rejected_claims": verification.get("rejected_claims", [])}
        elif verification["verdict"] == "retrieve_more":
            update["expanded"] = True
            update["suggested_query"] = verification.get("suggested_query") or state["question"]
            update["feedback"] = {"text": "Se agregó evidencia adicional. " +
                                  (verification.get("feedback") or ""),
                                  "rejected_claims": verification.get("rejected_claims", [])}
        return update

    def finalize(state: AgentState) -> dict[str, Any]:
        return {"final": compose_final(state)}

    # ------------------------------------------------------------- aristas
    def after_analyst(state: AgentState) -> str:
        return "verifier" if verifier is not None else "finalize"

    def after_verifier(state: AgentState) -> str:
        verdict = state["verification"]["verdict"]
        return {"revise": "analyst", "retrieve_more": "retrieve"}.get(verdict, "finalize")

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("analyst", analyst_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "analyst")
    if verifier is not None:
        graph.add_node("verifier", verifier_node)
        graph.add_conditional_edges("analyst", after_analyst, ["verifier", "finalize"])
        graph.add_conditional_edges("verifier", after_verifier, ["analyst", "retrieve", "finalize"])
    else:
        graph.add_edge("analyst", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def compose_final(state: AgentState) -> dict[str, Any]:
    """Respuesta final con citas trazables, o abstención explícita."""
    draft = state.get("draft") or {}
    verification = state.get("verification")
    evidence = {e["eid"]: e for e in state.get("evidence", [])}
    claims = draft.get("claims", [])
    statuses = verification.get("claim_status") if verification else None

    abstained = bool(draft.get("abstain")) or (verification or {}).get("verdict") == "abstain"
    kept = claims
    note = None
    if not abstained and statuses:
        kept = [c for c, s in zip(claims, statuses) if s != "no"]
        if not kept:
            abstained = True
        elif len(kept) < len(claims):
            note = (f"Se omitieron {len(claims) - len(kept)} afirmación(es) que el verificador "
                    "no pudo respaldar con la evidencia.")

    if abstained:
        reason = draft.get("abstain_reason") or (verification or {}).get("feedback")
        answer = P.ABSTENTION_MESSAGE + (f" Motivo: {reason}" if reason else "")
        kept = []
    elif len(kept) == len(claims):
        answer = draft.get("answer", "")
    else:
        answer = " ".join(f"{c['text']} " + "".join(f"[{e}]" for e in c["citations"]) for c in kept)

    cited = sorted({eid for c in kept for eid in c.get("citations", [])},
                   key=lambda x: int(x[1:]))
    sources = [{k: evidence[e].get(k) for k in ("eid", "chunk_id", "doc_id", "title",
                                                 "organization", "publication_year", "version",
                                                 "section", "page", "source_url", "citation",
                                                 "text")}
               for e in cited if e in evidence]
    years = {s["publication_year"] for s in sources if s.get("publication_year")}
    warnings = []
    if len(years) > 1:
        warnings.append(f"La respuesta combina fuentes de distintos años ({', '.join(sorted(years))}).")
    if note:
        warnings.append(note)
    if state.get("weak_evidence") and not abstained:
        warnings.append("La relevancia de la evidencia recuperada es baja; verificá la fuente.")

    return {
        "answer": answer.strip(),
        "abstained": abstained,
        "claims": [{**c, "status": (statuses[claims.index(c)] if statuses else None)} for c in kept],
        "sources": sources,
        "retrieved": [{k: e.get(k) for k in ("eid", "chunk_id", "doc_id", "page", "section", "score")}
                      for e in state.get("evidence", [])],
        "verdict": (verification or {}).get("verdict"),
        "supported_ratio": (verification or {}).get("supported_ratio"),
        "iterations": state.get("iteration", 0),
        "warnings": warnings,
        "disclaimer": P.DISCLAIMER,
    }
