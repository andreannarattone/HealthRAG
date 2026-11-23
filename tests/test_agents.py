"""Agentes y comunicación dinámica (con LLM simulado)."""
from app.agents.analyst import AnalystAgent
from app.agents.verifier import VerifierAgent
from app.rag.pipeline import HealthRAGService

EVIDENCE = [
    {"eid": "E1", "doc_id": "D", "page": 1, "text": "El consumo de sal recomendado es menor a 5 gramos por día."},
    {"eid": "E2", "doc_id": "D", "page": 2, "text": "La actividad física regular es de 150 minutos semanales."},
]
AGENTS_CFG = {"max_iterations": 2, "min_supported_ratio": 0.8, "heuristic_support_threshold": 0.35}


def test_extractive_analyst_cites_evidence():
    draft = AnalystAgent(None, {})._extractive("¿cuánta sal por día se recomienda?", EVIDENCE, None)
    assert not draft["abstain"]
    assert draft["claims"][0]["citations"] == ["E1"]


def test_extractive_analyst_abstains_without_coverage():
    draft = AnalystAgent(None, {})._extractive("vacuna antigripal en adultos", EVIDENCE, None)
    assert draft["abstain"]


def test_heuristic_verifier_detects_wrong_number_and_citation():
    v = VerifierAgent(None, {}, AGENTS_CFG)
    ok = v.check_claim({"text": "Se recomienda menos de 5 gramos de sal por día", "citations": ["E1"]},
                       {e["eid"]: e for e in EVIDENCE})
    bad_number = v.check_claim({"text": "Se recomienda menos de 9 gramos de sal por día",
                                "citations": ["E1"]}, {e["eid"]: e for e in EVIDENCE})
    wrong_cite = v.check_claim({"text": "Se recomienda menos de 5 gramos de sal por día",
                                "citations": ["E2"]}, {e["eid"]: e for e in EVIDENCE})
    assert ok["supported"] == "yes"
    assert bad_number["score"] < ok["score"]
    assert not wrong_cite["citation_ok"] and "E1" in wrong_cite["better_citations"]


def test_policy_revise_then_finalize_partial():
    v = VerifierAgent(None, {}, AGENTS_CFG)
    draft = {"claims": [{"text": "Se recomienda menos de 5 gramos de sal por día", "citations": ["E1"]},
                        {"text": "El café está contraindicado siempre", "citations": ["E2"]}]}
    first, _ = v.run("sal", draft, EVIDENCE, iteration=1, expanded=False)
    assert first["verdict"] == "revise" and first["rejected_claims"] == ["El café está contraindicado siempre"]
    last, _ = v.run("sal", draft, EVIDENCE, iteration=2, expanded=False)
    assert last["verdict"] == "finalize_partial"


def test_multiagent_dialogue_with_llm(cfg, retriever, scripted_llm):
    """Analista propone una afirmación sin respaldo -> verificador pide revisión ->
    analista corrige -> verificador aprueba."""
    bad = {"answer": "La sal debe ser 0 g [E1]. El café es seguro [E1].",
           "claims": [{"text": "La sal debe ser 0 g", "citations": ["E1"]},
                      {"text": "El café es seguro", "citations": ["E1"]}], "abstain": False}
    verify_bad = {"claims": [{"index": 0, "supported": "no", "citation_ok": False},
                             {"index": 1, "supported": "no", "citation_ok": False}],
                  "verdict": "revise", "feedback": "Las afirmaciones no están en la evidencia."}
    good = {"answer": "Se recomienda menos de 5 gramos de sal por día [E1].",
            "claims": [{"text": "Se recomienda menos de 5 gramos de sal por día", "citations": ["E1"]}],
            "abstain": False}
    verify_good = {"claims": [{"index": 0, "supported": "yes", "citation_ok": True}],
                   "verdict": "approve", "feedback": ""}
    llm = scripted_llm(["esto no es json", bad, verify_bad, good, verify_good])
    service = HealthRAGService(cfg, retriever=retriever, llm=llm)
    r = service.answer("¿Cuánta sal se recomienda por día?", mode="multiagent", use_cache=False)

    assert r["verdict"] == "approve" and r["iterations"] == 2 and not r["abstained"]
    assert r["sources"] and r["sources"][0]["eid"] == "E1"
    assert r["tokens"] == 5 * 150 and r["cost_usd"] > 0
    # El feedback del verificador llegó al analista en el segundo turno.
    assert "Observaciones del Agente Verificador" in llm.calls[3]["user"]
    kinds = [(m["from"], m["type"]) for m in r["trace"]]
    assert ("verificador", "revise") in kinds and ("verificador", "approve") in kinds


def test_retrieve_more_when_analyst_abstains(cfg, retriever, scripted_llm):
    abstain = {"answer": "", "claims": [], "abstain": True, "abstain_reason": "falta evidencia"}
    llm = scripted_llm([abstain, abstain])
    service = HealthRAGService(cfg, retriever=retriever, llm=llm)
    r = service.answer("tema inexistente en el corpus", mode="multiagent", use_cache=False)
    steps = [m["from"] for m in r["trace"]]
    assert steps.count("recuperador") == 2          # hubo re-recuperación
    assert r["abstained"] and r["verdict"] == "abstain"
