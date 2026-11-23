"""Experimentos comparativos EXP1–EXP4.

    python -m app.evaluation.run_experiments                 # EXP1-3 + recuperación
    python -m app.evaluation.run_experiments --exp4          # + caché y top-k
    python -m app.evaluation.run_experiments --limit 5       # prueba rápida
    python -m app.evaluation.run_experiments --exps EXP3 --out reports_exp3  # solo EXP3

Salidas en reports/: métricas por consulta, resumen CSV/Markdown, gráficos y
una muestra para revisión humana de fidelidad y citas.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import load_config, resolve_path
from app.evaluation.dataset import EvalQuery, load_queries
from app.evaluation import metrics as M
from app.rag.pipeline import HealthRAGService
from app.utils.logging_config import configure_logging

logger = logging.getLogger("healthrag.evaluation")

EXPERIMENTS = {"EXP1": "basic", "EXP2": "advanced", "EXP3": "multiagent"}


# --------------------------------------------------------------------------
def evaluate_retrieval(service: HealthRAGService, queries: list[EvalQuery],
                       ks: list[int]) -> pd.DataFrame:
    """Recuperación aislada: vectorial vs híbrida (RRF) vs híbrida + reranker."""
    rows = []
    kmax = max(ks)
    for mode in ("basic", "hybrid", "advanced"):
        for q in queries:
            if not q.qrels:
                continue
            res = service.retriever.retrieve(q.query, mode=mode, top_k=kmax)
            ret = [(r.chunk.doc_id, r.chunk.page) for r in res.chunks]
            rows.append({"retrieval_mode": mode, "query_id": q.query_id,
                         "page_level": q.page_level,
                         "retrieval_ms": res.timings_ms.get("total_ms", 0.0),
                         "useful_context@5": M.useful_context_ratio(ret[:5], q.qrels),
                         **M.retrieval_metrics(ret, q.qrels, ks)})
    return pd.DataFrame(rows)


def evaluate_end_to_end(service: HealthRAGService, queries: list[EvalQuery], mode: str,
                        exp: str, top_k: int | None = None, use_cache: bool = False
                        ) -> list[dict[str, Any]]:
    rows = []
    for q in queries:
        try:
            r = service.answer(q.query, mode=mode, top_k=top_k, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 - se registra y se sigue
            logger.exception("Fallo en %s/%s", exp, q.query_id)
            rows.append({"exp": exp, "mode": mode, "query_id": q.query_id, "error": str(exc)})
            continue
        sources = r.get("sources", [])
        retrieved = [(e["doc_id"], e["page"]) for e in r.get("retrieved", [])]
        fc = M.faithfulness_and_citations(r.get("claims", []), sources, service.verifier)
        cited_units = [(s["doc_id"], s["page"]) for s in sources]
        row = {
            "exp": exp, "mode": mode, "top_k": top_k, "query_id": q.query_id,
            "category": q.category, "query": q.query,
            "abstained": r["abstained"], "blocked": r.get("blocked", False),
            "correct_abstention": (r["abstained"] if q.category == "sin_evidencia" else None),
            "false_abstention": (r["abstained"] if q.category != "sin_evidencia" else None),
            "faithfulness": fc["faithfulness"], "citation_precision": fc["citation_precision"],
            "n_claims": fc["n_claims"],
            "cited_source_relevance": (M.useful_context_ratio(cited_units, q.qrels)
                                       if q.qrels and cited_units else None),
            "context_hit": (any(M.is_relevant(d, p, q.qrels) for d, p in retrieved)
                            if q.qrels else None),
            "redundancy": M.redundancy(r.get("claims", [])),
            "iterations": r.get("iterations"), "verdict": r.get("verdict"),
            "supported_ratio_verifier": r.get("supported_ratio"),
            "latency_ms": r["latency_ms"], "tokens": r.get("tokens", 0),
            "cost_usd": r.get("cost_usd", 0.0), "cache_hit": r.get("cache_hit", False),
            "answer": r["answer"],
            "sources": "; ".join(s["chunk_id"] for s in sources),
        }
        if q.reference_answer and not r["abstained"]:
            row["rougeL"] = M.rouge_l(r["answer"], q.reference_answer)
            row["bleu"] = M.bleu(r["answer"], q.reference_answer)
            row["semantic_sim"] = M.semantic_similarity(r["answer"], q.reference_answer,
                                                        service.retriever.embedder)
        rows.append(row)
    return rows


def summarize(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    agg = {
        "query_id": "count", "abstained": "mean", "correct_abstention": "mean",
        "false_abstention": "mean", "faithfulness": "mean", "citation_precision": "mean",
        "cited_source_relevance": "mean", "context_hit": "mean", "redundancy": "mean",
        "iterations": "mean", "tokens": "mean", "cost_usd": "mean",
    }
    for col in ("rougeL", "bleu", "semantic_sim"):
        if col in df.columns:
            agg[col] = "mean"
    agg = {k: v for k, v in agg.items() if k in df.columns}
    numeric = df.copy()
    for col in agg:
        if col != "query_id":
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    out = numeric.groupby(group, dropna=False).agg(agg).rename(columns={"query_id": "n"})
    lat = numeric.groupby(group, dropna=False)["latency_ms"]
    out["latency_p50_ms"] = lat.median()
    out["latency_p95_ms"] = lat.quantile(0.95)
    return out.reset_index().round(4)


def _plots(retr_summary: pd.DataFrame, e2e_summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib no instalado: se omiten gráficos")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [c for c in retr_summary.columns if c.startswith(("recall@", "mrr", "ndcg@5"))]
    retr_summary.set_index("retrieval_mode")[cols].T.plot.bar(ax=ax, rot=0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Recuperación por arquitectura")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_metrics.png", dpi=150)
    plt.close(fig)

    base = e2e_summary[e2e_summary["exp"].isin(EXPERIMENTS)]
    if not base.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        base.set_index("exp")[["faithfulness", "citation_precision", "context_hit"]].plot.bar(ax=axes[0], rot=0)
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Calidad de respuesta")
        base.set_index("exp")[["latency_p50_ms", "latency_p95_ms"]].plot.bar(ax=axes[1], rot=0)
        axes[1].set_title("Latencia (ms)")
        for a in axes:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "e2e_metrics.png", dpi=150)
        plt.close(fig)


def run(config_overrides: dict | None = None, limit: int | None = None, exp4: bool = False,
        queries_path: str | None = None, service: HealthRAGService | None = None,
        experiments: list[str] | None = None) -> dict[str, Path]:
    """Corre la evaluación. `experiments` permite elegir un subconjunto
    (p. ej. ["EXP3"]) para repetir solo ese experimento si se agota el cupo
    del proveedor LLM; por defecto corre EXP1-EXP3."""
    selected = experiments or list(EXPERIMENTS)
    unknown = [e for e in selected if e not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Experimentos desconocidos: {unknown}. Opciones: {list(EXPERIMENTS)}")
    cfg = load_config(overrides=config_overrides)
    configure_logging(cfg)
    ev = cfg["evaluation"]
    out_dir = resolve_path(ev.get("results_dir", "reports"))
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(resolve_path(queries_path or ev["queries_path"]))[:limit]
    service = service or HealthRAGService(cfg)
    ks = [int(k) for k in ev.get("ks", [1, 3, 5, 10])]
    t0 = time.perf_counter()
    logger.info("Evaluación: %d consultas, proveedor=%s", len(queries), service.provider)

    retr = evaluate_retrieval(service, queries, ks)
    metric_cols = [c for c in retr.columns if c not in {"retrieval_mode", "query_id", "page_level"}]
    retr_summary = retr.groupby("retrieval_mode")[metric_cols].mean().round(4).reset_index() \
        if not retr.empty else pd.DataFrame(columns=["retrieval_mode"])

    rows: list[dict[str, Any]] = []
    for exp, mode in EXPERIMENTS.items():
        if exp not in selected:
            continue
        logger.info("%s (%s)", exp, mode)
        rows += evaluate_end_to_end(service, queries, mode, exp)

    if exp4:
        service.cache.clear()
        for label in ("EXP4_cache_frio", "EXP4_cache_caliente"):
            rows += evaluate_end_to_end(service, queries, "multiagent", label, use_cache=True)
        for k in (3, 8):
            rows += evaluate_end_to_end(service, queries, "multiagent", f"EXP4_top{k}", top_k=k)

    e2e = pd.DataFrame(rows)
    e2e_summary = summarize(e2e, ["exp", "mode"])
    by_cat = summarize(e2e[e2e["exp"].isin(EXPERIMENTS)], ["exp", "category"])

    paths = {
        "retrieval_per_query": out_dir / "retrieval_per_query.csv",
        "retrieval_summary": out_dir / "retrieval_summary.csv",
        "e2e_per_query": out_dir / "e2e_per_query.csv",
        "e2e_summary": out_dir / "e2e_summary.csv",
        "by_category": out_dir / "e2e_by_category.csv",
        "human_review": out_dir / "human_review_sample.csv",
        "markdown": out_dir / "summary.md",
    }
    retr.to_csv(paths["retrieval_per_query"], index=False)
    retr_summary.to_csv(paths["retrieval_summary"], index=False)
    e2e.to_csv(paths["e2e_per_query"], index=False)
    e2e_summary.to_csv(paths["e2e_summary"], index=False)
    by_cat.to_csv(paths["by_category"], index=False)

    # Muestra estratificada para revisión humana de fidelidad/citas.
    answered = e2e[(e2e["exp"].isin(EXPERIMENTS)) & (~e2e["abstained"].astype(bool))]
    if not answered.empty:
        sample = pd.concat([g.sample(min(len(g), 10), random_state=42)
                            for _, g in answered.groupby("exp")])
        sample = sample[["exp", "query_id", "query", "answer", "sources"]].copy()
        sample["fidelidad_humana(0/0.5/1)"] = ""
        sample["citas_correctas(0/0.5/1)"] = ""
        sample["comentarios"] = ""
        sample.to_csv(paths["human_review"], index=False)

    n_page = sum(q.page_level for q in queries if q.qrels)
    n_qrels = sum(bool(q.qrels) for q in queries)
    md = [
        "# Resultados de evaluación — HealthRAG", "",
        f"- Consultas: {len(queries)} (con evidencia de referencia: {n_qrels}; "
        f"a nivel página: {n_page})",
        f"- Proveedor LLM: {service.provider} · Embeddings: {service.retriever.embedder.name} · "
        f"Reranker: {getattr(service.retriever.reranker, 'name', 'ninguno')}",
        f"- Duración total: {time.perf_counter() - t0:.1f} s", "",
        "## Recuperación", "",
        (retr_summary[[c for c in retr_summary.columns if c == "retrieval_mode" or c == "mrr"
                       or c.endswith(("@5", "@10")) or c == "retrieval_ms"]].to_markdown(index=False)
         if not retr_summary.empty else "_sin qrels_"),
        "", "(tabla completa con k=1,3,5,10 en `retrieval_summary.csv`)", "",
        "## Respuestas (EXP1–EXP4)", "", e2e_summary.to_markdown(index=False), "",
        "## Por categoría de pregunta", "", by_cat.to_markdown(index=False), "",
        "> Fidelidad y precisión de citas se estiman con un verificador léxico-numérico "
        "independiente del LLM; complementar con la revisión humana de "
        "`human_review_sample.csv`. Si las consultas tienen relevancia solo a nivel "
        "documento, recall/precision son optimistas.",
    ]
    paths["markdown"].write_text("\n".join(md), encoding="utf-8")
    _plots(retr_summary, e2e_summary, out_dir)
    (out_dir / "run_config.json").write_text(json.dumps(
        {"llm": cfg["llm"], "retrieval": cfg["retrieval"], "agents": cfg["agents"]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Resultados en %s", out_dir)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos HealthRAG")
    parser.add_argument("--limit", type=int, default=None, help="Usar solo N consultas")
    parser.add_argument("--exp4", action="store_true", help="Incluir EXP4 (caché y top-k)")
    parser.add_argument("--queries", default=None, help="CSV/XLSX de consultas alternativo")
    parser.add_argument("--exps", nargs="+", default=None, choices=list(EXPERIMENTS),
                        help="Correr solo estos experimentos (p. ej. --exps EXP3)")
    parser.add_argument("--out", default=None, help="Carpeta de resultados (por defecto reports/)")
    parser.add_argument("--mlflow", action="store_true",
                        help="Registrar la corrida en MLflow (ver app/evaluation/mlflow_tracking.py)")
    args = parser.parse_args()
    overrides = {"evaluation": {"results_dir": args.out}} if args.out else None
    paths = run(config_overrides=overrides, limit=args.limit, exp4=args.exp4,
                queries_path=args.queries, experiments=args.exps)
    print(paths["markdown"].read_text(encoding="utf-8"))
    if args.mlflow:
        from app.evaluation.mlflow_tracking import log_results

        provider = json.loads((paths["markdown"].parent / "run_config.json")
                              .read_text(encoding="utf-8"))["llm"]["provider"]
        log_results(paths["markdown"].parent,
                    "extractivo" if provider == "extractive" else "groq")


if __name__ == "__main__":
    main()
