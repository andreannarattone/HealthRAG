"""Registro de las corridas de evaluación en MLflow.

    python -m app.evaluation.mlflow_tracking                # importa las corridas oficiales
    python -m app.evaluation.mlflow_tracking --folder reports_x --condicion groq
    mlflow ui --backend-store-uri sqlite:///mlflow.db       # explorar en http://localhost:5000

Cada carpeta de resultados (``reports_*``) queda como una corrida padre con la
configuración (modelos, top-k, umbrales) como parámetros. Adentro hay una
corrida hija por experimento (EXP1-EXP4) con sus métricas de respuesta, y una por
modo de recuperación con Hit, Recall, Precision, nDCG y MRR. Los CSV, gráficos y
el resumen se adjuntan como artefactos. ``ejecucion.log`` no se sube porque
contiene el identificador de la cuenta del proveedor LLM.

Las corridas ya importadas (misma carpeta y experimentos) se saltean, así que el
comando se puede repetir sin duplicar nada.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import resolve_path

logger = logging.getLogger("healthrag.evaluation.mlflow")

EXPERIMENT_NAME = "HealthRAG"
DEFAULT_URI = "sqlite:///" + resolve_path("mlflow.db").as_posix()
ARTIFACT_DIR = resolve_path("mlruns")
# Archivos que se suben como artefactos (nunca ejecucion.log).
ARTIFACTS = ("e2e_summary.csv", "e2e_per_query.csv", "e2e_by_category.csv",
             "retrieval_summary.csv", "retrieval_per_query.csv", "human_review_sample.csv",
             "summary.md", "run_config.json", "NOTA.md", "e2e_metrics.png", "retrieval_metrics.png")
E2E_METRICS = ("abstained", "correct_abstention", "false_abstention", "faithfulness",
               "citation_precision", "cited_source_relevance", "context_hit", "redundancy",
               "iterations", "tokens", "cost_usd", "rougeL", "bleu", "semantic_sim",
               "latency_p50_ms", "latency_p95_ms")


@dataclass
class Corrida:
    """Una carpeta de resultados y qué parte de ella se registra."""
    folder: str
    condicion: str                       # extractivo | groq
    descripcion: str
    fecha: str | None = None             # "AAAA-MM-DD HH:MM" (hora de fin de la corrida)
    exps: list[str] | None = None        # None = todos los del resumen
    retrieval: bool = True
    estado: str = "oficial"              # oficial | piloto
    tags: dict[str, str] = field(default_factory=dict)


# Corridas que usa el informe. EXP3 de reports_llm_v2 está descartado (ver su NOTA.md).
OFICIALES = [
    Corrida("reports_extractivo", "extractivo", "Línea de base sin LLM, EXP1-EXP4, 29 consultas",
            "2026-11-15 13:05"),
    Corrida("reports_groq", "groq", "Piloto con Groq sobre 10 consultas", "2026-11-15 13:46",
            retrieval=False, estado="piloto"),
    Corrida("reports_llm_v2", "groq", "Groq, EXP1 y EXP2 sobre 29 consultas", "2026-11-16 19:39",
            exps=["EXP1", "EXP2"], retrieval=False,
            tags={"nota": "EXP3 de esta carpeta descartado por el límite diario de Groq"}),
    Corrida("reports_llm_exp3_v2", "groq", "Groq, EXP3 repetido sobre 29 consultas",
            "2026-11-18 20:45", exps=["EXP3"], retrieval=False),
]


# --------------------------------------------------------------------------- utilidades
def metric_name(col: str) -> str:
    """MLflow no acepta "@" en los nombres: hit@5 -> hit_at_5."""
    return col.replace("@", "_at_")


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = "" if v is None else str(v)[:500]
    return out


def _ms(fecha: str | None) -> int | None:
    return int(datetime.strptime(fecha, "%Y-%m-%d %H:%M").timestamp() * 1000) if fecha else None


def _client(tracking_uri: str | None):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_URI
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_DIR.as_uri(),
                                          tags={"proyecto": "HealthRAG · PLN III"})
    else:
        exp_id = exp.experiment_id
    return client, exp_id


def _ya_importada(client, exp_id: str, key: str) -> bool:
    runs = client.search_runs([exp_id], filter_string=f"tags.healthrag_key = '{key}'", max_results=1)
    return bool(runs)


def _log_metrics(client, run_id: str, values: dict[str, float], ts: int | None) -> None:
    from mlflow.entities import Metric

    ts = ts or int(datetime.now().timestamp() * 1000)
    metrics = [Metric(metric_name(k), float(v), ts, 0) for k, v in values.items() if pd.notna(v)]
    for i in range(0, len(metrics), 900):
        client.log_batch(run_id, metrics=metrics[i:i + 900])


def _log_params(client, run_id: str, params: dict[str, str]) -> None:
    from mlflow.entities import Param

    items = [Param(k, v) for k, v in params.items()]
    for i in range(0, len(items), 90):
        client.log_batch(run_id, params=items[i:i + 90])


# --------------------------------------------------------------------------- registro
def log_folder(c: Corrida, tracking_uri: str | None = None) -> str | None:
    """Registra una carpeta de resultados. Devuelve el id de la corrida padre."""
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

    folder = resolve_path(c.folder)
    summary_path = folder / "e2e_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe {summary_path}")
    client, exp_id = _client(tracking_uri)
    summary = pd.read_csv(summary_path)
    exps = c.exps or summary["exp"].tolist()
    key = f"{c.folder}:{','.join(exps)}"
    if _ya_importada(client, exp_id, key):
        logger.info("Ya registrada: %s", key)
        print(f"Ya estaba registrada: {key}")
        return None

    cfg_path = folder / "run_config.json"
    params = flatten(json.loads(cfg_path.read_text(encoding="utf-8"))) if cfg_path.exists() else {}
    ts = _ms(c.fecha)
    base_tags = {"condicion": c.condicion, "carpeta": c.folder, "estado": c.estado,
                 **({"fecha_corrida": c.fecha} if c.fecha else {}), **c.tags}

    parent = client.create_run(exp_id, start_time=ts, run_name=f"{c.folder}",
                               tags={**base_tags, "healthrag_key": key,
                                     "mlflow.note.content": c.descripcion})
    pid = parent.info.run_id
    _log_params(client, pid, params)
    for name in ARTIFACTS:
        if (folder / name).exists():
            client.log_artifact(pid, str(folder / name))

    def child(name: str, tags: dict[str, str], metrics: dict[str, float]) -> None:
        run = client.create_run(exp_id, start_time=ts, run_name=name,
                                tags={**base_tags, **tags, MLFLOW_PARENT_RUN_ID: pid,
                                      MLFLOW_RUN_NAME: name})
        _log_params(client, run.info.run_id, params)
        _log_metrics(client, run.info.run_id, metrics, ts)
        client.set_terminated(run.info.run_id, end_time=ts)

    for _, row in summary[summary["exp"].isin(exps)].iterrows():
        metrics = {m: row[m] for m in E2E_METRICS if m in row}
        metrics["n_consultas"] = row.get("n")
        child(f"{row['exp']} · {c.condicion}", {"exp": row["exp"], "modo": row["mode"],
                                               "tipo": "respuesta"}, metrics)

    retr_path = folder / "retrieval_summary.csv"
    if c.retrieval and retr_path.exists():
        for _, row in pd.read_csv(retr_path).iterrows():
            metrics = {k: v for k, v in row.items() if k != "retrieval_mode"}
            child(f"recuperación · {row['retrieval_mode']}",
                  {"modo_recuperacion": row["retrieval_mode"], "tipo": "recuperacion"}, metrics)

    client.set_terminated(pid, end_time=ts)
    print(f"Registrada: {key} (run {pid})")
    return pid


def log_results(out_dir: Path | str, condicion: str, tracking_uri: str | None = None) -> str | None:
    """Registra una corrida recién terminada (lo usa run_experiments --mlflow)."""
    out_dir = Path(out_dir)
    folder = out_dir.relative_to(resolve_path(".")).as_posix() if out_dir.is_absolute() else str(out_dir)
    return log_folder(Corrida(folder, condicion, f"Corrida de evaluación ({condicion})",
                              datetime.now().strftime("%Y-%m-%d %H:%M")), tracking_uri)


def main() -> None:
    ap = argparse.ArgumentParser(description="Registra corridas de HealthRAG en MLflow")
    ap.add_argument("--folder", help="Registrar solo esta carpeta (por defecto, las oficiales)")
    ap.add_argument("--condicion", default="groq", help="extractivo | groq (con --folder)")
    ap.add_argument("--exps", nargs="+", default=None, help="Experimentos a registrar (con --folder)")
    ap.add_argument("--tracking-uri", default=None, help=f"Por defecto {DEFAULT_URI}")
    args = ap.parse_args()
    corridas = ([Corrida(args.folder, args.condicion, f"Corrida {args.folder}", exps=args.exps)]
                if args.folder else OFICIALES)
    for c in corridas:
        log_folder(c, args.tracking_uri)


if __name__ == "__main__":
    main()
