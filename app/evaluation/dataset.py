"""Conjunto de evaluación: carga y conversión desde el Excel original.

Formato canónico (data/evaluation/queries.csv):

    query_id,query,category,relevant,reference_answer,notes

* category: directa | sintesis | comparacion | sin_evidencia
* relevant: unidades separadas por ";" con formato ``DOC_ID`` o ``DOC_ID:PAG``
  (p. ej. ``GPCN_IAM:21; GPCN_IAM:22``). Sin página = cualquier página del
  documento (relevancia a nivel documento, menos estricta).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CATEGORIES = {"directa", "sintesis", "comparacion", "sin_evidencia"}


@dataclass
class EvalQuery:
    query_id: str
    query: str
    category: str
    qrels: list[tuple[str, int | None]] = field(default_factory=list)
    reference_answer: str | None = None

    @property
    def page_level(self) -> bool:
        return bool(self.qrels) and all(p is not None for _, p in self.qrels)


def parse_relevant(value: str | float | None) -> list[tuple[str, int | None]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    units = []
    for item in str(value).split(";"):
        item = item.strip()
        if not item:
            continue
        doc, _, page = item.partition(":")
        units.append((doc.strip(), int(page) if page.strip().isdigit() else None))
    return units


def _legacy_label(doc_label: str) -> tuple[str, int | None]:
    """'GPCN_IAM_p21' -> ('GPCN_IAM', 21); 'GPCN_HTA_p..' -> ('GPCN_HTA', None)."""
    m = re.match(r"^(.*?)_p(\d+|\.\.|)$", doc_label.strip())
    if not m:
        return doc_label.strip(), None
    return m.group(1), int(m.group(2)) if m.group(2).isdigit() else None


def convert_legacy_excel(xlsx_path: str | Path) -> pd.DataFrame:
    """Convierte el Excel original (una columna 'query,doc_id,label')."""
    raw = pd.read_excel(xlsx_path, header=None, dtype=str)[0].dropna().tolist()
    rows = []
    for line in raw[1:] if raw and raw[0].lower().startswith("query") else raw:
        query, doc_label, _label = line.rsplit(",", 2)
        doc, page = _legacy_label(doc_label)
        rows.append({"query": query.strip(), "category": "directa",
                     "relevant": f"{doc}:{page}" if page else doc,
                     "reference_answer": "", "notes": "" if page else "completar página"})
    df = pd.DataFrame(rows)
    df.insert(0, "query_id", [f"Q{i:03d}" for i in range(1, len(df) + 1)])
    return df


def load_queries(path: str | Path) -> list[EvalQuery]:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = convert_legacy_excel(path)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    queries = []
    for row in df.to_dict(orient="records"):
        category = (row.get("category") or "directa").strip()
        if category not in CATEGORIES:
            raise ValueError(f"Categoría inválida en {row.get('query_id')}: {category}")
        queries.append(EvalQuery(
            query_id=row["query_id"], query=row["query"].strip(), category=category,
            qrels=parse_relevant(row.get("relevant")),
            reference_answer=(row.get("reference_answer") or "").strip() or None,
        ))
    return queries


def main() -> None:
    """CLI: python -m app.evaluation.dataset querys-ejemplos.xlsx"""
    import sys

    from app.config import load_config, resolve_path

    src = sys.argv[1]
    out = resolve_path(load_config()["evaluation"]["queries_path"])
    df = convert_legacy_excel(src)
    if out.exists():
        print(f"{out} ya existe; se escribe {out.with_suffix('.converted.csv')}")
        out = out.with_suffix(".converted.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"{len(df)} consultas -> {out}")


if __name__ == "__main__":
    main()
