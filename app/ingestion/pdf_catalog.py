"""Etapa 1: inspección técnica de los PDF y catálogo del corpus."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

try:
    import pymupdf as fitz
except ImportError:  # versiones antiguas de PyMuPDF
    import fitz
import pandas as pd

logger = logging.getLogger("healthrag.ingestion.catalog")

BIBLIO_COLUMNS = [
    "doc_id", "title", "organization", "publication_year", "version",
    "section", "page", "source_url",
]

REQUIRED_COLUMNS = BIBLIO_COLUMNS + [
    "file_name", "file_path", "sha256", "page_count", "pages_with_text",
    "total_text_chars", "is_probably_scanned", "status", "notes",
]

TECHNICAL_COLUMNS = [
    "file_name", "file_path", "sha256", "page_count", "pages_with_text",
    "total_text_chars", "is_probably_scanned", "status", "notes",
]


def sha256_file(path: Path) -> str:
    """Calcula SHA-256 por bloques para detectar archivos duplicados."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pdf(path: Path, min_text_chars_per_page: int = 30) -> dict[str, Any]:
    """Extrae metadatos técnicos del PDF sin modificar el documento."""
    result: dict[str, Any] = {
        "file_name": path.name,
        "file_path": str(path.as_posix()),
        "sha256": None,
        "page_count": 0,
        "pages_with_text": 0,
        "total_text_chars": 0,
        "is_probably_scanned": True,
        "status": "ok",
        "notes": "",
    }

    try:
        # Dentro del try: un archivo ilegible no debe cortar todo el catálogo.
        result["sha256"] = sha256_file(path)
        with fitz.open(path) as pdf:
            result["page_count"] = len(pdf)
            page_lengths = [len(page.get_text("text").strip()) for page in pdf]
            result["pages_with_text"] = sum(
                n >= min_text_chars_per_page for n in page_lengths
            )
            result["total_text_chars"] = sum(page_lengths)
            result["is_probably_scanned"] = (
                result["pages_with_text"] == 0
                or result["total_text_chars"] < min_text_chars_per_page
            )

            if result["page_count"] == 0:
                result["status"] = "invalid"
                result["notes"] = "El PDF no contiene páginas."
            elif result["is_probably_scanned"]:
                result["status"] = "review_required"
                result["notes"] = "No se detectó texto suficiente. Puede requerir OCR."
    except Exception as exc:  # noqa: BLE001 - se registra y se continúa
        logger.exception("Error inspeccionando %s", path)
        result["status"] = "error"
        result["notes"] = f"{type(exc).__name__}: {exc}"

    return result


def _append_note(series: pd.Series, mask: pd.Series, note: str) -> pd.Series:
    out = series.fillna("").astype(str)
    out.loc[mask] = (out.loc[mask] + " " + note).str.strip()
    return out


def build_corpus_catalog(
    raw_dir: str | Path,
    output_csv: str | Path,
    min_text_chars_per_page: int = 30,
    metadata_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Inspecciona los PDF de ``raw_dir`` y genera el catálogo del corpus.

    ``metadata_csv`` (opcional) es el CSV completado manualmente con los
    datos bibliográficos verificados; se une por ``file_name``.
    Nunca se infieren título, versión, año ni URL a partir del nombre.
    """
    raw_dir, output_csv = Path(raw_dir), Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = [inspect_pdf(p, min_text_chars_per_page)
            for p in sorted(raw_dir.rglob("*.pdf"))]
    technical = pd.DataFrame(rows, columns=TECHNICAL_COLUMNS)
    logger.info("PDF inspeccionados: %d", len(technical))

    if metadata_csv and Path(metadata_csv).exists():
        metadata = pd.read_csv(metadata_csv, dtype=str).dropna(how="all")
        if "file_name" not in metadata.columns:
            raise ValueError("metadata_csv debe incluir la columna file_name.")
        manual_notes = metadata.pop("notes") if "notes" in metadata.columns else None
        technical = technical.merge(metadata, on="file_name", how="left")
        if manual_notes is not None:
            # Se concatenan las notas manuales con las técnicas (no se pisan).
            notes_by_file = dict(zip(metadata["file_name"], manual_notes.fillna("")))
            technical["notes"] = [
                " ".join(x for x in (str(n or ""), notes_by_file.get(f, "")) if x).strip()
                for n, f in zip(technical["notes"], technical["file_name"])
            ]
        orphan = set(metadata["file_name"]) - set(technical["file_name"])
        if orphan:
            logger.warning("Metadatos sin PDF correspondiente: %s", sorted(orphan))

    for column in REQUIRED_COLUMNS:
        if column not in technical.columns:
            technical[column] = pd.NA

    if not technical.empty:
        # Duplicados por contenido (mismo hash en más de un archivo).
        dup_hash = technical["sha256"].notna() & technical["sha256"].duplicated(keep=False)
        technical.loc[dup_hash, "status"] = "duplicate_review"
        technical["notes"] = _append_note(
            technical["notes"], dup_hash,
            "Hash repetido: revisar si es una copia del mismo documento.")

        # doc_id es la clave de trazabilidad: debe existir y ser único.
        missing_id = technical["doc_id"].isna() & technical["status"].eq("ok")
        technical.loc[missing_id, "status"] = "missing_metadata"
        technical["notes"] = _append_note(
            technical["notes"], missing_id,
            "Falta doc_id: completar data/metadata_template.csv.")

        dup_id = technical["doc_id"].notna() & technical["doc_id"].duplicated(keep=False)
        technical.loc[dup_id, "status"] = "duplicate_doc_id"
        technical["notes"] = _append_note(
            technical["notes"], dup_id, "doc_id repetido en más de un archivo.")

    technical = technical[REQUIRED_COLUMNS]
    technical.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info("Catálogo guardado en %s", output_csv)
    return technical


def main() -> None:
    """CLI: python -m app.ingestion.pdf_catalog"""
    from app.config import load_config, resolve_path
    from app.utils.logging_config import configure_logging

    cfg = load_config()
    configure_logging(cfg)
    c = cfg["corpus"]
    catalog = build_corpus_catalog(
        raw_dir=resolve_path(c["raw_dir"]),
        output_csv=resolve_path(c["catalog_path"]),
        min_text_chars_per_page=c.get("min_text_chars_per_page", 30),
        metadata_csv=resolve_path(c["metadata_path"]),
    )
    print(catalog[["file_name", "doc_id", "page_count", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
