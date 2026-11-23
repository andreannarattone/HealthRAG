"""Etapa 2: extracción, limpieza, detección de secciones y chunking.

Decisiones de diseño
--------------------
* Los chunks se generan **dentro de cada página** (no se mezclan páginas),
  para que cada cita apunte a una página exacta.
* El texto se divide en *unidades* (oraciones, ítems de lista o filas de
  tabla) y se empaquetan hasta ~``chunk_size`` tokens con ``overlap``
  tokens de solapamiento. Así no se corta una oración ni un valor de
  presión arterial a la mitad.
* La sección se toma del índice (TOC) del PDF cuando existe y se refina con
  heurísticas tipográficas (tamaño de fuente / negrita / numeración).
"""
from __future__ import annotations

import json
import logging
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    import pymupdf as fitz
except ImportError:  # versiones antiguas de PyMuPDF
    import fitz
import pandas as pd

from app.ingestion.tokenizer import TokenCounter, get_token_counter

logger = logging.getLogger("healthrag.ingestion.processor")

PROCESSABLE_STATUS = {"ok", "missing_metadata"}

_PAGE_NUMBER = re.compile(
    r"^\s*(p[áa]g(ina)?\.?\s*)?\d{1,4}(\s*(de|/|of)\s*\d{1,4})?\s*$", re.IGNORECASE
)
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3}\.?|[IVXLC]{1,5}\.)\s+\S")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡0-9•\-–(])")
_BULLETS = re.compile(r"^[•●▪■‣⁃◦*·]\s*")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str | None
    organization: str | None
    publication_year: str | None
    version: str | None
    section: str | None
    page: int
    source_url: str | None
    file_name: str
    text: str
    n_tokens: int
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        parts = [self.title or self.doc_id]
        if self.organization:
            parts.append(self.organization)
        if self.publication_year:
            parts.append(str(self.publication_year))
        if self.section:
            parts.append(f"sección «{self.section}»")
        parts.append(f"pág. {self.page}")
        return ", ".join(parts)


# --------------------------------------------------------------------------
# Limpieza
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Normalización unicode y de espacios, sin alterar el contenido clínico."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")            # guion blando
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _line_signature(line: str) -> str:
    """Firma para detectar encabezados/pies repetidos (ignora números)."""
    return re.sub(r"\d+", "#", normalize_text(line).lower())


def detect_repeated_lines(pages_lines: list[list[str]], min_ratio: float = 0.5,
                          edge_lines: int = 3) -> set[str]:
    """Líneas en los bordes de página que se repiten en muchas páginas."""
    if len(pages_lines) < 3:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages_lines:
        edges = lines[:edge_lines] + lines[-edge_lines:]
        counter.update({_line_signature(l) for l in edges if l.strip()})
    threshold = max(2, int(len(pages_lines) * min_ratio))
    return {sig for sig, n in counter.items() if n >= threshold and len(sig) > 1}


def merge_broken_lines(lines: list[str]) -> list[str]:
    """Une líneas cortadas por el maquetado del PDF en párrafos/unidades."""
    merged: list[str] = []
    for raw in lines:
        line = normalize_text(raw)
        if not line:
            if merged and merged[-1] != "":
                merged.append("")          # separador de párrafo
            continue
        line = _BULLETS.sub("- ", line)
        if merged and merged[-1]:
            prev = merged[-1]
            if prev.endswith("-") and line[:1].islower():           # hiper-\ntensión
                merged[-1] = prev[:-1] + line
                continue
            starts_item = line.startswith("- ") or _NUMBERED_HEADING.match(line)
            if not prev.endswith((".", ":", ";", "?", "!", "|")) and not starts_item \
                    and not prev.startswith("|"):
                merged[-1] = f"{prev} {line}"
                continue
        merged.append(line)
    return [m for m in merged if m]


def split_units(paragraphs: list[str]) -> list[str]:
    """Divide párrafos en oraciones; las filas de tabla quedan enteras."""
    units: list[str] = []
    for paragraph in paragraphs:
        if paragraph.startswith("|"):
            units.append(paragraph)
            continue
        units.extend(s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip())
    return units


# --------------------------------------------------------------------------
# Extracción con estructura
# --------------------------------------------------------------------------
@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    y0: float


def _page_lines(page: fitz.Page, table_bboxes: list[fitz.Rect]) -> list[_Line]:
    out: list[_Line] = []
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            bbox = fitz.Rect(line["bbox"])
            if any(bbox.intersects(tb) for tb in table_bboxes):
                continue
            text = "".join(s["text"] for s in spans)
            size = max(s.get("size", 0) for s in spans)
            bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()
                       for s in spans)
            out.append(_Line(text=text, size=round(size, 1), bold=bold, y0=bbox.y0))
    return out


def _page_tables(page: fitz.Page) -> list[tuple[fitz.Rect, list[str]]]:
    """Tablas como filas markdown (cada fila es una unidad indivisible)."""
    try:
        tables = page.find_tables()
    except Exception:  # noqa: BLE001 - versiones sin soporte o PDF raro
        return []
    result = []
    for table in tables.tables:
        try:
            rows = table.extract()
        except Exception:  # noqa: BLE001
            continue
        md_rows = []
        for row in rows:
            cells = [normalize_text(str(c or "")).replace("\n", " ") for c in row]
            cells = [c for c in cells if c]            # celdas combinadas/vacías
            if cells:
                md_rows.append("| " + " | ".join(cells) + " |")
        if len(md_rows) >= 2 and table_text_ok(md_rows):
            result.append((fitz.Rect(table.bbox), md_rows))
        # Si la extracción de la tabla sale degradada, no se excluye su zona y
        # el contenido se toma del texto corrido de la página (más fiable).
    return result


def table_text_ok(rows: list[str]) -> bool:
    """Detecta tablas mal extraídas (letras intercaladas o palabras pegadas).

    Ej.: "C toe rf áa cl ie ca o" o "Aumentobruscodelapresiónarterial".
    """
    words = [w for w in re.findall(r"[^\W\d_]+", " ".join(rows))]
    if len(words) < 4:
        return True
    tiny = sum(len(w) <= 2 for w in words) / len(words)
    glued = sum(len(w) >= 22 for w in words) / len(words)
    return tiny < 0.35 and glued < 0.05


def _is_heading(line: _Line, body_size: float) -> bool:
    """Heurística tipográfica de títulos.

    Se exige más de una palabra (las siglas en negrita de las listas de
    abreviaturas, como "IAM" o "PICO", no son títulos) salvo que la fuente
    sea claramente mayor que la del cuerpo.
    """
    text = line.text.strip()
    if not (4 <= len(text) <= 140) or not re.search(r"[A-Za-zÁÉÍÓÚáéíóúñÑ]{3}", text):
        return False
    if text.startswith(("©", "|", "(")) or re.match(r"^[a-záéíóúñ]", text):
        return False                                   # continuación de oración
    numbered = bool(_NUMBERED_HEADING.match(text))
    if text.endswith((",", ";")) or (text.endswith(".") and not numbered):
        return False
    n_words = len(text.split())
    much_bigger = line.size >= body_size * 1.3
    bigger = line.size >= body_size * 1.15 and n_words >= 2
    upper = text.isupper() and n_words >= 2 and len(text) <= 90
    bold = line.bold and n_words >= 2 and len(text) <= 100
    return much_bigger or bigger or upper or bold or (numbered and line.bold)


_INTERNAL_BOOKMARK = re.compile(r"^_(Hlk|Toc|Ref|GoBack)\w*$")


def _toc_by_page(doc: fitz.Document) -> dict[int, str]:
    """Título del TOC con el que *comienza* cada página (1-based).

    Solo se registran las páginas donde empieza una entrada; en el resto se
    conserva la sección detectada por tipografía. Se descartan los
    marcadores internos de Word (``_Hlk...``, ``_Toc...``).
    """
    mapping: dict[int, str] = {}
    for _lvl, title, page in doc.get_toc(simple=True) or []:
        title = normalize_text(title)
        if page > 0 and title and not _INTERNAL_BOOKMARK.match(title):
            mapping[page] = title          # la última entrada de la página prevalece
    return mapping


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def pack_units(units: list[str], counter: TokenCounter, chunk_size: int,
               overlap: int) -> list[tuple[str, int]]:
    """Empaqueta unidades en chunks de ~chunk_size tokens con solapamiento."""
    sized: list[tuple[str, int]] = []
    for unit in units:
        n = counter.count(unit)
        if n <= chunk_size:
            sized.append((unit, n))
            continue
        # Unidad demasiado larga: se divide por palabras.
        words, buf = unit.split(), []
        for word in words:
            buf.append(word)
            if counter.count(" ".join(buf)) >= chunk_size:
                piece = " ".join(buf)
                sized.append((piece, counter.count(piece)))
                buf = []
        if buf:
            piece = " ".join(buf)
            sized.append((piece, counter.count(piece)))

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(sized):
        total, end = 0, start
        while end < len(sized) and (total + sized[end][1] <= chunk_size or end == start):
            total += sized[end][1]
            end += 1
        text = " ".join(u for u, _ in sized[start:end]) if not sized[start][0].startswith("|") \
            else "\n".join(u for u, _ in sized[start:end])
        chunks.append((text, counter.count(text)))
        if end >= len(sized):
            break
        # Retroceder unidades hasta cubrir ~overlap tokens (sin repetir todo).
        back, new_start = 0, end
        while new_start - 1 > start and back + sized[new_start - 1][1] <= overlap:
            new_start -= 1
            back += sized[new_start][1]
        start = new_start if new_start > start else end
    return chunks


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():   # 2024.0 -> 2024
        text = text[:-2]
    return text or None


def process_pdf(pdf_path: Path, meta: dict[str, Any], counter: TokenCounter,
                chunk_size: int = 200, overlap: int = 60,
                header_footer_min_ratio: float = 0.5,
                min_chunk_chars: int = 40, extract_tables: bool = True,
                exclude_sections: list[str] | None = None,
                min_group_tokens: int = 50) -> list[Chunk]:
    """Procesa un PDF y devuelve la lista de chunks con metadatos."""
    doc_id = _clean(meta.get("doc_id")) or re.sub(r"\W+", "_", pdf_path.stem)
    chunks: list[Chunk] = []
    excluded = [re.compile(p, re.IGNORECASE) for p in exclude_sections or []]
    n_excluded = 0

    with fitz.open(pdf_path) as doc:
        toc_map = _toc_by_page(doc)
        pages: list[tuple[list[_Line], list[tuple[fitz.Rect, list[str]]]]] = []
        heights: list[float] = []
        sizes: Counter[float] = Counter()
        for page in doc:
            tables = _page_tables(page) if extract_tables else []
            lines = _page_lines(page, [bbox for bbox, _ in tables])
            for ln in lines:
                sizes[ln.size] += len(ln.text)
            pages.append((lines, tables))
            heights.append(page.rect.height or 842.0)

        body_size = sizes.most_common(1)[0][0] if sizes else 10.0
        repeated = detect_repeated_lines(
            [[l.text for l in lines] for lines, _ in pages], header_footer_min_ratio
        )

        current_section: str | None = None
        for page_no, (lines, tables) in enumerate(pages, start=1):
            current_section = toc_map.get(page_no, current_section)
            height = heights[page_no - 1]
            # Elementos de la página en orden vertical: (y, tipo, contenido)
            elements: list[tuple[float, str, Any]] = [(l.y0, "line", l) for l in lines]
            elements += [(bbox.y0, "table", rows) for bbox, rows in tables]
            elements.sort(key=lambda e: e[0])

            page_chunks_units: list[tuple[str | None, list[str]]] = []
            buffer_lines: list[str] = []

            def flush(section: str | None) -> None:
                if buffer_lines:
                    units = split_units(merge_broken_lines(buffer_lines))
                    if units:
                        page_chunks_units.append((section, units))
                    buffer_lines.clear()

            prev_heading = False
            for _y, kind, content in elements:
                if kind == "table":
                    flush(current_section)
                    page_chunks_units.append((current_section, content))
                    prev_heading = False
                    continue
                line: _Line = content
                # Encabezados/pies y números de página solo se buscan en los bordes:
                # un "140" aislado dentro de una tabla es un valor clínico.
                at_edge = line.y0 < height * 0.1 or line.y0 > height * 0.88
                if at_edge and (_line_signature(line.text) in repeated
                                or _PAGE_NUMBER.match(line.text)):
                    continue
                if _is_heading(line, body_size):
                    flush(current_section)
                    title = normalize_text(line.text)
                    # Título en varias líneas consecutivas: se une en uno solo.
                    current_section = (f"{current_section} {title}" if prev_heading and current_section
                                       else title)[:160]
                    prev_heading = True
                    # El título también queda en el texto: aporta términos a la
                    # búsqueda y conserva rótulos de tablas ("En consultorio").
                    buffer_lines.append(title)
                    continue
                prev_heading = False
                buffer_lines.append(line.text)
            flush(current_section)

            groups = _merge_small_groups(page_chunks_units, counter, min_group_tokens)
            for section, units in groups:
                if section and any(p.search(section) for p in excluded):
                    n_excluded += 1
                    continue
                packed = pack_units(units, counter, chunk_size, overlap)
                # Un último fragmento muy corto se une al anterior de la misma página.
                if len(packed) >= 2 and packed[-1][1] < min_group_tokens // 2 \
                        and packed[-2][1] + packed[-1][1] <= int(chunk_size * 1.25):
                    sep = "\n" if units[0].startswith("|") else " "
                    merged = packed[-2][0] + sep + packed[-1][0]
                    packed[-2:] = [(merged, counter.count(merged))]
                for text, n_tok in packed:
                    if len(text) < min_chunk_chars:
                        continue
                    idx = len(chunks)
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}_p{page_no}_c{idx}",
                        doc_id=doc_id,
                        title=_clean(meta.get("title")),
                        organization=_clean(meta.get("organization")),
                        publication_year=_clean(meta.get("publication_year")),
                        version=_clean(meta.get("version")),
                        section=section,
                        page=page_no,
                        source_url=_clean(meta.get("source_url")),
                        file_name=pdf_path.name,
                        text=text,
                        n_tokens=n_tok,
                        chunk_index=idx,
                        metadata={"tokenizer": counter.name},
                    ))
    logger.info("%s: %d chunks (%d grupos excluidos por sección)", doc_id, len(chunks), n_excluded)
    return chunks


def _merge_small_groups(groups: list[tuple[str | None, list[str]]], counter: TokenCounter,
                        min_tokens: int) -> list[tuple[str | None, list[str]]]:
    """Une grupos de texto muy cortos de una misma página con el siguiente.

    Evita fragmentos de pocas palabras cuando la tipografía genera muchos
    cortes de sección. Las tablas no se mezclan con texto corrido.
    """
    out: list[tuple[str | None, list[str]]] = []
    for section, units in groups:
        is_table = units[0].startswith("|")
        if out:
            prev_section, prev_units = out[-1]
            prev_table = prev_units[0].startswith("|")
            small = counter.count(" ".join(prev_units)) < min_tokens
            if small and prev_table == is_table:
                out[-1] = (prev_section or section, prev_units + units)
                continue
        out.append((section, list(units)))
    return out


def process_corpus(catalog_csv: str | Path, output_jsonl: str | Path,
                   chunk_size: int = 200, overlap: int = 60,
                   tokenizer_name: str | None = None,
                   header_footer_min_ratio: float = 0.5,
                   min_chunk_chars: int = 40,
                   exclude_sections: list[str] | None = None) -> pd.DataFrame:
    """Procesa todos los PDF válidos del catálogo y exporta chunks.jsonl."""
    catalog = pd.read_csv(catalog_csv, dtype=str)
    counter = get_token_counter(tokenizer_name)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[Chunk] = []
    for row in catalog.to_dict(orient="records"):
        status = row.get("status")
        if status not in PROCESSABLE_STATUS:
            logger.warning("Se omite %s (status=%s)", row.get("file_name"), status)
            continue
        if status == "missing_metadata":
            logger.warning("%s sin doc_id: se usa el nombre de archivo.", row["file_name"])
        try:
            all_chunks.extend(process_pdf(
                Path(row["file_path"]), row, counter, chunk_size, overlap,
                header_footer_min_ratio, min_chunk_chars,
                exclude_sections=exclude_sections,
            ))
        except Exception:  # noqa: BLE001
            logger.exception("Error procesando %s", row.get("file_name"))

    write_chunks(all_chunks, output_jsonl)
    stats = chunk_stats(all_chunks)
    logger.info("Total chunks: %d -> %s", len(all_chunks), output_jsonl)
    return stats


def write_chunks(chunks: Iterable[Chunk], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: str | Path) -> list[Chunk]:
    return list(iter_chunks(path))


def iter_chunks(path: str | Path) -> Iterator[Chunk]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Chunk(**json.loads(line))


def chunk_stats(chunks: list[Chunk]) -> pd.DataFrame:
    """Resumen por documento para validar el chunking."""
    if not chunks:
        return pd.DataFrame(columns=["doc_id", "chunks", "pages", "tokens_mean",
                                     "tokens_min", "tokens_max", "sin_seccion"])
    df = pd.DataFrame([asdict(c) for c in chunks])
    return df.groupby("doc_id").agg(
        chunks=("chunk_id", "count"),
        pages=("page", "nunique"),
        tokens_mean=("n_tokens", lambda s: round(statistics.mean(s), 1)),
        tokens_min=("n_tokens", "min"),
        tokens_max=("n_tokens", "max"),
        sin_seccion=("section", lambda s: int(s.isna().sum())),
    ).reset_index()


def main() -> None:
    """CLI: python -m app.ingestion.pdf_processor"""
    from app.config import load_config, resolve_path
    from app.utils.logging_config import configure_logging

    cfg = load_config()
    configure_logging(cfg)
    p = cfg["processing"]
    stats = process_corpus(
        catalog_csv=resolve_path(cfg["corpus"]["catalog_path"]),
        output_jsonl=resolve_path(cfg["corpus"]["chunks_path"]),
        chunk_size=p["chunk_size_tokens"],
        overlap=p["chunk_overlap_tokens"],
        tokenizer_name=p.get("tokenizer_name"),
        header_footer_min_ratio=p.get("header_footer_min_ratio", 0.5),
        min_chunk_chars=p.get("min_chunk_chars", 40),
        exclude_sections=p.get("exclude_sections"),
    )
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
