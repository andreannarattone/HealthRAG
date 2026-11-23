from pathlib import Path

import pandas as pd

from app.ingestion.pdf_catalog import build_corpus_catalog
from app.ingestion.pdf_processor import (detect_repeated_lines, merge_broken_lines,
                                         pack_units, read_chunks)
from app.ingestion.tokenizer import RegexTokenCounter
from tests.sample_corpus import DOCS, make_sample_pdf


def test_catalog_marks_missing_metadata_and_duplicates(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    spec = DOCS["TEST_HTA"]
    make_sample_pdf(raw / "a.pdf", spec)
    (raw / "b.pdf").write_bytes((raw / "a.pdf").read_bytes())   # copia exacta
    (raw / "roto.pdf").write_text("no soy un pdf")
    cat = build_corpus_catalog(raw, tmp_path / "cat.csv")
    status = dict(zip(cat["file_name"], cat["status"]))
    assert status["a.pdf"] == "duplicate_review"
    assert status["b.pdf"] == "duplicate_review"
    assert status["roto.pdf"] == "error"


def test_catalog_merges_metadata_without_overwriting_notes(corpus_dir):
    cat = pd.read_csv(corpus_dir / "catalog.csv", dtype=str)
    assert set(cat["doc_id"]) == {"TEST_HTA", "TEST_IAM"}
    assert (cat["status"] == "ok").all()
    assert cat["publication_year"].tolist() == ["2024", "2023"]


def test_chunks_keep_traceability(corpus_dir):
    chunks = read_chunks(corpus_dir / "chunks.jsonl")
    assert chunks, "no se generaron chunks"
    for c in chunks:
        assert c.doc_id in DOCS
        assert c.page >= 1 and c.chunk_id.startswith(f"{c.doc_id}_p{c.page}_")
        assert c.section, "toda página del corpus de prueba tiene sección"
        assert c.source_url
    sal = next(c for c in chunks if "5 gramos" in c.text)
    assert sal.page == 3 and sal.section.startswith("3.")
    # Encabezado repetido (título en cada página) eliminado.
    assert not any(c.text.startswith(DOCS["TEST_HTA"]["title"]) for c in chunks)


def test_pack_units_respects_size_and_overlap():
    counter = RegexTokenCounter(subword_factor=1.0)
    units = [f"Oración número {i} con varias palabras de relleno clínico." for i in range(40)]
    chunks = pack_units(units, counter, chunk_size=50, overlap=15)
    assert len(chunks) > 1
    assert all(n <= 50 for _, n in chunks)
    # Hay solapamiento: la última oración de un chunk reaparece en el siguiente.
    first_last = chunks[0][0].split(". ")[-1]
    assert first_last.rstrip(".") in chunks[1][0]


def test_pack_units_splits_very_long_unit():
    counter = RegexTokenCounter(subword_factor=1.0)
    long_unit = " ".join(["palabra"] * 300)
    chunks = pack_units([long_unit], counter, chunk_size=100, overlap=20)
    assert len(chunks) >= 3 and all(n <= 101 for _, n in chunks)


def test_merge_broken_lines_and_hyphenation():
    lines = ["La hiper-", "tensión arterial se define", "por valores elevados.", "- ítem de lista"]
    merged = merge_broken_lines(lines)
    assert merged[0] == "La hipertensión arterial se define por valores elevados."
    assert merged[1] == "- ítem de lista"


def test_detect_repeated_lines():
    pages = [["Guía X 2024", f"contenido {i}", f"Página {i}"] for i in range(5)]
    rep = detect_repeated_lines(pages, 0.5)
    assert "guía x #" in rep and "página #" in rep


def test_isolated_numbers_inside_page_are_kept(tmp_path):
    """Regresión: un "140" aislado en una tabla no es un número de página."""
    import pymupdf as fitz
    from app.ingestion.pdf_processor import process_pdf
    doc = fitz.open()
    for n in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 300), "Criterios diagnósticos según el método utilizado", fontsize=10)
        page.insert_text((72, 320), "En consultorio", fontsize=10)
        page.insert_text((72, 340), "140", fontsize=10)
        page.insert_text((72, 360), "90", fontsize=10)
        page.insert_text((290, 820), str(n), fontsize=8)          # número de página real
    path = tmp_path / "num.pdf"
    doc.save(path)
    chunks = process_pdf(path, {"doc_id": "NUM"}, RegexTokenCounter(), min_chunk_chars=10)
    assert all("140" in c.text and "90" in c.text for c in chunks)
    assert not any(c.text.rstrip().endswith((" 1", " 2", " 3")) for c in chunks)
