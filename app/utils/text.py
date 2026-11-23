"""Utilidades de texto: análisis léxico en español para BM25 y métricas."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

try:
    import snowballstemmer
    _STEMMER = snowballstemmer.stemmer("spanish")
except ImportError:  # pragma: no cover
    _STEMMER = None

SPANISH_STOPWORDS = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello
aquellos aqui asi aun cada como con contra cual cuales cualquier cuando de del desde
donde dos durante e el ella ellas ello ellos en entre era eran es esa esas ese eso esos
esta estaba estaban estan estar este esto estos fue fueron ha habia han hasta hay la las
le les lo los mas me mi mis mucho muy ni no nos o os otra otras otro otros para pero poco
por porque que quien quienes se sea segun ser si sido sin sobre son su sus tambien tan
tanto te tiene tienen todo todos tu tus un una unas uno unos y ya debe deben puede pueden
cual cuales cuanto cuanta cuantos cuantas cada usted ustedes vez veces
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[.,/][0-9]+)*")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=50_000)
def _stem(token: str) -> str:
    return _STEMMER.stemWord(token) if _STEMMER else token


def analyze(text: str, stem: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Minúsculas, sin tildes, sin stopwords y con stemming (Snowball)."""
    tokens = _TOKEN.findall(strip_accents(text.lower()))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
    return [_stem(t) for t in tokens] if stem else tokens


def content_overlap(claim: str, evidence: str) -> float:
    """Fracción de términos de contenido de ``claim`` presentes en ``evidence``.

    Se usa en el verificador offline y como métrica auxiliar de respaldo.
    Los números (dosis, umbrales, mmHg) cuentan doble: un valor clínico
    ausente en la evidencia es una señal fuerte de afirmación no respaldada.
    """
    claim_tokens = analyze(claim)
    if not claim_tokens:
        return 0.0
    evidence_tokens = set(analyze(evidence))
    weights = [2.0 if any(ch.isdigit() for ch in t) else 1.0 for t in claim_tokens]
    hit = sum(w for t, w in zip(claim_tokens, weights) if t in evidence_tokens)
    return hit / sum(weights)


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,/]\d+)*", text))
