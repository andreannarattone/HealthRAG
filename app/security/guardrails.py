"""Filtros de entrada, saneamiento de evidencia y validación de salida."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.utils.text import strip_accents

# --- Patrones -----------------------------------------------------------------
_INJECTION = [
    r"ignor[ae]\w*\s+(\w+\s+){0,3}(instrucci|indicaci|regla|prompt)",
    r"ignore\s+(all\s+|the\s+)?(previous|above|prior)\s+instructions",
    r"olvid[ae]\w*\s+(\w+\s+){0,3}(instrucci|regla)",
    r"(revel|mostr|muestr|imprim)\w*\s+(\w+\s+){0,3}(system\s*prompt|prompt\s+del\s+sistema|instrucciones\s+del\s+sistema)",
    r"\bsystem\s*prompt\b",
    r"\bjailbreak\b",
    # Acotado a "como si fueras": "el fármaco actúa como un..." es texto clínico normal.
    r"\bactu[aá]\s+como\s+si\s+(fueras|sos)\b",
    r"\bfing[ei]\s+(ser|que\s+sos)\b",
    r"modo\s+(desarrollador|developer|dan)\b",
    r"</?\s*(evidencia|system|assistant)\s*>",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION]

_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    # Acepta "DNI 30.123.456", "DNI: 30123456", "mi DNI es 30.123.456", "documento nro 30123456".
    "dni": re.compile(r"\b(dni|documento)\b\s*(:|es|n[°ºo]\.?|nro\.?|n[uú]mero)?\s*"
                      r"\d{1,2}\.?\d{3}\.?\d{3}\b", re.IGNORECASE),
    "cuil": re.compile(r"\b\d{2}-\d{8}-\d\b"),
    "telefono": re.compile(r"(\+?54\s?9?\s?)?\b(11|2\d{2,3}|3\d{2,3})[\s-]?\d{3,4}[\s-]?\d{4}\b"),
    "historia_clinica": re.compile(r"historia\s+cl[ií]nica\s*(n[°ºo]\.?|nro\.?|numero|número)\s*\d+",
                                   re.IGNORECASE),
}

_INDIVIDUAL_CASE = re.compile(
    r"\b(mi\s+paciente|mi\s+(madre|padre|hij[oa]|espos[oa]|abuel[oa])|tengo\s+\d{1,3}\s+a[nñ]os|"
    r"me\s+recetaron|(debo|deberia|debería)\s+tomar|que\s+(le\s+)?(doy|receto|indico)|"
    r"cuanto\s+(le\s+)?(doy|tomo)|que\s+(hago|tomo)|"
    r"tengo\s+(la\s+presion\s+(en\s+)?)?\d{2,3}\s*/\s*\d{2,3})\b",
    re.IGNORECASE,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class GuardResult:
    allowed: bool
    query: str
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_injection(text: str) -> bool:
    plain = strip_accents(text)
    return any(p.search(plain) or p.search(text) for p in _INJECTION_RE)


def detect_pii(text: str) -> list[str]:
    return [name for name, pattern in _PII.items() if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Reemplaza los datos personales detectados por [tipo] (p. ej. para la auditoría)."""
    for name, pattern in _PII.items():
        text = pattern.sub(f"[{name}]", text)
    return text


def validate_query(query: Any, security_cfg: dict[str, Any]) -> GuardResult:
    """Valida y normaliza la consulta antes de que llegue a los agentes."""
    if not isinstance(query, str):
        return GuardResult(False, "", ["La consulta debe ser texto."])
    q = _CONTROL.sub(" ", unicodedata.normalize("NFKC", query))
    q = re.sub(r"\s+", " ", q).strip()
    result = GuardResult(True, q)

    min_c = int(security_cfg.get("min_query_chars", 5))
    max_c = int(security_cfg.get("max_query_chars", 600))
    if len(q) < min_c:
        result.allowed = False
        result.reasons.append(f"La consulta es demasiado corta (mínimo {min_c} caracteres).")
    if len(q) > max_c:
        result.allowed = False
        result.reasons.append(f"La consulta supera el máximo de {max_c} caracteres.")

    if detect_injection(q):
        result.allowed = False
        result.flags.append("prompt_injection")
        result.reasons.append("La consulta contiene instrucciones para modificar el sistema.")

    pii = detect_pii(q)
    if pii:
        result.flags.append("pii:" + ",".join(pii))
        if security_cfg.get("block_pii", True):
            result.allowed = False
            result.reasons.append("La consulta parece incluir datos personales "
                                  f"({', '.join(pii)}). No ingreses datos de pacientes.")

    if _INDIVIDUAL_CASE.search(strip_accents(q)):
        result.flags.append("individual_case")
        result.warnings.append("La consulta parece referirse a un caso individual. HealthRAG "
                               "solo describe lo que indican las guías; no da indicaciones "
                               "para pacientes concretos.")
    return result


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sanitize_evidence(text: str) -> tuple[str, bool]:
    """Quita de un fragmento las oraciones con instrucciones al modelo.

    Los documentos se tratan como datos: si un PDF contuviera texto del tipo
    "ignorá las instrucciones anteriores", no debe llegar al LLM como orden.
    """
    sentences = _SENTENCE.split(text)
    kept = [s for s in sentences if not detect_injection(s)]
    flagged = len(kept) != len(sentences)
    return (" ".join(kept).strip() if flagged else text), flagged


_CITATION = re.compile(r"\[(E\d+)\]")


def validate_output(draft: dict[str, Any], valid_eids: set[str]) -> tuple[dict[str, Any], list[str]]:
    """Comprueba la estructura de la respuesta y elimina citas inexistentes."""
    issues: list[str] = []
    answer = str(draft.get("answer") or "")
    for eid in set(_CITATION.findall(answer)) - valid_eids:
        answer = answer.replace(f"[{eid}]", "")
        issues.append(f"Cita inexistente eliminada: {eid}")

    claims = []
    for claim in draft.get("claims") or []:
        if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
            continue
        cites = [c.strip("[] ") for c in claim.get("citations") or [] if isinstance(c, str)]
        good = [c for c in cites if c in valid_eids]
        if len(good) != len(cites):
            issues.append(f"Citas inválidas en afirmación: {set(cites) - set(good)}")
        claims.append({**claim, "text": str(claim["text"]).strip(), "citations": good})

    abstain = bool(draft.get("abstain"))
    if not abstain and not claims and not _CITATION.search(answer):
        issues.append("Respuesta sin citas: se trata como abstención.")
        abstain = True
    if detect_pii(answer):
        issues.append("La respuesta contenía posibles datos personales.")
    return {**draft, "answer": answer.strip(), "claims": claims, "abstain": abstain}, issues
