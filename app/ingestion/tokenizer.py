"""Conteo de tokens para el chunking.

Se usa el tokenizer de Hugging Face del modelo de embeddings, para que el
tamaño del chunk coincida con lo que realmente verá el encoder. Si no está
disponible (sin internet o sin ``transformers``), se usa un fallback por
expresiones regulares y se deja registrado en el log.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger("healthrag.ingestion.tokenizer")

_REGEX_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class RegexTokenCounter:
    """Aproximación: palabras y signos de puntuación.

    Para español, los tokenizers subword generan ~1.3 tokens por palabra; el
    factor permite acercar la estimación al tokenizer real.
    """

    def __init__(self, subword_factor: float = 1.3) -> None:
        self.name = f"regex(x{subword_factor})"
        self.factor = subword_factor

    def count(self, text: str) -> int:
        return int(round(len(_REGEX_TOKEN.findall(text)) * self.factor))


class HFTokenCounter:
    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # import diferido

        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def count(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False)["input_ids"])


@lru_cache(maxsize=4)
def get_token_counter(model_name: str | None) -> TokenCounter:
    if model_name:
        try:
            counter = HFTokenCounter(model_name)
            logger.info("Tokenizer cargado: %s", model_name)
            return counter
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "No se pudo cargar el tokenizer '%s' (%s). Se usa fallback regex.",
                model_name, type(exc).__name__,
            )
    return RegexTokenCounter()
