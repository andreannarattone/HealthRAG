"""Carga de configuración externa (config.yaml + variables de entorno).

Las rutas relativas se resuelven contra la raíz del proyecto, de modo que
los módulos funcionan igual desde la terminal, la API o un notebook.
"""
from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv es opcional en tiempo de ejecución
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Variables de entorno que sobrescriben claves del YAML.
ENV_OVERRIDES = {
    "HEALTHRAG_LLM_PROVIDER": ("llm", "provider"),
    "HEALTHRAG_LLM_BASE_URL": ("llm", "base_url"),
    "HEALTHRAG_ANALYST_MODEL": ("llm", "analyst_model"),
    "HEALTHRAG_VERIFIER_MODEL": ("llm", "verifier_model"),
    "HEALTHRAG_EMBEDDING_MODEL": ("retrieval", "embedding_model"),
    "HEALTHRAG_RERANKER_MODEL": ("retrieval", "reranker_model"),
    "HEALTHRAG_REQUIRE_API_KEY": ("security", "require_api_key"),
    "HEALTHRAG_LOG_LEVEL": ("logging", "level"),
}


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"null", "none", ""}:
        return None
    return value


def resolve_path(path: str | Path) -> Path:
    """Convierte una ruta del config en absoluta respecto de la raíz."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(config_path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Lee config.yaml, aplica variables de entorno y overrides explícitos."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)

    path = resolve_path(
        config_path or os.getenv("HEALTHRAG_CONFIG", "config.yaml")
    )
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {path}")

    with path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    for env_name, (section, key) in ENV_OVERRIDES.items():
        if env_name in os.environ:
            config.setdefault(section, {})[key] = _coerce(os.environ[env_name])

    if overrides:
        config = deep_update(config, overrides)
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge recursivo que no modifica el diccionario original."""
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Configuración compartida por la aplicación (cacheada)."""
    return load_config()
