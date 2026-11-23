"""Configuración de logging a consola y archivo rotativo."""
from __future__ import annotations

import logging
import warnings
from logging.handlers import RotatingFileHandler
from typing import Any

from app.config import load_config, resolve_path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(config: dict[str, Any] | str | None = None) -> logging.Logger:
    """Configura el logger raíz del proyecto ("healthrag").

    Acepta el diccionario de configuración ya cargado o una ruta a un YAML
    (compatible con la versión de la Etapa 1, que recibía la ruta).
    Los módulos usan ``logging.getLogger("healthrag.<modulo>")`` y heredan
    estos handlers.
    """
    if config is None or isinstance(config, str):
        config = load_config(config)

    settings = config.get("logging", {})
    level = getattr(logging, str(settings.get("level", "INFO")).upper(), logging.INFO)

    log_path = resolve_path(settings.get("file", "logs/healthrag.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("healthrag")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Ruido conocido de librerías de terceros que no aporta al diagnóstico.
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    for noisy in ("httpx", "sentence_transformers", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logger
