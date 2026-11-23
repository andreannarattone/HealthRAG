"""Control de acceso por API key con roles y límite de tasa en memoria."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Principal:
    user_id: str      # hash corto de la clave (nunca la clave en logs)
    role: str         # "user" | "admin"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def load_api_keys() -> dict[str, str]:
    """HEALTHRAG_API_KEYS="clave1,clave2:admin" -> {clave: rol}."""
    keys: dict[str, str] = {}
    for item in os.getenv("HEALTHRAG_API_KEYS", "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, role = item.partition(":")
        keys[key] = role or "user"
    return keys


def authenticate(api_key: str | None, require: bool,
                 keys: dict[str, str] | None = None) -> Principal | None:
    """Devuelve el Principal o None si la clave es inválida."""
    keys = load_api_keys() if keys is None else keys
    if not require:
        return Principal(_hash(api_key) if api_key else "anonymous",
                         keys.get(api_key or "", "user"))
    if not api_key or api_key not in keys:
        return None
    return Principal(_hash(api_key), keys[api_key])


class RateLimiter:
    """Ventana deslizante de 60 s por usuario."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, user_id: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            window = self._hits[user_id]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)
            return True
