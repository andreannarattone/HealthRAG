"""Caché LRU con expiración para respuestas (optimización de costo/latencia)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, max_items: int = 512, ttl_seconds: float = 3600) -> None:
        self.max_items, self.ttl = max_items, ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def make_key(**parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            item = self._data.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl:
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"items": len(self._data), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
