"""Auditoría continua: un registro JSONL por consulta procesada."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(l) for l in lines if l.strip()]
