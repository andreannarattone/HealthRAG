"""Acción final automática (opcional): notificación por webhook.

Se dispara cuando el sistema se abstiene o el verificador descarta
afirmaciones, para que un responsable revise el caso o amplíe el corpus.
Compatible con webhooks de Slack/Discord (campo "text"/"content").
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger("healthrag.actions")


class WebhookNotifier:
    def __init__(self, actions_cfg: dict[str, Any]) -> None:
        self.enabled = bool(actions_cfg.get("webhook_enabled"))
        self.url = os.getenv(actions_cfg.get("webhook_url_env", "HEALTHRAG_WEBHOOK_URL"), "")
        if self.enabled and not self.url:
            logger.warning("webhook_enabled=true pero no hay URL configurada; se desactiva.")
            self.enabled = False

    @staticmethod
    def should_notify(response: dict[str, Any]) -> bool:
        return bool(response.get("abstained")) or any(
            "omitieron" in w for w in response.get("warnings", []))

    def notify(self, response: dict[str, Any]) -> None:
        if not self.enabled or not self.should_notify(response):
            return
        text = (f"HealthRAG [{response.get('request_id')}] "
                f"{'abstención' if response.get('abstained') else 'afirmaciones descartadas'}: "
                f"{response.get('query', '')[:200]}")
        threading.Thread(target=self._post, args=({"text": text, "content": text},),
                         daemon=True).start()

    def _post(self, payload: dict[str, Any]) -> None:
        try:
            import httpx
            httpx.post(self.url, json=payload, timeout=5).raise_for_status()
        except Exception as exc:  # noqa: BLE001 - la notificación nunca rompe la consulta
            logger.warning("No se pudo enviar el webhook: %s", exc)
