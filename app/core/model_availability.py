from __future__ import annotations

import os
import time
from typing import Any


MODEL_UNAVAILABLE_MARKERS = (
    "model_not_found",
    "model not found",
    "model does not exist",
    "model doesn't exist",
    "model unavailable",
    "model is unavailable",
    "invalid model",
    "unknown model",
    "no available channel",
    "no available channels",
    "no channel available",
    "unavailable channel",
    "model not supported",
    "模型不存在",
    "模型未找到",
    "模型不可用",
    "无可用渠道",
    "没有可用渠道",
)


def is_model_unavailable_error(error: Any) -> bool:
    """Identify explicit model-level availability failures."""
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    if any(marker in message for marker in MODEL_UNAVAILABLE_MARKERS):
        return True
    return False


class ModelAvailabilityRegistry:
    """Keep a short in-memory cooldown for unavailable model routes."""

    def __init__(self, cooldown_s: float | None = None):
        if cooldown_s is None:
            raw = os.environ.get("ROUTER_MODEL_COOLDOWN_S", "60").strip()
            try:
                cooldown_s = float(raw)
            except ValueError:
                cooldown_s = 60.0
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._entries: dict[str, dict[str, Any]] = {}

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for routing_key in [
            key
            for key, entry in self._entries.items()
            if float(entry.get("unavailable_until", 0.0)) <= now
        ]:
            self._entries.pop(routing_key, None)

    def is_available(self, routing_key: str) -> bool:
        self._purge_expired()
        return routing_key not in self._entries

    def mark_unavailable(
        self,
        routing_key: str,
        *,
        model_name: str | None = None,
        reason: str = "",
        status_code: int | None = None,
        cooldown_s: float | None = None,
    ) -> dict[str, Any]:
        duration = self.cooldown_s if cooldown_s is None else max(0.0, float(cooldown_s))
        marked_at = time.time()
        entry = {
            "routing_key": routing_key,
            "model_name": model_name or routing_key,
            "reason": reason[:500],
            "status_code": status_code,
            "marked_at": marked_at,
            "unavailable_until": time.monotonic() + duration,
            "unavailable_until_at": marked_at + duration,
            "cooldown_s": duration,
        }
        self._entries[routing_key] = entry
        return self.snapshot(routing_key) or entry

    def mark_error(
        self,
        routing_key: str,
        *,
        model_name: str | None = None,
        error: Any,
        cooldown_s: float | None = None,
    ) -> dict[str, Any] | None:
        if not is_model_unavailable_error(error):
            return None
        return self.mark_unavailable(
            routing_key,
            model_name=model_name,
            reason=str(error),
            status_code=getattr(error, "status_code", None),
            cooldown_s=cooldown_s,
        )

    def snapshot(self, routing_key: str | None = None) -> dict[str, Any] | None:
        self._purge_expired()
        if routing_key is not None:
            entry = self._entries.get(routing_key)
            if not entry:
                return None
            result = dict(entry)
            result["remaining_s"] = round(
                max(0.0, float(entry["unavailable_until"]) - time.monotonic()),
                3,
            )
            return result
        return {key: self.snapshot(key) for key in sorted(self._entries)}

    def unavailable_models(self) -> set[str]:
        self._purge_expired()
        return set(self._entries)

    def clear(self, routing_key: str | None = None) -> None:
        if routing_key is None:
            self._entries.clear()
        else:
            self._entries.pop(routing_key, None)
