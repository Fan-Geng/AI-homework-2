import json
import os
import time
from typing import Any

import httpx

from app.core.config import apply_env_aliases
from app.core.db import Database
from app.core.model_discovery import ENV_KEY_BY_PROVIDER, _candidate_headers, normalize_base_url


def candidate_health_urls(base_url: str, health_url: str = "") -> list[str]:
    urls = []
    explicit = normalize_base_url(health_url)
    if explicit:
        urls.append(explicit)

    base = normalize_base_url(base_url)
    if not base:
        return urls

    urls.extend(
        [
            f"{base}/health",
            f"{base}/status",
        ]
    )

    root = base[:-3] if base.endswith("/v1") else base
    if root != base:
        urls.extend(
            [
                f"{root}/health",
                f"{root}/status",
                f"{root}/api/health",
                f"{root}/api/status",
            ]
        )

    seen = set()
    result = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _find_first(payload: Any, keys: tuple[str, ...]):
    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for value in payload.values():
            found = _find_first(value, keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_first(item, keys)
            if found not in (None, ""):
                return found
    return None


def parse_health_payload(payload: Any) -> dict[str, Any]:
    status_value = _find_first(payload, ("status", "state"))
    healthy_value = _find_first(payload, ("healthy", "ok", "available"))
    score_value = _find_first(payload, ("score", "health_score", "provider_score"))
    latency_value = _find_first(payload, ("latency_ms", "response_time_ms", "avg_latency_ms", "latency"))
    message_value = _find_first(payload, ("message", "detail", "description", "summary", "error"))

    healthy = None
    if isinstance(healthy_value, bool):
        healthy = healthy_value
    elif isinstance(status_value, str):
        lowered = status_value.strip().lower()
        if lowered in {"ok", "healthy", "up", "available"}:
            healthy = True
        elif lowered in {"down", "error", "degraded", "unhealthy"}:
            healthy = False

    status = str(status_value).strip() if status_value not in (None, "") else "unknown"
    score = None
    if isinstance(score_value, (int, float)):
        score = float(score_value)
    latency_ms = 0.0
    if isinstance(latency_value, (int, float)):
        latency_ms = float(latency_value)

    message = ""
    if message_value not in (None, ""):
        message = str(message_value).strip()

    return {
        "healthy": healthy,
        "status": status,
        "score": score,
        "latency_ms": latency_ms,
        "message": message,
        "raw": payload,
    }


async def fetch_provider_remote_health(
    provider_name: str,
    base_url: str,
    api_key: str = "",
    timeout: int = 10,
    health_url: str = "",
) -> dict[str, Any]:
    urls = candidate_health_urls(base_url, health_url)
    if not urls:
        raise ValueError(f"Provider '{provider_name}' has no base_url or health_url configured")

    header_sets = [{}]
    if api_key:
        header_sets.extend(_candidate_headers(provider_name, api_key))

    last_error = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for url in urls:
            for headers in header_sets:
                started = time.monotonic()
                try:
                    resp = await client.get(url, headers=headers)
                    latency_ms = (time.monotonic() - started) * 1000
                    if resp.status_code != 200:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        continue
                    try:
                        payload = resp.json()
                    except json.JSONDecodeError:
                        text = resp.text.strip()
                        last_error = f"non-json provider health response: {text[:200]}"
                        continue
                    parsed = parse_health_payload(payload)
                    parsed.update(
                        {
                            "provider": provider_name,
                            "source_url": url,
                            "latency_ms": parsed["latency_ms"] or round(latency_ms, 1),
                        }
                    )
                    return parsed
                except Exception as exc:
                    last_error = str(exc)

    raise RuntimeError(
        f"Failed to fetch provider health for '{provider_name}' from {urls[0]}: {last_error}"
    )


async def check_provider_remote_health(
    db: Database,
    provider_name: str,
    timeout: int = 10,
) -> dict[str, Any]:
    apply_env_aliases()
    provider = await db.get_provider(provider_name)
    if not provider:
        raise ValueError(f"Provider '{provider_name}' not found")

    keys = await db.list_provider_keys(provider_name)
    api_key = next((row["key_value"] for row in keys if (row["key_value"] or "").strip()), "")
    if not api_key:
        api_key = os.environ.get(ENV_KEY_BY_PROVIDER.get(provider_name, ""), "").strip()

    source_url = (provider["health_url"] or "").strip() if "health_url" in provider.keys() else ""
    if not source_url:
        source_url = provider["base_url"]

    try:
        result = await fetch_provider_remote_health(
            provider_name=provider_name,
            base_url=provider["base_url"],
            api_key=api_key,
            timeout=timeout,
            health_url=(provider["health_url"] or "").strip() if "health_url" in provider.keys() else "",
        )
        await db.insert_provider_health_check(
            provider_name=provider_name,
            source_url=result["source_url"],
            status=result["status"],
            healthy=result["healthy"],
            score=result["score"],
            latency_ms=result["latency_ms"],
            message=result["message"],
            raw=result["raw"],
        )
        return result
    except Exception as exc:
        await db.insert_provider_health_check(
            provider_name=provider_name,
            source_url=source_url,
            status="error",
            healthy=False,
            score=None,
            latency_ms=0.0,
            message=str(exc),
            raw=None,
        )
        raise


async def check_all_providers_remote_health(db: Database, timeout: int = 10) -> list[dict[str, Any]]:
    results = []
    providers = await db.list_providers(enabled_only=True)
    for provider in providers:
        name = provider["name"]
        try:
            results.append(await check_provider_remote_health(db, name, timeout=timeout))
        except Exception as exc:
            results.append(
                {
                    "provider": name,
                    "source_url": (provider["health_url"] or provider["base_url"]) if "health_url" in provider.keys() else provider["base_url"],
                    "status": "error",
                    "healthy": False,
                    "score": None,
                    "latency_ms": 0.0,
                    "message": str(exc),
                    "raw": None,
                    "error": str(exc),
                }
            )
    return results
