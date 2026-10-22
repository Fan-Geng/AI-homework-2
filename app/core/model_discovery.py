import logging
import os
import csv
from pathlib import Path

import httpx

from app.core.db import Database
from app.core.config import apply_env_aliases

logger = logging.getLogger("model_router.discovery")

MODEL_PREFIXES = {
    "gateway": (),
    "openai": ("gpt", "o", "chatgpt", "codex"),
    "anthropic": ("claude",),
    "deepseek": ("deepseek",),
    "glm": ("glm", "qwen", "kimi", "minimax"),
}

ENV_KEY_BY_PROVIDER = {
    "gateway": "GATEWAY_KEY_1",
    "openai": "OPENAI_KEY_1",
    "anthropic": "ANTHROPIC_KEY_1",
    "deepseek": "DEEPSEEK_KEY_1",
    "glm": "GLM_KEY_1",
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRICE_TABLE = ROOT / "experiments" / "chatbot-arena" / "outputs" / "model_price_table.csv"


def normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _candidate_headers(provider_name: str, api_key: str) -> list[dict]:
    candidates = []
    bearer = {"Authorization": f"Bearer {api_key}"}
    x_api_key = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    if provider_name == "anthropic":
        candidates.extend([x_api_key, bearer])
    else:
        candidates.extend([bearer, x_api_key])
    return candidates


def _extract_model_ids(payload) -> list[str]:
    if isinstance(payload, dict):
        items = payload.get("data")
        if items is None:
            items = payload.get("models")
        if items is None:
            items = payload.get("items")
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    model_ids = []
    for item in items or []:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
        else:
            model_id = None
        if model_id:
            model_ids.append(str(model_id).strip())

    seen = set()
    result = []
    for model_id in model_ids:
        if model_id and model_id not in seen:
            seen.add(model_id)
            result.append(model_id)
    return result


def _filter_models(provider_name: str, model_ids: list[str]) -> list[str]:
    prefixes = MODEL_PREFIXES.get(provider_name, ())
    if not prefixes:
        return model_ids
    filtered = [mid for mid in model_ids if mid.lower().startswith(prefixes)]
    return filtered or model_ids


def _load_price_table(path: Path = DEFAULT_PRICE_TABLE) -> dict[str, tuple[float, float]]:
    if not path.exists():
        return {}
    prices: dict[str, tuple[float, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model = (row.get("model") or "").strip()
            if not model:
                continue
            try:
                prompt = float(row.get("input_usd_per_1m") or 0)
                completion = float(row.get("output_usd_per_1m") or 0)
            except (TypeError, ValueError):
                continue
            prices[model] = (prompt, completion)
    return prices


async def fetch_available_models(
    provider_name: str,
    base_url: str,
    api_key: str,
    timeout: int = 20,
) -> list[str]:
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise ValueError(f"Provider '{provider_name}' has no base_url configured")
    if not api_key:
        raise ValueError(f"Provider '{provider_name}' has no API key configured")

    last_error = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        for headers in _candidate_headers(provider_name, api_key):
            try:
                resp = await client.get(f"{base_url}/models", headers=headers)
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    continue
                model_ids = _extract_model_ids(resp.json())
                model_ids = _filter_models(provider_name, model_ids)
                if model_ids:
                    return model_ids
                last_error = "models endpoint returned no usable model ids"
            except Exception as exc:
                last_error = str(exc)

    raise RuntimeError(
        f"Failed to fetch models for provider '{provider_name}' from {base_url}: {last_error}"
    )


async def sync_provider_models_from_remote(
    db: Database,
    provider_name: str,
    timeout: int = 20,
) -> dict:
    apply_env_aliases()
    provider = await db.get_provider(provider_name)
    if not provider:
        raise ValueError(f"Provider '{provider_name}' not found")

    keys = await db.list_provider_keys(provider_name)
    api_key = next((row["key_value"] for row in keys if (row["key_value"] or "").strip()), "")
    if not api_key:
        api_key = os.environ.get(ENV_KEY_BY_PROVIDER.get(provider_name, ""), "")
    if not api_key:
        raise ValueError(f"Provider '{provider_name}' has no API key configured")

    model_ids = await fetch_available_models(
        provider_name=provider_name,
        base_url=provider["base_url"],
        api_key=api_key,
        timeout=timeout,
    )
    stats = await db.sync_provider_models(provider_name, model_ids)
    price_table = _load_price_table()
    priced = 0
    for model_id in model_ids:
        pricing = price_table.get(model_id)
        if not pricing:
            continue
        rows = await db.get_models_by_provider(provider_name, enabled_only=False)
        for row in rows:
            if row["model_name"] == model_id and row["routing_key"] == model_id:
                await db.update_model(row["id"], prompt_price=pricing[0], completion_price=pricing[1])
                priced += 1
                break
    return {
        "provider": provider_name,
        "base_url": provider["base_url"],
        "models": model_ids,
        "priced": priced,
        **stats,
    }


async def sync_all_providers_from_remote(db: Database, timeout: int = 20) -> list[dict]:
    results = []
    providers = await db.list_providers(enabled_only=True)
    for provider in providers:
        name = provider["name"]
        try:
            results.append(await sync_provider_models_from_remote(db, name, timeout=timeout))
        except Exception as exc:
            logger.warning("Model sync failed for provider '%s': %s", name, exc)
            results.append({
                "provider": name,
                "base_url": provider["base_url"],
                "error": str(exc),
                "models": [],
                "created": 0,
                "deleted": 0,
                "preserved_aliases": 0,
            })
    return results
