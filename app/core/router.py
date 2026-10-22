import os
import re

from app.core.config import ModelConfig, ModelPricing, ProviderConfig, ApiKeyConfig, apply_env_aliases
from app.core.db import Database
from app.core.exceptions import ModelNotFoundError
from app.core.model_availability import ModelAvailabilityRegistry
from app.providers.base import BaseProvider, KeyPool
from app.providers.openai import OpenAIProvider
from app.providers.anthropic import AnthropicProvider
from app.providers.deepseek import DeepSeekProvider


PROVIDER_MAP = {
    "gateway": OpenAIProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "deepseek": DeepSeekProvider,
    "glm": OpenAIProvider,
}

ENV_KEY_BY_PROVIDER = {
    "gateway": "GATEWAY_KEY_1",
    "openai": "OPENAI_KEY_1",
    "anthropic": "ANTHROPIC_KEY_1",
    "deepseek": "DEEPSEEK_KEY_1",
    "glm": "GLM_KEY_1",
}

DISABLED_ROUTING_KEYS = {"codex-auto-review", "codex-review"}


class RouterEngine:
    def __init__(self, db: Database):
        self.db = db
        self.providers: dict[str, BaseProvider] = {}
        self.model_map: dict[str, tuple[str, ModelConfig]] = {}
        self.model_availability = ModelAvailabilityRegistry()

    def is_model_available(self, routing_key: str) -> bool:
        return self.model_availability.is_available(routing_key)

    def model_availability_snapshot(self, routing_key: str | None = None):
        return self.model_availability.snapshot(routing_key)

    def mark_model_error(self, routing_key: str, model_name: str, error):
        marked = []
        routing_keys = {
            key
            for key, (_, config) in self.model_map.items()
            if key == routing_key or config.name == model_name
        }
        for key in routing_keys:
            entry = self.model_availability.mark_error(
                key,
                model_name=model_name,
                error=error,
            )
            if entry:
                marked.append(entry)
        return marked or None

    async def reload(self):
        apply_env_aliases()
        self.providers.clear()
        self.model_map.clear()
        retry_floor = self._provider_retry_floor()

        rows = await self.db.list_providers(enabled_only=True)
        for row in rows:
            name = row["name"]
            keys_raw = await self.db.list_provider_keys(name)
            keys = [
                ApiKeyConfig(
                    key=k["key_value"],
                    rpm=k["rpm"],
                    tpm=k["tpm"],
                    weight=k["weight"],
                )
                for k in keys_raw
                if (k["key_value"] or "").strip()
            ]
            if not keys:
                env_key_name = ENV_KEY_BY_PROVIDER.get(name, "")
                env_key_value = os.environ.get(env_key_name, "").strip()
                if env_key_value:
                    keys = [ApiKeyConfig(key=env_key_value)]
            if not keys:
                continue
            timeout = row["timeout"]
            # Default to no upstream completion timeout; set ROUTER_PROVIDER_TIMEOUT
            # to a positive number when an environment needs a hard ceiling.
            timeout_override = os.environ.get("ROUTER_PROVIDER_TIMEOUT", "0").strip()
            if timeout_override:
                try:
                    timeout = int(timeout_override)
                except ValueError:
                    pass
            pconf = ProviderConfig(
                base_url=row["base_url"],
                timeout=timeout,
                max_retries=max(int(row["max_retries"] or 0), retry_floor),
                key_strategy=row["key_strategy"],
                keys=keys,
                models=[],
            )
            cls = PROVIDER_MAP.get(name)
            if not cls:
                continue

            key_pool = KeyPool(pconf.keys, pconf.key_strategy)
            provider = cls(pconf, key_pool)
            provider.provider_name = name
            self.providers[name] = provider

            models_raw = await self.db.get_models_by_provider(name, enabled_only=True)
            for mr in models_raw:
                if mr["routing_key"] in DISABLED_ROUTING_KEYS or mr["model_name"] in DISABLED_ROUTING_KEYS:
                    continue
                mc = ModelConfig(
                    name=mr["model_name"],
                    routing_keys=[mr["routing_key"]],
                    pricing=ModelPricing(prompt=mr["prompt_price"], completion=mr["completion_price"]),
                )
                self.model_map[mr["routing_key"]] = (name, mc)

    def _provider_retry_floor(self) -> int:
        raw = os.environ.get("ROUTER_MIN_PROVIDER_RETRIES", "3").strip()
        try:
            return max(0, int(raw))
        except ValueError:
            return 3

    async def route(self, model: str) -> tuple[BaseProvider, ModelConfig]:
        if model in DISABLED_ROUTING_KEYS:
            available = [key for key in self.model_map.keys() if key not in DISABLED_ROUTING_KEYS]
            raise ModelNotFoundError(model, available)
        entry = self.model_map.get(model)
        if not entry:
            available = [key for key in self.model_map.keys() if key not in DISABLED_ROUTING_KEYS]
            raise ModelNotFoundError(model, available)
        provider_name, model_config = entry
        return self.providers[provider_name], model_config

    def _candidate_cost(self, model_config: ModelConfig) -> float:
        pricing = model_config.pricing
        if not pricing:
            return 0.0
        return float(pricing.prompt or 0.0) + float(pricing.completion or 0.0)

    def _candidate_latency(self, provider_name: str) -> float:
        provider = self.providers.get(provider_name)
        if not provider or not getattr(provider, "health", None):
            return 0.0
        snapshot = provider.health.snapshot()
        return float(snapshot.get("avg_latency_ms") or 0.0)

    def _candidate_quality(self, provider_name: str, model_config: ModelConfig) -> float:
        provider = self.providers.get(provider_name)
        success_rate = 0.0
        if provider and getattr(provider, "health", None):
            snapshot = provider.health.snapshot()
            success_rate = float(snapshot.get("success_rate") or 0.0)
        return self._candidate_cost(model_config) + (success_rate / 1000.0)

    def _rank_model_candidates(
        self,
        candidates: list[tuple[str, str, ModelConfig]],
        *,
        preferred_providers: list[str] | None = None,
        cost_bias: float = 0.0,
        latency_bias: float = 0.0,
        quality_bias: float = 0.0,
    ) -> list[tuple[str, str, ModelConfig]]:
        provider_rank = {
            provider_name: idx
            for idx, provider_name in enumerate(preferred_providers or [])
        }

        costs = [self._candidate_cost(model_config) for _, _, model_config in candidates]
        latencies = [self._candidate_latency(provider_name) for _, provider_name, _ in candidates]
        qualities = [self._candidate_quality(provider_name, model_config) for _, provider_name, model_config in candidates]

        def normalize(values: list[float], value: float) -> float:
            low = min(values) if values else 0.0
            high = max(values) if values else 0.0
            if high <= low:
                return 0.0
            return (value - low) / (high - low)

        def sort_key(item: tuple[str, str, ModelConfig]):
            routing_key, provider_name, model_config = item
            cost = normalize(costs, self._candidate_cost(model_config))
            latency = normalize(latencies, self._candidate_latency(provider_name))
            quality = normalize(qualities, self._candidate_quality(provider_name, model_config))
            bias_score = (cost_bias * cost) + (latency_bias * latency) - (quality_bias * quality)
            return (
                provider_rank.get(provider_name, len(provider_rank)),
                round(bias_score, 8),
                provider_name,
                routing_key,
            )

        return sorted(candidates, key=sort_key)

    def resolve_model_target(
        self,
        requested_model: str,
        *,
        default_model: str | None = None,
        preferred_providers: list[str] | None = None,
        cost_bias: float = 0.0,
        latency_bias: float = 0.0,
        quality_bias: float = 0.0,
    ) -> str:
        if requested_model in DISABLED_ROUTING_KEYS:
            available = [key for key in self.model_map.keys() if key not in DISABLED_ROUTING_KEYS]
            raise ModelNotFoundError(requested_model, available)
        if requested_model in self.model_map:
            return requested_model

        if requested_model in {"default", "auto", "profile-default"} and default_model:
            if default_model in self.model_map:
                return default_model

        candidates = []
        for routing_key, (provider_name, model_config) in self.model_map.items():
            if routing_key in DISABLED_ROUTING_KEYS or model_config.name in DISABLED_ROUTING_KEYS:
                continue
            if model_config.name == requested_model:
                candidates.append((routing_key, provider_name, model_config))

        if candidates:
            candidates = self._rank_model_candidates(
                candidates,
                preferred_providers=preferred_providers,
                cost_bias=cost_bias,
                latency_bias=latency_bias,
                quality_bias=quality_bias,
            )
            return candidates[0][0]

        available = [key for key in self.model_map.keys() if key not in DISABLED_ROUTING_KEYS]
        raise ModelNotFoundError(requested_model, available)

    def list_models(self, allowed_models: set[str] | None = None) -> list[dict]:
        seen = set()
        result = []
        for key, (provider_name, _) in self.model_map.items():
            if key in DISABLED_ROUTING_KEYS:
                continue
            if allowed_models is not None and key not in allowed_models:
                continue
            if key not in seen:
                seen.add(key)
                result.append({
                    "id": key,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider_name,
                })
        return result

    def _conversation_text(self, messages: list) -> str:
        parts = []
        for msg in messages:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "")
            if role in ("user", "assistant"):
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if content:
                    parts.append(content)
        return "\n".join(parts)

    async def inspect_keyword(
        self,
        messages: list,
        routing_profile_id: str | None = None,
        only_available: bool = False,
    ) -> dict | None:
        rules = await self.db.list_keyword_rules(
            enabled_only=True,
            routing_profile_id=routing_profile_id if routing_profile_id is not None else "__global__",
            include_global=True,
        )
        if not rules:
            return None

        text = self._conversation_text(messages)
        if not text:
            return None

        for rule in rules:
            rule = dict(rule)
            keyword = rule["keyword"]
            match_mode = rule["match_mode"]
            try:
                matched = False
                if match_mode == "regex":
                    matched = bool(re.search(keyword, text))
                else:
                    matched = keyword.lower() in text.lower()
                if matched:
                    if only_available and rule["target_routing_key"] not in self.model_map:
                        continue
                    return {
                        "keyword": keyword,
                        "match_mode": match_mode,
                        "target_routing_key": rule["target_routing_key"],
                        "priority": rule.get("priority", 0),
                        "note": rule.get("note", ""),
                    }
            except re.error:
                continue
        return None

    async def match_keyword(self, messages: list, routing_profile_id: str | None = None) -> str | None:
        match = await self.inspect_keyword(
            messages,
            routing_profile_id=routing_profile_id,
            only_available=True,
        )
        return match["target_routing_key"] if match else None

    async def inspect_semantic(self, messages: list, embedder) -> dict | None:
        if not embedder or not embedder.enabled:
            return None

        text = self._conversation_text(messages)
        if not text:
            return None

        result = embedder.classify(text)
        if result is None:
            return None
        category, score, routing_key = result
        if routing_key not in self.model_map:
            return None
        return {
            "category": category,
            "score": score,
            "target_routing_key": routing_key,
        }

    async def match_semantic(self, messages: list, embedder) -> tuple[str, str] | None:
        match = await self.inspect_semantic(messages, embedder)
        if not match:
            return None
        return (match["target_routing_key"], match["category"])
