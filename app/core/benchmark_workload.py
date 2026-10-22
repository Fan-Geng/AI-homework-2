from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPT = "Summarize why testing improves reliability in three short bullet points."
DEFAULT_BENCHMARK_SEED = 20260728


@dataclass(frozen=True)
class PromptCategory:
    category: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class PromptSelection:
    request_index: int
    category: str
    prompt: str


def load_prompt_categories(prompt: str = "", prompt_file: str = "") -> list[PromptCategory]:
    if prompt_file.strip():
        return _load_prompt_file(prompt_file.strip())
    chosen = prompt.strip() or DEFAULT_PROMPT
    return [PromptCategory(category="default", prompts=(chosen,))]


def _load_prompt_file(path: str) -> list[PromptCategory]:
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Prompt file is empty: {path}")

    if path.endswith(".json"):
        payload = json.loads(raw)
        return _normalize_json_payload(payload)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Prompt file has no usable lines: {path}")
    return [PromptCategory(category="default", prompts=tuple(lines))]


def _normalize_json_payload(payload) -> list[PromptCategory]:
    category_map: dict[str, list[str]] = {}

    if isinstance(payload, dict):
        items = payload.get("categories")
        if not isinstance(items, list) or not items:
            raise ValueError("Prompt JSON object must contain a non-empty 'categories' list")
        iterable = items
    elif isinstance(payload, list) and payload:
        iterable = payload
    else:
        raise ValueError("Prompt JSON must be a non-empty list or object with a 'categories' list")

    for idx, item in enumerate(iterable):
        if isinstance(item, str):
            category_map.setdefault(f"item_{idx + 1}", []).append(item)
            continue

        if not isinstance(item, dict):
            raise ValueError("Prompt JSON items must be strings or objects")

        category = str(item.get("category", f"item_{idx + 1}")).strip()
        if not category:
            raise ValueError("Prompt category cannot be empty")

        prompts_raw = item.get("prompts")
        if prompts_raw is not None:
            if not isinstance(prompts_raw, list) or not prompts_raw:
                raise ValueError("Each category 'prompts' field must be a non-empty list")
            prompts = [str(prompt).strip() for prompt in prompts_raw if str(prompt).strip()]
        else:
            single_prompt = str(item.get("prompt", "")).strip()
            if not single_prompt:
                raise ValueError("Each prompt item must contain 'prompt' or 'prompts'")
            prompts = [single_prompt]

        if not prompts:
            raise ValueError(f"Prompt category '{category}' has no usable prompts")
        category_map.setdefault(category, []).extend(prompts)

    result = [
        PromptCategory(category=category, prompts=tuple(prompts))
        for category, prompts in category_map.items()
        if prompts
    ]
    if not result:
        raise ValueError("No usable prompt categories were loaded")
    return result


def build_balanced_prompt_plan(
    categories: list[PromptCategory],
    total_requests: int,
    seed: int = DEFAULT_BENCHMARK_SEED,
) -> list[PromptSelection]:
    if total_requests <= 0:
        return []
    if not categories:
        raise ValueError("At least one prompt category is required")

    rng = random.Random(seed)
    category_names = [item.category for item in categories]
    counts = {name: total_requests // len(category_names) for name in category_names}
    remainder = total_requests % len(category_names)
    if remainder:
        extra_targets = rng.sample(category_names, remainder)
        for name in extra_targets:
            counts[name] += 1

    slots: list[str] = []
    for name in category_names:
        slots.extend([name] * counts[name])
    rng.shuffle(slots)

    category_lookup = {item.category: item for item in categories}
    plan: list[PromptSelection] = []
    for index, category_name in enumerate(slots):
        prompt = rng.choice(category_lookup[category_name].prompts)
        plan.append(PromptSelection(request_index=index, category=category_name, prompt=prompt))
    return plan


def format_category_counts(plan: list[PromptSelection]) -> dict[str, int]:
    return dict(sorted(Counter(item.category for item in plan).items()))
