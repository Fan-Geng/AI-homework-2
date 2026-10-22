"""Benchmark and smoke-test utility for Model Router.

Usage examples:
    python -m app.benchmark --admin-key "$ADMIN_API_KEY"
    python -m app.benchmark --api-key "$USER_KEY" --model gpt-5.4-mini
    python -m app.benchmark --admin-key "$ADMIN_API_KEY" --requests 100 --concurrency 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.benchmark_workload import (
    DEFAULT_BENCHMARK_SEED,
    DEFAULT_PROMPT,
    build_balanced_prompt_plan,
    format_category_counts,
    load_prompt_categories,
)
from app.core.routing_policy import DEFAULT_MODEL_ALIASES


@dataclass
class BenchResult:
    ok: bool
    status_code: int
    latency_ms: float
    request_index: int
    prompt_category: str
    model: str | None = None
    actual_model: str | None = None
    total_tokens: int | None = None
    error: str | None = None
    cascade_final_model: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Model Router")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Router base URL")
    parser.add_argument("--admin-key", default="", help="Admin API key used to create a temporary user key")
    parser.add_argument("--api-key", default="", help="Existing user API key for benchmark requests")
    parser.add_argument("--model", default="", help="Routing key to benchmark; defaults to auto recommendation")
    parser.add_argument("--allowed-models", default="*", help="Allowed models for an auto-created user key")
    parser.add_argument("--rate-limit-rpm", type=int, default=600, help="RPM for an auto-created user key")
    parser.add_argument("--requests", type=int, default=20, help="Total benchmark requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent benchmark requests")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup requests before measurement")
    parser.add_argument("--timeout", type=float, default=0.0, help="Per-request timeout in seconds; use 0 to disable")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text used for benchmark requests")
    parser.add_argument("--prompt-file", default="", help="Optional JSON or text prompt file")
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED, help="Seed for balanced random prompt scheduling")
    parser.add_argument("--max-tokens", type=int, default=120, help="max_tokens for the benchmark request")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for the benchmark")
    parser.add_argument("--skip-admin-status", action="store_true", help="Skip the admin status check")
    parser.add_argument("--skip-sync-models", action="store_true", help="Skip live provider model sync before testing")
    parser.add_argument("--create-only", action="store_true", help="Create a user key if needed, print it, then exit")
    parser.add_argument("--results-output", default="", help="Optional path to save per-request JSONL results")
    return parser


def auth_headers(token: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def create_user_key(
    client: httpx.AsyncClient,
    base_url: str,
    admin_key: str,
    allowed_models: str,
    rate_limit_rpm: int,
) -> str:
    response = await client.post(
        f"{base_url}/admin/api-keys",
        headers=auth_headers(admin_key),
        json={
            "note": "benchmark",
            "rate_limit_rpm": rate_limit_rpm,
            "allowed_models": allowed_models,
        },
    )
    response.raise_for_status()
    data = response.json()
    key_value = data.get("key_value")
    if not key_value:
        raise RuntimeError("Admin API did not return a user API key")
    return key_value


async def fetch_models(client: httpx.AsyncClient, base_url: str, api_key: str) -> list[dict[str, Any]]:
    response = await client.get(f"{base_url}/v1/models", headers=auth_headers(api_key))
    response.raise_for_status()
    data = response.json()
    return data.get("data", [])


async def fetch_admin_status(client: httpx.AsyncClient, base_url: str, admin_key: str) -> dict[str, Any]:
    response = await client.get(f"{base_url}/admin/status", headers=auth_headers(admin_key))
    response.raise_for_status()
    return response.json()


async def check_provider_health(client: httpx.AsyncClient, base_url: str, admin_key: str, timeout: int = 10) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/admin/providers/check-health",
        headers=auth_headers(admin_key),
        json={"timeout": timeout},
    )
    response.raise_for_status()
    return response.json()


async def sync_models(client: httpx.AsyncClient, base_url: str, admin_key: str, timeout: int = 20) -> dict[str, Any]:
    response = await client.post(
        f"{base_url}/admin/providers/sync-models",
        headers=auth_headers(admin_key),
        json={"timeout": timeout},
    )
    response.raise_for_status()
    return response.json()


async def run_single_request(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    request_index: int,
    prompt_category: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
) -> BenchResult:
    payload = {
        "model": model,
        "stream": stream,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    try:
        if stream:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=auth_headers(api_key),
                json=payload,
            ) as response:
                status_code = response.status_code
                body_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line:
                        body_lines.append(line)
                latency_ms = (time.perf_counter() - started) * 1000
                if status_code != 200:
                    text = "\n".join(body_lines)
                    return BenchResult(
                        False,
                        status_code,
                        latency_ms,
                        request_index=request_index,
                        prompt_category=prompt_category,
                        error=text[:300],
                    )

                parsed = parse_stream_payload(body_lines)
                return BenchResult(
                    ok=True,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    request_index=request_index,
                    prompt_category=prompt_category,
                    model=parsed.get("model"),
                    actual_model=parsed.get("actual_model"),
                    total_tokens=parsed.get("usage", {}).get("total_tokens"),
                    cascade_final_model=((parsed.get("cascade") or {}).get("final_model")),
                )

        response = await client.post(
            f"{base_url}/v1/chat/completions",
            headers=auth_headers(api_key),
            json=payload,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            data = response.json()
        except json.JSONDecodeError:
            text = response.text[:500]
            return BenchResult(
                False,
                response.status_code,
                latency_ms,
                request_index=request_index,
                prompt_category=prompt_category,
                error=f"non-json router response: {text}",
            )
        if response.status_code != 200:
            return BenchResult(
                False,
                response.status_code,
                latency_ms,
                request_index=request_index,
                prompt_category=prompt_category,
                error=json.dumps(data)[:300],
            )
        return BenchResult(
            ok=True,
            status_code=response.status_code,
            latency_ms=latency_ms,
            request_index=request_index,
            prompt_category=prompt_category,
            model=data.get("model"),
            actual_model=data.get("actual_model"),
            total_tokens=(data.get("usage") or {}).get("total_tokens"),
            cascade_final_model=((data.get("cascade") or {}).get("final_model")),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return BenchResult(False, 0, latency_ms, request_index=request_index, prompt_category=prompt_category, error=str(exc))


def parse_stream_payload(lines: list[str]) -> dict[str, Any]:
    latest_payload: dict[str, Any] = {}
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:]
        if raw == "[DONE]":
            continue
        try:
            latest_payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
    return latest_payload


async def run_batch(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    prompt_plan,
    max_tokens: int,
    stream: bool,
    concurrency: int,
) -> list[BenchResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[BenchResult] = []

    async def worker(selection) -> None:
        async with semaphore:
            result = await run_single_request(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                request_index=selection.request_index,
                prompt_category=selection.category,
                prompt=selection.prompt,
                max_tokens=max_tokens,
                stream=stream,
            )
            results.append(result)

    await asyncio.gather(*(worker(selection) for selection in prompt_plan))
    return sorted(results, key=lambda item: item.request_index)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def print_summary(
    model: str,
    requested: int,
    concurrency: int,
    total_runtime_s: float,
    prompt_plan,
    seed: int,
    results: list[BenchResult],
) -> None:
    successes = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]
    latencies = sorted(r.latency_ms for r in results)
    success_latencies = sorted(r.latency_ms for r in successes)
    tokens = [r.total_tokens for r in successes if r.total_tokens is not None]
    actual_models = sorted({r.actual_model for r in successes if r.actual_model})
    cascade_models = sorted({r.cascade_final_model for r in successes if r.cascade_final_model})
    category_counts = format_category_counts(prompt_plan)

    print("\n=== Benchmark Summary ===")
    print(f"Model requested:     {model}")
    print(f"Requests:            {requested}")
    print(f"Concurrency:         {concurrency}")
    print(f"Prompt seed:         {seed}")
    print(f"Total runtime:       {total_runtime_s:.2f}s")
    print(f"Throughput:          {(requested / total_runtime_s) if total_runtime_s else 0.0:.2f} req/s")
    print(f"Success rate:        {(len(successes) / requested * 100) if requested else 0.0:.1f}%")
    print(f"Failures:            {len(failures)}")
    print(f"Prompt categories:   {json.dumps(category_counts, ensure_ascii=False)}")
    print(f"Latency p50:         {percentile(latencies, 0.50):.1f} ms")
    print(f"Latency p95:         {percentile(latencies, 0.95):.1f} ms")
    print(f"Latency p99:         {percentile(latencies, 0.99):.1f} ms")
    if success_latencies:
        print(f"Latency avg:         {statistics.mean(success_latencies):.1f} ms")
    if tokens:
        print(f"Avg tokens/response: {statistics.mean(tokens):.1f}")
    if actual_models:
        print(f"Actual routed models:{', '.join(actual_models)}")
    if cascade_models:
        print(f"Cascade final models:{', '.join(cascade_models)}")

    if failures:
        print("\nSample failures:")
        for failure in failures[:5]:
            label = failure.error or "unknown error"
            print(
                f"  - idx={failure.request_index} category={failure.prompt_category} "
                f"status={failure.status_code} latency={failure.latency_ms:.1f}ms error={label}"
            )


def save_results(path: str, results: list[BenchResult]) -> None:
    if not path.strip():
        return
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def print_provider_health(status: dict[str, Any], title: str) -> None:
    print(title)
    providers = status.get("providers", {})
    if not providers:
        print("  no providers loaded")
        return
    for name in sorted(providers):
        provider = providers[name]
        health = provider.get("health", {})
        models = ", ".join(provider.get("models", []))
        print(
            f"  {name}: keys={provider.get('key_count', 0)} "
            f"ok={health.get('successes', 0)}/{health.get('total_requests', 0)} "
            f"fail={health.get('failures', 0)} "
            f"circuit_open={health.get('circuit_open', False)} "
            f"avg_latency_ms={health.get('avg_latency_ms', 0.0)}"
        )
        if models:
            print(f"    models: {models}")
        if health.get("last_error"):
            print(f"    last_error: {health['last_error']}")


def print_remote_provider_health(payload: dict[str, Any], title: str) -> None:
    print(title)
    results = payload.get("results", [])
    if not results:
        print("  no provider health results")
        return
    for row in results:
        print(
            f"  {row.get('provider')}: healthy={row.get('healthy')} "
            f"status={row.get('status')} score={row.get('score')} "
            f"latency_ms={row.get('latency_ms')}"
        )
        if row.get("message"):
            print(f"    message: {row['message']}")


async def main_async(args: argparse.Namespace) -> int:
    prompt_categories = load_prompt_categories(prompt=args.prompt, prompt_file=args.prompt_file)
    timeout = None if args.timeout <= 0 else httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.admin_key and not args.skip_sync_models:
            sync_result = await sync_models(client, args.base_url, args.admin_key)
            print("Model sync:")
            print(json.dumps(sync_result, indent=2))

        benchmark_key = args.api_key
        if not benchmark_key:
            benchmark_key = await create_user_key(
                client=client,
                base_url=args.base_url,
                admin_key=args.admin_key,
                allowed_models=args.allowed_models,
                rate_limit_rpm=args.rate_limit_rpm,
            )
            print(f"Created benchmark user key: {benchmark_key}")

        if args.create_only:
            return 0

        if args.admin_key and not args.skip_admin_status:
            remote_health = await check_provider_health(client, args.base_url, args.admin_key)
            print_remote_provider_health(remote_health, "Provider-reported health before benchmark:")
            status = await fetch_admin_status(client, args.base_url, args.admin_key)
            print_provider_health(status, "Admin status before benchmark:")

        models = await fetch_models(client, args.base_url, benchmark_key)
        if not models:
            print("No routable models found. Check your provider keys and seeded config.", file=sys.stderr)
            return 1

        model = args.model or "auto"
        model_ids = {entry["id"] for entry in models}
        if model not in model_ids and model not in DEFAULT_MODEL_ALIASES:
            print(f"Requested model '{model}' is not exposed by /v1/models.", file=sys.stderr)
            print("Use --model auto to benchmark recommendation routing.", file=sys.stderr)
            print("Available models:", ", ".join(sorted(model_ids)), file=sys.stderr)
            return 1

        print("Available models:", ", ".join(sorted(model_ids)))
        print(f"Benchmark model: {model}")
        print(f"Warmup requests: {args.warmup}")
        print(f"Prompt categories loaded: {len(prompt_categories)}")

        if args.warmup > 0:
            warmup_plan = build_balanced_prompt_plan(prompt_categories, args.warmup, seed=args.seed - 1)
            warmup_results = await run_batch(
                client=client,
                base_url=args.base_url,
                api_key=benchmark_key,
                model=model,
                prompt_plan=warmup_plan,
                max_tokens=args.max_tokens,
                stream=args.stream,
                concurrency=min(args.concurrency, max(1, args.warmup)),
            )
            warmup_failures = [r for r in warmup_results if not r.ok]
            print(f"Warmup complete: {len(warmup_results) - len(warmup_failures)}/{len(warmup_results)} succeeded")
            if warmup_failures:
                for failure in warmup_failures[:3]:
                    print(
                        f"  warmup failure: idx={failure.request_index} "
                        f"category={failure.prompt_category} status={failure.status_code} error={failure.error}"
                    )

        prompt_plan = build_balanced_prompt_plan(prompt_categories, args.requests, seed=args.seed)
        started = time.perf_counter()
        results = await run_batch(
            client=client,
            base_url=args.base_url,
            api_key=benchmark_key,
            model=model,
            prompt_plan=prompt_plan,
            max_tokens=args.max_tokens,
            stream=args.stream,
            concurrency=args.concurrency,
        )
        total_runtime_s = time.perf_counter() - started
        save_results(args.results_output, results)
        print_summary(model, args.requests, args.concurrency, total_runtime_s, prompt_plan, args.seed, results)

        if args.admin_key and not args.skip_admin_status:
            remote_health = await check_provider_health(client, args.base_url, args.admin_key)
            print_remote_provider_health(remote_health, "\nProvider-reported health after benchmark:")
            status = await fetch_admin_status(client, args.base_url, args.admin_key)
            print_provider_health(status, "\nAdmin status after benchmark:")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(main_async(args)))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
