"""CLI management tool for Model Router.

Usage:
    python -m app.manage init            Seed DB from YAML
    python -m app.manage reset           Clear DB and re-seed (prompts confirm)
    python -m app.manage sync-models     Sync live model list from provider endpoints
    python -m app.manage check-health    Check provider-reported health endpoints
    python -m app.manage preview-route   Explain how a request would be routed
    python -m app.manage gen-key         Generate a user API key
    python -m app.manage list-keys       List all user API keys
    python -m app.manage list-models     List all routes
    python -m app.manage stats           Show usage statistics
"""

import asyncio
import json
import os
import secrets
import sys
from dotenv import load_dotenv
from app.core.config import apply_env_aliases, load_config

load_dotenv()
apply_env_aliases()
DB_PATH = os.environ.get("ROUTER_DB", "router.db")


async def cmd_init():
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    await db.migrate()
    count = await db._fetchone("SELECT COUNT(*) FROM providers")
    if count and count[0] > 0:
        print(
            f"Database {DB_PATH} is not empty; skipped re-seeding. "
            "Use `python -m app.manage reset` or a fresh ROUTER_DB if you want a clean bootstrap."
        )
    else:
        await db.seed_from_yaml("config/providers.yaml")
        print(f"Seeded from config/providers.yaml into {DB_PATH}")
    await db.close()


async def cmd_sync_models():
    from app.core.db import Database
    from app.core.model_discovery import sync_all_providers_from_remote, sync_provider_models_from_remote

    provider_name = sys.argv[2] if len(sys.argv) > 2 else None

    db = Database(DB_PATH)
    await db.connect()
    await db.migrate()

    if provider_name:
        result = await sync_provider_models_from_remote(db, provider_name)
        print(f"{provider_name}: {len(result['models'])} models, +{result['created']} canonical, -{result['deleted']} stale")
    else:
        results = await sync_all_providers_from_remote(db)
        for result in results:
            if result.get("error"):
                print(f"{result['provider']}: ERROR {result['error']}")
            else:
                print(f"{result['provider']}: {len(result['models'])} models, +{result['created']} canonical, -{result['deleted']} stale")

    await db.close()


async def cmd_check_health():
    from app.core.db import Database
    from app.core.provider_health import check_all_providers_remote_health, check_provider_remote_health

    provider_name = sys.argv[2] if len(sys.argv) > 2 else None

    db = Database(DB_PATH)
    await db.connect()
    await db.migrate()

    if provider_name:
        result = await check_provider_remote_health(db, provider_name)
        print(
            f"{provider_name}: healthy={result['healthy']} status={result['status']} "
            f"score={result['score']} latency_ms={result['latency_ms']} "
            f"url={result['source_url']}"
        )
        if result.get("message"):
            print(f"  message: {result['message']}")
    else:
        results = await check_all_providers_remote_health(db)
        for result in results:
            print(
                f"{result['provider']}: healthy={result['healthy']} status={result['status']} "
                f"score={result['score']} latency_ms={result['latency_ms']} "
                f"url={result['source_url']}"
            )
            if result.get("message"):
                print(f"  message: {result['message']}")

    await db.close()


async def cmd_preview_route():
    if len(sys.argv) < 4:
        print('Usage: python -m app.manage preview-route <model> "<prompt>"')
        return

    model = sys.argv[2]
    prompt = sys.argv[3]

    from app.core.db import Database
    from app.core.router import RouterEngine
    from app.core.cascade import CascadeEngine
    from app.core.embedder import SemanticEmbedder
    from app.core.route_preview import build_route_preview
    from app.schemas.chat import ChatMessage, ChatRequest

    db = Database(DB_PATH)
    await db.connect()
    await db.migrate()

    router = RouterEngine(db)
    await router.reload()

    cascade = CascadeEngine(db, router)
    await cascade.reload()

    keyword_routing_enabled = (await db.get_setting("keyword_routing_enabled", "false")) == "true"
    cascade_enabled = (await db.get_setting("cascade_enabled", "false")) == "true"
    semantic_cfg = load_config().semantic_routing or {}
    sem_enabled_db = (await db.get_setting("semantic_routing_enabled", "false")) == "true"
    embedder = SemanticEmbedder(
        model_name=semantic_cfg.get("model", "all-MiniLM-L6-v2"),
        threshold=semantic_cfg.get("threshold", 0.4),
        cache_dir=semantic_cfg.get("cache_dir", ".embed_cache"),
        categories=semantic_cfg.get("categories"),
    )
    sem_enabled = sem_enabled_db or semantic_cfg.get("enabled", False)
    if sem_enabled:
        embedder.load()
    semantic_routing_enabled = embedder.enabled and sem_enabled

    body = ChatRequest(
        model=model,
        messages=[ChatMessage(role="user", content=prompt)],
    )
    preview = await build_route_preview(
        body,
        router,
        cascade,
        keyword_routing_enabled=keyword_routing_enabled,
        semantic_routing_enabled=semantic_routing_enabled,
        embedder=embedder,
        cascade_enabled=cascade_enabled,
    )
    print(json.dumps(preview, indent=2, ensure_ascii=False))
    await db.close()


async def cmd_reset():
    ans = input("This will delete ALL data in the database. Continue? [y/N] ")
    if ans.lower() != "y":
        print("Cancelled.")
        return
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    await cmd_init()
    print(f"Database reset and re-seeded.")


async def cmd_gen_key():
    note = input("Note/customer name: ").strip() or "generated"
    key_value = "sk-" + secrets.token_hex(24)
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    await db.migrate()
    await db.create_api_key(key_value=key_value, note=note)
    print(f"API Key: {key_value}")
    print(f"Note:    {note}")
    await db.close()


async def cmd_list_keys():
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    keys = await db.list_api_keys()
    if not keys:
        print("No API keys found.")
    else:
        print(f"{'ID':<4} {'Key (suffix)':<16} {'Note':<20} {'RPM':<6} {'Enabled':<8}")
        print("-" * 60)
        for k in keys:
            print(f"{k['id']:<4} {k['key_suffix']:<16} {k['note']:<20} {k['rate_limit_rpm']:<6} {k['enabled']:<8}")
    await db.close()


async def cmd_list_models():
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    models = await db.list_models(enabled_only=False)
    if not models:
        print("No models configured.")
    else:
        print(f"{'ID':<4} {'Routing Key':<24} {'Model':<24} {'Provider':<12} {'Enabled':<8}")
        print("-" * 76)
        for m in models:
            print(f"{m['id']:<4} {m['routing_key']:<24} {m['model_name']:<24} {m['provider_name']:<12} {m['enabled']:<8}")
    await db.close()


async def cmd_stats():
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    summary = await db.usage_summary()
    if summary:
        print(f"Total requests:     {summary['total_requests']}")
        print(f"Total prompt tokens: {summary['total_prompt_tokens']}")
        print(f"Total completion t:  {summary['total_completion_tokens']}")
        print(f"Total tokens:       {summary['total_tokens']}")
        print(f"Total cost:         ${summary['total_cost']:.6f}")
        print(f"Total errors:       {summary['total_errors']}")
        print(f"Avg latency:        {summary['avg_latency_ms']:.0f}ms")
    else:
        print("No usage data.")
    await db.close()


async def cmd_list_rules():
    from app.core.db import Database
    db = Database(DB_PATH)
    await db.connect()
    rules = await db.list_keyword_rules(enabled_only=False)
    if not rules:
        print("No keyword rules configured.")
    else:
        print(f"{'ID':<4} {'Keyword':<24} {'Mode':<10} {'Target':<24} {'Pri':<5} {'Enabled':<8}")
        print("-" * 80)
        for r in rules:
            print(f"{r['id']:<4} {r['keyword']:<24} {r['match_mode']:<10} {r['target_routing_key']:<24} {r['priority']:<5} {r['enabled']:<8}")
    await db.close()


async def cmd_list_intents():
    import yaml
    from pathlib import Path
    from app.core.embedder import SemanticEmbedder

    config_path = Path("config/providers.yaml")
    if not config_path.exists():
        print("config/providers.yaml not found.")
        return

    raw = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    semantic_cfg = data.get("semantic_routing") or {}
    categories = semantic_cfg.get("categories") or {}

    if not categories:
        print("No semantic routing categories configured.")
        return

    print(f"{'Category':<20} {'Target':<24} {'Description'}")
    print("-" * 80)
    for name, cfg in categories.items():
        desc = cfg.get("description", "")
        target = cfg.get("target_routing_key", "")
        print(f"{name:<20} {target:<24} {desc}")

    threshold = semantic_cfg.get("threshold", 0.4)
    model_name = semantic_cfg.get("model", "all-MiniLM-L6-v2")
    print(f"\nModel: {model_name}")
    print(f"Threshold: {threshold}")


COMMANDS = {
    "init": cmd_init,
    "reset": cmd_reset,
    "sync-models": cmd_sync_models,
    "check-health": cmd_check_health,
    "preview-route": cmd_preview_route,
    "gen-key": cmd_gen_key,
    "list-keys": cmd_list_keys,
    "list-models": cmd_list_models,
    "list-rules": cmd_list_rules,
    "list-intents": cmd_list_intents,
    "stats": cmd_stats,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python -m app.manage <command>")
        print()
        print("Commands:")
        for name in COMMANDS:
            print(f"  {name}")
        sys.exit(1)

    asyncio.run(COMMANDS[sys.argv[1]]())


if __name__ == "__main__":
    main()
