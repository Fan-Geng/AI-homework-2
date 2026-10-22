#!/usr/bin/env python3
"""
Provider Discovery Utility

This utility helps you discover available models from your configured API providers.

Usage:
    python -m app.discover
    python -m app.discover --generate-config
"""

import asyncio
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv

from app.core.config import load_config
from app.core.model_discovery import ENV_KEY_BY_PROVIDER, fetch_available_models


class ProviderDiscovery:
    """Discover available models from configured providers."""

    def __init__(self):
        load_dotenv()
        self.results: Dict[str, Dict[str, Any]] = {}
        config = load_config()
        self.base_urls = {
            name: provider.base_url
            for name, provider in config.providers.items()
        }

    def _base_url(self, provider_name: str) -> str:
        return self.base_urls.get(provider_name, "")

    async def _test_provider_key(self, provider_name: str, key: str) -> bool:
        try:
            await fetch_available_models(provider_name, self._base_url(provider_name), key, timeout=10)
            return True
        except Exception:
            return False

    async def _fetch_provider_models(self, provider_name: str, key: str) -> List[str]:
        try:
            return await fetch_available_models(provider_name, self._base_url(provider_name), key, timeout=10)
        except Exception:
            return []

    async def test_openai_key(self, key: str) -> bool:
        return await self._test_provider_key("openai", key)

    async def test_anthropic_key(self, key: str) -> bool:
        return await self._test_provider_key("anthropic", key)

    async def test_deepseek_key(self, key: str) -> bool:
        return await self._test_provider_key("deepseek", key)

    async def test_glm_key(self, key: str) -> bool:
        return await self._test_provider_key("glm", key)

    async def fetch_openai_models(self, key: str) -> List[str]:
        return await self._fetch_provider_models("openai", key)

    async def fetch_anthropic_models(self, key: str) -> List[str]:
        return await self._fetch_provider_models("anthropic", key)

    async def fetch_deepseek_models(self, key: str) -> List[str]:
        return await self._fetch_provider_models("deepseek", key)

    async def fetch_glm_models(self, key: str) -> List[str]:
        return await self._fetch_provider_models("glm", key)

    async def _discover_provider(self, provider_name: str):
        env_key = ENV_KEY_BY_PROVIDER.get(provider_name, "")
        label = provider_name.replace("_", " ").title()
        key = os.getenv(env_key)
        if not key:
            print(f"📌 {label}: ⚠️  No API key configured")
            self.results[provider_name] = {
                "status": "not_configured",
                "key_valid": False,
                "models": [],
                "key_count": 0,
                "base_url": self._base_url(provider_name),
            }
            return

        print(f"📌 Testing {label}...", end=" ")
        if await self._test_provider_key(provider_name, key):
            print("✅")
            models = await self._fetch_provider_models(provider_name, key)
            self.results[provider_name] = {
                "status": "available",
                "key_valid": True,
                "models": models,
                "key_count": 1,
                "base_url": self._base_url(provider_name),
            }
        else:
            print("❌")
            self.results[provider_name] = {
                "status": "unavailable",
                "key_valid": False,
                "models": [],
                "key_count": 0,
                "base_url": self._base_url(provider_name),
            }

    async def discover_all(self) -> Dict[str, Dict[str, Any]]:
        """Discover all configured providers and models."""
        print("\n🔍 Discovering available providers and models...\n")
        for provider_name in self.base_urls.keys():
            await self._discover_provider(provider_name)
        return self.results

    def print_summary(self):
        """Print discovery summary."""
        print("\n" + "=" * 60)
        print("📊 Discovery Summary")
        print("=" * 60)

        available_count = sum(1 for r in self.results.values() if r["status"] == "available")
        total_models = sum(len(r["models"]) for r in self.results.values())

        print(f"\n✅ Available providers: {available_count}/{len(self.results)}")
        print(f"📦 Total models discovered: {total_models}")

        for provider, result in self.results.items():
            status_icon = "✅" if result["status"] == "available" else "❌" if result["status"] == "unavailable" else "⚠️"
            print(f"\n{status_icon} {provider.upper()}")
            print(f"   Status: {result['status']}")
            print(f"   Base URL: {result['base_url'] or '(not configured)'}")
            print(f"   API Key: {'Valid' if result['key_valid'] else 'Invalid or missing'}")
            if result["models"]:
                print(f"   Models: {len(result['models'])} available")
                for model in result["models"][:5]:
                    print(f"     - {model}")
                if len(result["models"]) > 5:
                    print(f"     ... and {len(result['models']) - 5} more")

    def generate_discovery_report(self) -> str:
        """Generate a comment-only discovery report, not a runnable config."""
        report_lines = [
            "# Discovery report only",
            "# This file is not a runnable providers.yaml",
            "# Copy values manually after reviewing model names and pricing",
        ]
        for provider, result in self.results.items():
            if result["status"] == "available" and result["models"]:
                report_lines.append(f"# {provider.upper()} - {len(result['models'])} models available")
                for model in result["models"]:
                    report_lines.append(f"#   - {model}")
        return "\n".join(report_lines)


async def main():
    print("\n" + "=" * 60)
    print("🔍 Model Router - Provider Discovery")
    print("=" * 60)

    discovery = ProviderDiscovery()
    await discovery.discover_all()
    discovery.print_summary()

    if "--generate-config" in sys.argv:
        print("\n📝 Exporting discovery report...")
        print("   This output is a comment-only report, not a runnable config file.")
        config_update = discovery.generate_discovery_report()
        print(config_update)
        with open("discovered_models.txt", "w", encoding="utf-8") as f:
            f.write(config_update)
        print("\n✅ Discovery report saved to discovered_models.txt")

    print("\n" + "=" * 60)
    print("🚀 Next Steps:")
    print("1. Ensure your provider base URLs and API keys are correct")
    print("2. Start the service: python -m app.main")
    print("3. Or sync models live: POST /admin/providers/sync-models")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
