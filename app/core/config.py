import os
import re
import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

ENV_ALIASES = {
    "GATEWAY_BASE_URL": ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL", "GLM_BASE_URL"),
    "GATEWAY_KEY_1": ("GATEWAY_API_KEY", "OPENAI_KEY_1", "OPENAI_API_KEY"),
    "GATEWAY_HEALTH_URL": ("OPENAI_HEALTH_URL",),
    "OPENAI_BASE_URL": ("GATEWAY_BASE_URL",),
    "ANTHROPIC_BASE_URL": ("GATEWAY_BASE_URL",),
    "DEEPSEEK_BASE_URL": ("GATEWAY_BASE_URL",),
    "GLM_BASE_URL": ("GATEWAY_BASE_URL", "OTHERS_BASE_URL"),
    "OPENAI_KEY_1": ("OPENAI_API_KEY",),
    "ANTHROPIC_KEY_1": ("ANTHROPIC_API_KEY",),
    "DEEPSEEK_KEY_1": ("DEEPSEEK_API_KEY",),
    "GLM_KEY_1": ("OTHERS_KEY_1", "GLM_API_KEY", "OTHERS_API_KEY"),
    "OTHERS_KEY_1": ("GLM_KEY_1", "GLM_API_KEY", "OTHERS_API_KEY"),
}


def apply_env_aliases():
    for canonical, aliases in ENV_ALIASES.items():
        if os.environ.get(canonical):
            continue
        for alias in aliases:
            if os.environ.get(alias):
                os.environ[canonical] = os.environ[alias]
                break


class ModelPricing(BaseModel):
    prompt: float
    completion: float


class ModelConfig(BaseModel):
    name: str
    routing_keys: list[str]
    pricing: ModelPricing | None = None


class ApiKeyConfig(BaseModel):
    key: str
    rpm: int = 60
    tpm: int | None = None
    weight: int = 1


class ProviderConfig(BaseModel):
    base_url: str
    health_url: str = ""
    timeout: int = 0
    max_retries: int = 3
    key_strategy: str = "round_robin"
    keys: list[ApiKeyConfig]
    models: list[ModelConfig]


class AppConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    server: dict = {"host": "0.0.0.0", "port": 8000}
    auth: dict = {"api_key": ""}
    semantic_routing: dict = {}
    keyword_rules: list = []


def load_config(path: str = "config/providers.yaml") -> AppConfig:
    load_dotenv()
    apply_env_aliases()
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    raw = config_path.read_text(encoding="utf-8")
    raw = re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), ""), raw)
    data = yaml.safe_load(raw)
    return AppConfig(**data)
