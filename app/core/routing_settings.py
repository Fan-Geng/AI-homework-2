from typing import Any


ROUTING_SETTING_KEYS = (
    "keyword_routing_enabled",
    "semantic_routing_enabled",
    "cascade_enabled",
)


async def load_effective_routing_settings(
    db,
    *,
    routing_profile_id: str | None,
    keyword_default: bool,
    semantic_default: bool,
    cascade_default: bool,
) -> dict[str, Any]:
    defaults = {
        "keyword_routing_enabled": keyword_default,
        "semantic_routing_enabled": semantic_default,
        "cascade_enabled": cascade_default,
    }
    if hasattr(db, "get_effective_routing_settings"):
        return await db.get_effective_routing_settings(routing_profile_id, defaults=defaults)
    return defaults
