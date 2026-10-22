from fastapi import APIRouter, Request, Depends

from ._auth import require_admin

router = APIRouter(tags=["admin / status"])


@router.get("/status")
async def status(request: Request, _=Depends(require_admin)):
    providers = {}
    for name, provider in request.app.state.router.providers.items():
        key_count = len(provider.key_pool.keys)
        latest_remote = await request.app.state.db.list_provider_health_checks(name, limit=1)
        providers[name] = {
            "key_count": key_count,
            "models": [rk for rk, (pn, _) in request.app.state.router.model_map.items() if pn == name],
            "health": provider.health.snapshot(),
            "remote_health": dict(latest_remote[0]) if latest_remote else None,
        }
    
    cascade_rules = await request.app.state.db.list_cascade_rules(enabled_only=True)
    routing_profiles = []
    if hasattr(request.app.state.db, "list_routing_profiles"):
        routing_profiles = await request.app.state.db.list_routing_profiles(enabled_only=True)
    
    return {
        "ok": True,
        "runtime_mode": getattr(request.app.state, "runtime_mode", "standalone"),
        "features": {
            "admin_api": getattr(request.app.state, "enable_admin_api", True),
            "standalone_keys": getattr(request.app.state, "enable_standalone_keys", True),
            "usage_api": getattr(request.app.state, "enable_usage_api", True),
        },
        "providers": providers,
        "total_models": len(request.app.state.router.model_map),
        "model_availability": {
            "cooldown_s": getattr(
                getattr(request.app.state.router, "model_availability", None),
                "cooldown_s",
                None,
            ),
            "unavailable": (
                request.app.state.router.model_availability_snapshot()
                if hasattr(request.app.state.router, "model_availability_snapshot")
                else {}
            ),
        },
        "routing_profiles": {
            "enabled_count": len(routing_profiles),
        },
        "cascade": {
            "enabled": getattr(request.app.state, "cascade_enabled", False),
            "rules_count": len(cascade_rules),
        },
    }
