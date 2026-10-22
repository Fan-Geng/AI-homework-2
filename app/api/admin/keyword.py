from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import JSONResponse

from app.core.routing_settings import load_effective_routing_settings
from app.schemas.admin import KeywordRuleCreate, KeywordRuleUpdate, SettingsUpdate
from ._auth import require_admin

router = APIRouter(tags=["admin / keyword"])


@router.get("/keyword-rules")
async def list_keyword_rules(
    request: Request,
    routing_profile_id: int | None = Query(None),
    _=Depends(require_admin),
):
    rows = await request.app.state.db.list_keyword_rules(
        enabled_only=False,
        routing_profile_id=routing_profile_id,
        include_global=True,
    )
    return [dict(r) for r in rows]


@router.post("/keyword-rules")
async def create_keyword_rule(body: KeywordRuleCreate, request: Request, _=Depends(require_admin)):
    try:
        await request.app.state.db.create_keyword_rule(**body.model_dump())
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.put("/keyword-rules/{rule_id}")
async def update_keyword_rule(rule_id: int, body: KeywordRuleUpdate, request: Request, _=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"ok": True}
    await request.app.state.db.update_keyword_rule(rule_id, **updates)
    return {"ok": True}


@router.delete("/keyword-rules/{rule_id}")
async def delete_keyword_rule(rule_id: int, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_keyword_rule(rule_id)
    return {"ok": True}


@router.get("/settings")
async def get_settings(
    request: Request,
    routing_profile_id: int | None = Query(None),
    _=Depends(require_admin),
):
    embedder = getattr(request.app.state, "embedder", None)
    settings = await load_effective_routing_settings(
        request.app.state.db,
        routing_profile_id=str(routing_profile_id) if routing_profile_id is not None else None,
        keyword_default=getattr(request.app.state, "keyword_routing_enabled", False),
        semantic_default=getattr(request.app.state, "semantic_routing_enabled", False),
        cascade_default=getattr(request.app.state, "cascade_enabled", False),
    )
    return {
        "routing_profile_id": routing_profile_id,
        "keyword_routing_enabled": settings["keyword_routing_enabled"],
        "semantic_routing_enabled": settings["semantic_routing_enabled"],
        "semantic_model_loaded": embedder.enabled if embedder else False,
        "cascade_enabled": settings["cascade_enabled"],
    }


@router.put("/settings")
async def update_settings(
    body: SettingsUpdate,
    request: Request,
    routing_profile_id: int | None = Query(None),
    _=Depends(require_admin),
):
    profile_updates = {}
    if body.keyword_routing_enabled is not None:
        if routing_profile_id is None:
            value = "true" if body.keyword_routing_enabled else "false"
            await request.app.state.db.set_setting("keyword_routing_enabled", value)
            request.app.state.keyword_routing_enabled = body.keyword_routing_enabled
        else:
            profile_updates["keyword_routing_enabled"] = body.keyword_routing_enabled
    if body.semantic_routing_enabled is not None:
        if routing_profile_id is None:
            value = "true" if body.semantic_routing_enabled else "false"
            await request.app.state.db.set_setting("semantic_routing_enabled", value)
            embedder = getattr(request.app.state, "embedder", None)
            if embedder and embedder.enabled:
                request.app.state.semantic_routing_enabled = body.semantic_routing_enabled
            else:
                request.app.state.semantic_routing_enabled = False
        else:
            profile_updates["semantic_routing_enabled"] = body.semantic_routing_enabled
    if body.cascade_enabled is not None:
        if routing_profile_id is None:
            value = "true" if body.cascade_enabled else "false"
            await request.app.state.db.set_setting("cascade_enabled", value)
            request.app.state.cascade_enabled = body.cascade_enabled
        else:
            profile_updates["cascade_enabled"] = body.cascade_enabled
    if routing_profile_id is not None and profile_updates:
        await request.app.state.db.update_routing_profile_settings(routing_profile_id, profile_updates)
    return {"ok": True, "routing_profile_id": routing_profile_id}


@router.get("/intents")
async def list_intents(request: Request, _=Depends(require_admin)):
    embedder = getattr(request.app.state, "embedder", None)
    if not embedder or not embedder.enabled:
        return {"intents": [], "message": "Semantic routing not available"}
    return {"intents": embedder.get_categories()}
