import json
from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import JSONResponse

from app.schemas.admin import CascadeRuleCreate, CascadeRuleUpdate
from ._auth import require_admin

router = APIRouter(tags=["admin / cascade"])


@router.get("/cascade/rules")
async def list_cascade_rules(
    request: Request,
    routing_profile_id: int | None = Query(None),
    _=Depends(require_admin),
):
    rows = await request.app.state.db.list_cascade_rules(
        enabled_only=False,
        routing_profile_id=routing_profile_id,
        include_global=True,
    )
    return [dict(r) for r in rows]


@router.get("/cascade/rules/{rule_id}")
async def get_cascade_rule(rule_id: int, request: Request, _=Depends(require_admin)):
    row = await request.app.state.db._fetchone(
        "SELECT * FROM cascade_rules WHERE id=?", (rule_id,)
    )
    if not row:
        return JSONResponse(status_code=404, content={"error": f"Cascade rule {rule_id} not found"})
    return dict(row)


@router.post("/cascade/rules")
async def create_cascade_rule(body: CascadeRuleCreate, request: Request, _=Depends(require_admin)):
    try:
        await request.app.state.db.create_cascade_rule(
            name=body.name,
            source_routing_key=body.source_routing_key,
            escalation_chain=json.dumps(body.escalation_chain),
            min_content_length=body.min_content_length,
            check_refusal=body.check_refusal,
            refusal_patterns=json.dumps(body.refusal_patterns),
            check_finish_reason=body.check_finish_reason,
            local_judge_model=body.local_judge_model,
            local_judge_threshold=body.local_judge_threshold,
            local_judge_system_prompt=body.local_judge_system_prompt,
            custom_validator_url=body.custom_validator_url,
            custom_validator_timeout=body.custom_validator_timeout,
            max_escalations=body.max_escalations,
            cost_threshold=body.cost_threshold,
            priority=body.priority,
            note=body.note,
            routing_profile_id=body.routing_profile_id,
        )
        await request.app.state.cascade.reload()
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.put("/cascade/rules/{rule_id}")
async def update_cascade_rule(rule_id: int, body: CascadeRuleUpdate, request: Request, _=Depends(require_admin)):
    updates = {}
    for k, v in body.model_dump().items():
        if v is not None:
            if k == "escalation_chain":
                updates[k] = json.dumps(v)
            elif k == "refusal_patterns":
                updates[k] = json.dumps(v)
            else:
                updates[k] = v
    if not updates:
        return {"ok": True}
    await request.app.state.db.update_cascade_rule(rule_id, **updates)
    await request.app.state.cascade.reload()
    return {"ok": True}


@router.delete("/cascade/rules/{rule_id}")
async def delete_cascade_rule(rule_id: int, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_cascade_rule(rule_id)
    await request.app.state.cascade.reload()
    return {"ok": True}


@router.get("/cascade/stats")
async def query_cascade_stats(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    model: str = None,
    limit: int = 100,
    offset: int = 0,
    _=Depends(require_admin),
):
    rows = await request.app.state.db.query_cascade_stats(
        from_date=from_date, to_date=to_date,
        model=model, limit=limit, offset=offset,
    )
    return [dict(r) for r in rows]


@router.get("/cascade/stats/summary")
async def cascade_stats_summary(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    _=Depends(require_admin),
):
    row = await request.app.state.db.cascade_stats_summary(
        from_date=from_date, to_date=to_date,
    )
    if not row:
        return {}
    result = dict(row)
    total = result.get("total_requests", 0)
    escalations = result.get("total_escalations", 0)
    result["escalation_rate"] = round(escalations / total, 4) if total > 0 else 0.0
    return result


@router.get("/cascade/stats/daily")
async def cascade_stats_daily(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    _=Depends(require_admin),
):
    rows = await request.app.state.db.cascade_stats_daily(
        from_date=from_date, to_date=to_date,
    )
    return [dict(r) for r in rows]
