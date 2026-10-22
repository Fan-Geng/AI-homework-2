from fastapi import APIRouter, Request, Depends, Query

from ._auth import require_admin, ensure_feature_enabled

router = APIRouter(tags=["admin / usage"])


@router.get("/usage")
async def query_usage(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    model: str = None,
    status: str = None,
    limit: int = 100,
    offset: int = 0,
    _=Depends(require_admin),
):
    ensure_feature_enabled(request, "enable_usage_api", "Usage telemetry API")
    rows = await request.app.state.db.query_usage(
        from_date=from_date, to_date=to_date,
        model=model, status=status,
        limit=limit, offset=offset,
    )
    return [dict(r) for r in rows]


@router.get("/usage/summary")
async def usage_summary(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    _=Depends(require_admin),
):
    ensure_feature_enabled(request, "enable_usage_api", "Usage telemetry API")
    row = await request.app.state.db.usage_summary(from_date=from_date, to_date=to_date)
    return dict(row) if row else {}


@router.get("/usage/daily")
async def usage_daily(
    request: Request,
    from_date: str = Query(None, alias="from"),
    to_date: str = Query(None, alias="to"),
    _=Depends(require_admin),
):
    ensure_feature_enabled(request, "enable_usage_api", "Usage telemetry API")
    rows = await request.app.state.db.usage_daily(from_date=from_date, to_date=to_date)
    return [dict(r) for r in rows]
