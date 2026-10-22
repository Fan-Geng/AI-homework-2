from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.core.model_discovery import sync_all_providers_from_remote, sync_provider_models_from_remote
from app.core.provider_health import check_all_providers_remote_health, check_provider_remote_health
from app.schemas.admin import (
    ProviderCreate,
    ProviderUpdate,
    ProviderKeyCreate,
    ProviderModelSyncRequest,
    ProviderHealthCheckRequest,
)
from ._auth import require_admin

router = APIRouter(tags=["admin / providers"])


@router.get("/providers")
async def list_providers(request: Request, _=Depends(require_admin)):
    rows = await request.app.state.db.list_providers(enabled_only=False)
    return [dict(r) for r in rows]


@router.get("/providers/{name}")
async def get_provider(name: str, request: Request, _=Depends(require_admin)):
    p = await request.app.state.db.get_provider(name)
    if not p:
        return JSONResponse(status_code=404, content={"error": f"Provider '{name}' not found"})
    keys = await request.app.state.db.list_provider_keys(name)
    models = await request.app.state.db.get_models_by_provider(name, enabled_only=False)
    health_checks = await request.app.state.db.list_provider_health_checks(name, limit=10)
    return {
        **dict(p),
        "keys": [dict(k) for k in keys],
        "models": [dict(m) for m in models],
        "health_checks": [dict(row) for row in health_checks],
    }


@router.post("/providers")
async def create_provider(body: ProviderCreate, request: Request, _=Depends(require_admin)):
    try:
        await request.app.state.db.create_provider(**body.model_dump())
        await request.app.state.router.reload()
        return {"ok": True, "name": body.name}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.put("/providers/{name}")
async def update_provider(name: str, body: ProviderUpdate, request: Request, _=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"ok": True}
    await request.app.state.db.update_provider(name, **updates)
    await request.app.state.router.reload()
    return {"ok": True}


@router.delete("/providers/{name}")
async def delete_provider(name: str, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_provider(name)
    await request.app.state.router.reload()
    return {"ok": True}


@router.post("/providers/{name}/keys")
async def add_provider_key(name: str, body: ProviderKeyCreate, request: Request, _=Depends(require_admin)):
    try:
        await request.app.state.db.create_provider_key(name, **body.model_dump())
        await request.app.state.router.reload()
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.delete("/providers/{name}/keys/{key_id}")
async def delete_provider_key(name: str, key_id: int, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_provider_key(key_id)
    await request.app.state.router.reload()
    return {"ok": True}


@router.post("/providers/{name}/sync-models")
async def sync_provider_models(
    name: str,
    body: ProviderModelSyncRequest,
    request: Request,
    _=Depends(require_admin),
):
    try:
        result = await sync_provider_models_from_remote(
            request.app.state.db,
            name,
            timeout=body.timeout,
        )
        await request.app.state.router.reload()
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/providers/sync-models")
async def sync_all_provider_models(
    body: ProviderModelSyncRequest,
    request: Request,
    _=Depends(require_admin),
):
    results = await sync_all_providers_from_remote(
        request.app.state.db,
        timeout=body.timeout,
    )
    await request.app.state.router.reload()
    return {"ok": True, "results": results}


@router.get("/providers/{name}/health")
async def list_provider_health_checks(
    name: str,
    request: Request,
    limit: int = 10,
    _=Depends(require_admin),
):
    provider = await request.app.state.db.get_provider(name)
    if not provider:
        return JSONResponse(status_code=404, content={"error": f"Provider '{name}' not found"})
    rows = await request.app.state.db.list_provider_health_checks(name, limit=max(1, min(limit, 100)))
    return {"ok": True, "provider": name, "results": [dict(row) for row in rows]}


@router.post("/providers/{name}/check-health")
async def check_provider_health(
    name: str,
    body: ProviderHealthCheckRequest,
    request: Request,
    _=Depends(require_admin),
):
    try:
        result = await check_provider_remote_health(
            request.app.state.db,
            name,
            timeout=body.timeout,
        )
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.post("/providers/check-health")
async def check_all_provider_health(
    body: ProviderHealthCheckRequest,
    request: Request,
    _=Depends(require_admin),
):
    results = await check_all_providers_remote_health(
        request.app.state.db,
        timeout=body.timeout,
    )
    return {"ok": True, "results": results}
