from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas.admin import (
    RoutingProfileBindingUpdate,
    RoutingProfileCreate,
    RoutingProfileUpdate,
    UserRoutingProfileBindingUpdate,
)
from ._auth import require_admin

router = APIRouter(tags=["admin / routing-profiles"])


@router.get("/routing-profiles")
async def list_routing_profiles(request: Request, _=Depends(require_admin)):
    rows = await request.app.state.db.list_routing_profiles(enabled_only=False)
    return [dict(row) for row in rows]


@router.get("/routing-profiles/{profile_id}")
async def get_routing_profile(profile_id: int, request: Request, _=Depends(require_admin)):
    row = await request.app.state.db.get_routing_profile(profile_id)
    if not row:
        return JSONResponse(status_code=404, content={"error": f"Routing profile {profile_id} not found"})
    return dict(row)


@router.post("/routing-profiles")
async def create_routing_profile(body: RoutingProfileCreate, request: Request, _=Depends(require_admin)):
    try:
        profile_id = await request.app.state.db.create_routing_profile(**body.model_dump())
        return {"ok": True, "id": profile_id}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.put("/routing-profiles/{profile_id}")
async def update_routing_profile(profile_id: int, body: RoutingProfileUpdate, request: Request, _=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"ok": True}
    await request.app.state.db.update_routing_profile(profile_id, **updates)
    return {"ok": True}


@router.delete("/routing-profiles/{profile_id}")
async def delete_routing_profile(profile_id: int, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_routing_profile(profile_id)
    return {"ok": True}


@router.put("/api-keys/{key_id}/routing-profile")
async def bind_api_key_routing_profile(
    key_id: int,
    body: RoutingProfileBindingUpdate,
    request: Request,
    _=Depends(require_admin),
):
    if body.routing_profile_id is None:
        await request.app.state.db.unbind_api_key_routing_profile(key_id)
    else:
        await request.app.state.db.bind_api_key_routing_profile(key_id, body.routing_profile_id)
    return {"ok": True}


@router.put("/routing-profiles/users/{user_id}")
async def bind_user_routing_profile(
    user_id: str,
    body: UserRoutingProfileBindingUpdate,
    request: Request,
    _=Depends(require_admin),
):
    if body.routing_profile_id is None:
        await request.app.state.db.unbind_user_routing_profile(user_id)
    else:
        await request.app.state.db.bind_user_routing_profile(user_id, body.routing_profile_id)
    return {"ok": True}
