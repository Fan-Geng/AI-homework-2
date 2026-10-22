import secrets

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.schemas.admin import ApiKeyCreate, ApiKeyResponse, ApiKeyListResponse
from ._auth import require_admin, ensure_feature_enabled

router = APIRouter(tags=["admin / api-keys"])


def _generate_key() -> str:
    return "sk-" + secrets.token_hex(24)


@router.get("/api-keys")
async def list_api_keys(request: Request, _=Depends(require_admin)):
    ensure_feature_enabled(request, "enable_standalone_keys", "Standalone API keys")
    rows = await request.app.state.db.list_api_keys()
    return [dict(r) for r in rows]


@router.post("/api-keys", response_model=ApiKeyResponse)
async def create_api_key(body: ApiKeyCreate, request: Request, _=Depends(require_admin)):
    ensure_feature_enabled(request, "enable_standalone_keys", "Standalone API keys")
    key_value = _generate_key()
    await request.app.state.db.create_api_key(
        key_value=key_value,
        note=body.note,
        rate_limit_rpm=body.rate_limit_rpm,
        allowed_models=body.allowed_models,
    )
    return ApiKeyResponse(
        id=0,
        key_value=key_value,
        note=body.note,
        rate_limit_rpm=body.rate_limit_rpm,
        allowed_models=body.allowed_models,
        enabled=True,
        created_at="",
    )


@router.delete("/api-keys/{key_id}")
async def delete_api_key(key_id: int, request: Request, _=Depends(require_admin)):
    ensure_feature_enabled(request, "enable_standalone_keys", "Standalone API keys")
    await request.app.state.db.delete_api_key(key_id)
    return {"ok": True}
