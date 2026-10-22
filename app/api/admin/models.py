from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from app.schemas.admin import ModelCreate, ModelUpdate
from ._auth import require_admin

router = APIRouter(tags=["admin / models"])


@router.get("/models")
async def list_models(request: Request, _=Depends(require_admin)):
    rows = await request.app.state.db.list_models(enabled_only=False)
    return [dict(r) for r in rows]


@router.post("/models")
async def create_model(body: ModelCreate, request: Request, _=Depends(require_admin)):
    try:
        await request.app.state.db.create_model(**body.model_dump())
        await request.app.state.router.reload()
        return {"ok": True}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.put("/models/{model_id}")
async def update_model(model_id: int, body: ModelUpdate, request: Request, _=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"ok": True}
    await request.app.state.db.update_model(model_id, **updates)
    await request.app.state.router.reload()
    return {"ok": True}


@router.delete("/models/{model_id}")
async def delete_model(model_id: int, request: Request, _=Depends(require_admin)):
    await request.app.state.db.delete_model(model_id)
    await request.app.state.router.reload()
    return {"ok": True}
