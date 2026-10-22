from fastapi import APIRouter, Request

from app.core.auth import resolve_request_context
from app.core.routing_policy import load_routing_policy
from app.schemas.chat import ModelList, ModelInfo

router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request):
    request_context = await resolve_request_context(request)
    routing_policy = await load_routing_policy(
        request.app.state.db,
        request_context.routing_profile_id,
    )
    allowed_models = routing_policy.effective_allowed_models(
        request_context.allowed_models,
        available_models=set(request.app.state.router.model_map.keys()),
    )
    data = request.app.state.router.list_models(allowed_models=allowed_models)
    return ModelList(data=[ModelInfo(**m) for m in data])
