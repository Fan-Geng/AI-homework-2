from fastapi import Request

from app.core.exceptions import AuthError, RateLimitError
from app.core.request_context import RequestContext, parse_allowed_models


INTERNAL_USER_ID_HEADER = "X-User-Id"
INTERNAL_PROFILE_ID_HEADER = "X-Routing-Profile-Id"
INTERNAL_ALLOWED_MODELS_HEADER = "X-Allowed-Models"
INTERNAL_REQUEST_ID_HEADER = "X-Request-Id"
INTERNAL_SUBJECT_ID_HEADER = "X-Subject-Id"


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        raise AuthError()
    token = header[7:].strip()
    if not token:
        raise AuthError()
    return token


def _state_value(request: Request, key: str, default=None):
    return getattr(request.app.state, key, default)


async def resolve_request_context(request: Request) -> RequestContext:
    cached = getattr(request.state, "request_context", None)
    if cached is not None:
        return cached

    token = _bearer_token(request)
    internal_auth_key = _state_value(request, "internal_auth_key") or _state_value(request, "auth_key", "")

    if token == internal_auth_key:
        user_id = request.headers.get(INTERNAL_USER_ID_HEADER) or None
        routing_profile_id = request.headers.get(INTERNAL_PROFILE_ID_HEADER) or None
        if routing_profile_id is None and user_id:
            profile = await request.app.state.db.get_routing_profile_for_user(user_id)
            if profile:
                routing_profile_id = str(profile["id"])
        context = RequestContext(
            subject_type="internal",
            subject_id=request.headers.get(INTERNAL_SUBJECT_ID_HEADER, "internal"),
            user_id=user_id,
            routing_profile_id=routing_profile_id,
            request_id=request.headers.get(INTERNAL_REQUEST_ID_HEADER) or None,
            allowed_models=parse_allowed_models(request.headers.get(INTERNAL_ALLOWED_MODELS_HEADER)),
            subject_note="internal_service",
        )
        request.state.request_context = context
        return context

    if not _state_value(request, "enable_standalone_keys", True):
        raise AuthError()

    key = await request.app.state.db.get_api_key(token)
    if not key:
        raise AuthError()

    limiter = request.app.state.api_key_limiter
    if not limiter.check_and_increment(key["id"], key["rate_limit_rpm"]):
        raise RateLimitError(key["rate_limit_rpm"])

    profile = await request.app.state.db.get_routing_profile_for_api_key(key["id"])
    routing_profile_id = str(profile["id"]) if profile else f"api_key:{key['id']}"

    context = RequestContext(
        subject_type="api_key",
        subject_id=str(key["id"]),
        api_key_id=key["id"],
        routing_profile_id=routing_profile_id,
        allowed_models=parse_allowed_models(key["allowed_models"]),
        subject_note=key["note"],
    )
    request.state.request_context = context
    return context
