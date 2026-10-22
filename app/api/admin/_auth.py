import hmac
import os

from fastapi import Header, HTTPException, Request


def ensure_feature_enabled(request: Request, attr_name: str, label: str):
    if not getattr(request.app.state, attr_name, True):
        raise HTTPException(404, f"{label} disabled")


async def require_admin(request: Request, authorization: str = Header("")):
    ensure_feature_enabled(request, "enable_admin_api", "Admin API")
    admin_key = getattr(request.app.state, "admin_key", "") or os.environ.get("ADMIN_API_KEY", "")
    if not admin_key:
        raise HTTPException(503, "Admin API misconfigured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Unauthorized")
    if not hmac.compare_digest(authorization[7:], admin_key):
        raise HTTPException(401, "Unauthorized")
