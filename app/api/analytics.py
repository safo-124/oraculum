"""POST /analytics/visit: the web app reports a page view (admin Activity page).

Open to everyone (guests included), never fails the page: any problem
returns 204 quietly. Each visitor is rate-limited so a script cannot flood
the table. The visitor is classified from their credentials when they have
any (signed-in user, admin) and as a guest otherwise.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app.api.auth import get_actor
from app.config import Settings, get_settings
from app.services.activity import record_activity, visitor_id
from app.services.rate_limit import get_rate_limiter

router = APIRouter(prefix="/analytics", tags=["analytics"])

_VISITS_PER_MINUTE = 20


class VisitIn(BaseModel):
    path: str = Field(default="/", max_length=200)


def _resolve_actor(request: Request, settings: Settings):
    try:
        return get_actor(
            request,
            request.headers.get("X-API-Key"),
            request.cookies.get(settings.auth_cookie_name),
            settings,
        )
    except HTTPException:
        return None  # no credentials: a guest visitor


@router.post("/visit", status_code=204)
def visit(
    payload: VisitIn,
    request: Request,
    settings: Settings = Depends(get_settings),
    limiter=Depends(get_rate_limiter),
) -> Response:
    if settings.activity_tracking_enabled:
        if limiter.allow(f"visit:{visitor_id(request, settings)}", _VISITS_PER_MINUTE):
            path = payload.path if payload.path.startswith("/") else "/"
            record_activity("visit", request, _resolve_actor(request, settings), settings, path=path.split("?")[0])
    return Response(status_code=204)
