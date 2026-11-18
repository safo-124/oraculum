"""Who visits the demo, and who signs up and uses it (admin Activity page).

Two events are recorded, fail-open (a failed write never breaks a request):
- "visit": the web app opened a page (POST /analytics/visit);
- "question": a chat request (/chat or /chat/stream).

Privacy by construction: no IP address and no question text is stored.
A visitor is a salted SHA-256 of the client IP, truncated to 16 hex
characters: enough to count distinct people, not reversible without the
salt, which never leaves the server.

The summary aggregates in Python over the window's rows; at demo scale
(thousands of rows) that is simpler and portable, and fast enough.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db.models import ActivityEvent, Document, User
from app.db.session import get_engine
from app.services.rate_limit import _client_ip

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _salt(configured: str | None, salt_file: str) -> str:
    if configured:
        return configured
    path = Path(salt_file)
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        path.parent.mkdir(parents=True, exist_ok=True)
        value = secrets.token_hex(32)
        path.write_text(value, encoding="utf-8")
        path.chmod(0o600)
        return value
    except OSError:
        logger.warning("Could not persist the analytics salt; using a per-process one")
        return secrets.token_hex(32)


def visitor_id(request: Request, settings: Settings) -> str:
    salt = _salt(settings.analytics_salt, str(settings.chroma_path.parent / "analytics_salt"))
    digest = hashlib.sha256(f"{salt}:{_client_ip(request)}".encode()).hexdigest()
    return digest[:16]


def actor_kind(actor) -> str:
    if actor is None:
        return "guest"
    if actor.mode == "session":
        return "admin" if actor.role == "admin" else "user"
    return actor.mode  # "guest" | "local" | "api_key"


def record_activity(
    kind: str, request: Request, actor, settings: Settings, path: str | None = None
) -> None:
    if not settings.activity_tracking_enabled:
        return
    try:
        session = sessionmaker(bind=get_engine(), expire_on_commit=False)()
    except Exception:
        logger.warning("activity log unavailable (no engine)", exc_info=True)
        return
    try:
        session.add(
            ActivityEvent(
                kind=kind,
                actor_kind=actor_kind(actor),
                visitor=visitor_id(request, settings),
                user_id=getattr(actor, "user_id", None) if actor_kind(actor) in ("user", "admin") else None,
                path=(path or None) and path[:64],
            )
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.warning("activity write failed", exc_info=True)
    finally:
        session.close()


def track_question(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Route dependency for the chat endpoints (runs after auth and rate limits)."""
    record_activity("question", request, getattr(request.state, "actor", None), settings)


# ------------------------------------------------------------------ summary


def _day(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).date().isoformat()


def activity_summary(session: Session, days: int) -> dict:
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_keys = [(since + timedelta(days=i)).date().isoformat() for i in range(days)]

    events = session.execute(
        select(
            ActivityEvent.kind,
            ActivityEvent.actor_kind,
            ActivityEvent.visitor,
            ActivityEvent.user_id,
            ActivityEvent.created_at,
        ).where(ActivityEvent.created_at >= since)
    ).all()
    users = session.execute(
        select(User.id, User.email, User.display_name, User.role, User.created_at)
    ).all()
    regular = {u.id: u for u in users if u.role == "user"}
    uploads = session.execute(
        select(Document.owner_user_id, Document.created_at).where(
            Document.owner_user_id.is_not(None)
        )
    ).all()

    daily = {
        d: {"day": d, "visitors": set(), "guests_tried": set(), "guest_questions": 0,
            "signups": 0, "user_questions": 0}
        for d in day_keys
    }
    visitors: set[str] = set()
    guests_tried: set[str] = set()
    guest_questions = user_questions = 0
    active_users: set = set()
    per_user = defaultdict(lambda: {"questions": 0, "uploads": 0, "last_active": None})

    def touch(user_id, when):
        stats = per_user[user_id]
        if stats["last_active"] is None or when > stats["last_active"]:
            stats["last_active"] = when

    for kind, kind_of_actor, visitor, user_id, created in events:
        bucket = daily.get(_day(created))
        if kind == "visit":
            visitors.add(visitor)
            if bucket:
                bucket["visitors"].add(visitor)
        elif kind == "question":
            if kind_of_actor == "guest":
                guest_questions += 1
                guests_tried.add(visitor)
                if bucket:
                    bucket["guest_questions"] += 1
                    bucket["guests_tried"].add(visitor)
            elif user_id in regular:
                user_questions += 1
                active_users.add(user_id)
                per_user[user_id]["questions"] += 1
                touch(user_id, created)
                if bucket:
                    bucket["user_questions"] += 1

    window_uploads = 0
    for owner, created in uploads:
        if owner not in regular:
            continue
        per_user[owner]["uploads"] += 1
        touch(owner, created)
        if created.replace(tzinfo=created.tzinfo or timezone.utc) >= since:
            window_uploads += 1
            active_users.add(owner)

    signups = 0
    for user in regular.values():
        created = user.created_at.replace(tzinfo=user.created_at.tzinfo or timezone.utc)
        if created >= since:
            signups += 1
            bucket = daily.get(_day(created))
            if bucket:
                bucket["signups"] += 1

    people = sorted(
        (
            {
                "email": user.email,
                "display_name": user.display_name,
                "joined": user.created_at,
                "questions": per_user[user.id]["questions"],
                "uploads": per_user[user.id]["uploads"],
                "last_active": per_user[user.id]["last_active"],
            }
            for user in regular.values()
        ),
        key=lambda row: (row["last_active"] or row["joined"]).replace(
            tzinfo=(row["last_active"] or row["joined"]).tzinfo or timezone.utc
        ),
        reverse=True,
    )[:100]

    return {
        "days": days,
        "totals": {
            "visitors": len(visitors),
            "guests_tried": len(guests_tried),
            "guest_questions": guest_questions,
            "signups": signups,
            "active_users": len(active_users),
            "user_questions": user_questions,
            "uploads": window_uploads,
            "total_users": len(regular),
        },
        "daily": [
            {**row, "visitors": len(row["visitors"]), "guests_tried": len(row["guests_tried"])}
            for row in daily.values()
        ],
        "users": people,
    }
