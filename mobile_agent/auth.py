from __future__ import annotations

import hashlib
from typing import Any

from langgraph_sdk import Auth

auth = Auth()

LOCAL_DEV_IDENTITY = "local-dev"


@auth.authenticate
async def authenticate(
    headers: dict[bytes, bytes],
    authorization: str | None = None,
) -> Auth.types.MinimalUserDict:
    token = _header_value(headers, b"x-api-key")
    if token:
        return _user("api-key", token)

    bearer = _bearer_token(authorization)
    if bearer:
        return _user("bearer", bearer)

    return {
        "identity": LOCAL_DEV_IDENTITY,
        "permissions": [],
        "is_authenticated": False,
    }


@auth.on.threads
async def authorize_threads(
    ctx: Auth.types.AuthContext,
    value: dict[str, Any],
) -> Auth.types.FilterType:
    filters = {"owner": ctx.user.identity}
    if ctx.action == "create":
        metadata = value.setdefault("metadata", {})
        metadata.update(filters)
    return filters


def _user(source: str, token: str) -> Auth.types.MinimalUserDict:
    return {
        "identity": f"{source}:{_digest(token)}",
        "permissions": [],
        "is_authenticated": True,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _header_value(headers: dict[bytes, bytes], name: bytes) -> str:
    lowered = {key.lower(): value for key, value in headers.items()}
    value = lowered.get(name.lower())
    if not value:
        return ""
    return value.decode("utf-8", errors="ignore").strip()


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        return ""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


__all__ = ["auth", "authenticate", "authorize_threads"]
