from __future__ import annotations

from collections.abc import Callable
from hmac import compare_digest
from typing import Annotated

from fastapi import Header, HTTPException

from face_classifier_service.settings import Settings


def raise_unauthorized(detail: str) -> None:
    raise HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def token_matches(token: str, valid_tokens: tuple[str, ...]) -> bool:
    matches = False
    for valid_token in valid_tokens:
        matches |= compare_digest(token, valid_token)
    return matches


def validate_bearer_token(authorization: str | None, settings: Settings) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise_unauthorized("missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if not token or not token_matches(token, settings.api_tokens):
        raise_unauthorized("invalid bearer token")


def create_auth_dependency(settings: Settings) -> Callable:
    async def require_bearer_token(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        validate_bearer_token(authorization, settings)

    return require_bearer_token
