"""Networkless Clerk session verification for account-owned profile storage.

No database, game engine, bot weights or account-management side effects.
Configuration is explicit; unverified browser identifiers are never identities.
"""

import os
import uuid
from typing import Optional

import jwt
from fastapi import HTTPException


def normalize_user_id(raw_id: Optional[str]) -> Optional[str]:
    """Map a verified Clerk subject to the existing PostgreSQL UUID key."""
    if not raw_id:
        return None
    try:
        return str(uuid.UUID(raw_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


def profile_id_from_authorization(authorization: Optional[str]) -> Optional[str]:
    """Return an identity only after signature, time, issuer and origin checks."""
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    public_key = os.environ.get("CLERK_JWT_KEY", "").replace("\\n", "\n").strip()
    issuer = os.environ.get("CLERK_ISSUER", "").strip()
    parties = tuple(value.strip() for value in os.environ.get(
        "CLERK_AUTHORIZED_PARTIES", "").split(",") if value.strip())
    if not public_key or not issuer or not parties:
        raise HTTPException(
            status_code=503,
            detail="Persistent profiles require Clerk verification configuration",
        )

    try:
        claims = jwt.decode(
            token.strip(), public_key, algorithms=["RS256"], issuer=issuer,
            options={"require": ["exp", "iat", "nbf", "iss", "sub"],
                     "verify_aud": False},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid Clerk session") from exc

    subject = claims.get("sub")
    if (not isinstance(subject, str) or not subject
            or claims.get("azp") not in parties or claims.get("sts") == "pending"):
        raise HTTPException(status_code=401, detail="Clerk session not authorized")
    return normalize_user_id(subject)
