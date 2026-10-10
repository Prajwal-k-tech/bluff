"""Anonymous bearer-cookie ownership; no names, accounts or fingerprinting.

The 256-bit credential stays HttpOnly. Only its namespaced hash-derived UUID
reaches storage. Choosing one's own token cannot select somebody else's UUID.
Losing the cookie loses access; these identities are not recoverable accounts.
"""

import hashlib
import os
import re
import secrets
from uuid import UUID

from fastapi import HTTPException

COOKIE_NAME = "bluff_guest_v1"
CONSENT_VERSION = "guest-fixed-rounds-v1"
_TOKEN = re.compile(r"[0-9a-f]{64}\Z")


def guest_id(token):
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    digest = hashlib.sha256(b"bluff-guest-v1\0" + token.encode("ascii")).digest()
    return str(UUID(bytes=digest[:16], version=4))


def check_origin(request, allowed_origins):
    """Protect cookie-authenticated writes as well as browser WebSockets."""
    origins = request.headers.getlist("origin")
    if len(origins) > 1 or (origins and origins[0] not in allowed_origins):
        raise HTTPException(403, "Origin not allowed")
    # Same-origin requests may omit Origin; cross-site browser requests may not.
    if not origins and request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Origin required")


def ensure_guest(request, response, *, rotate=False):
    token = None if rotate else request.cookies.get(COOKIE_NAME)
    identity = guest_id(token)
    if identity is None:
        token = secrets.token_hex(32)
        identity = guest_id(token)
        same_site = os.environ.get("BLUFF_COOKIE_SAMESITE", "lax").lower()
        secure = request.url.scheme == "https" or os.environ.get("BLUFF_COOKIE_SECURE") == "1"
        if same_site not in {"lax", "strict", "none"} or (same_site == "none" and not secure):
            raise HTTPException(503, "Guest cookie security is not configured")
        response.set_cookie(COOKIE_NAME, token, httponly=True, secure=secure,
                            samesite=same_site, max_age=365 * 24 * 3600, path="/")
    response.headers["Cache-Control"] = "no-store"
    return identity
