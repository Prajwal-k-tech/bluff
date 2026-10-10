"""Explicit browser-origin configuration, independent of either game server."""

import os
from urllib.parse import urlsplit


def configured_cors_origins() -> list[str]:
    """Allow exact HTTP(S) origins, never wildcard/path/credential URLs."""
    raw = (os.environ.get("BLUFF_CORS_ALLOWED_ORIGINS")
           or os.environ.get("CLERK_AUTHORIZED_PARTIES", ""))
    defaults = ["http://localhost:3000", "http://127.0.0.1:3000"]
    if not raw.strip():
        return defaults
    origins = []
    for value in raw.split(","):
        origin = value.strip().rstrip("/")
        if not origin:
            continue
        try:
            parsed = urlsplit(origin)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"Invalid CORS origin: {value.strip()!r}") from exc
        if (any(char in origin for char in ("*", "?", "#", "\\"))
                or any(char.isspace() or ord(char) < 32 or ord(char) == 127
                       for char in origin)
                or parsed.scheme not in ("http", "https")
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path or parsed.query
                or parsed.fragment or (port is not None and port == 0)):
            raise ValueError(f"Invalid CORS origin: {value.strip()!r}")
        origins.append(origin)
    return list(dict.fromkeys(origins)) or defaults
