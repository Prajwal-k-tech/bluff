"""Single-origin deployment; local development still uses server_v2 directly."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server_v2 import app as api, root as api_health


def create_app():
    """Fail closed unless an exported frontend has been explicitly supplied."""
    directory = Path(os.environ.get("BLUFF_WEB_DIR", "/app/web")).resolve()
    for page in ("index.html", "game/index.html"):
        if not (directory / page).is_file():
            raise RuntimeError(f"Missing exported frontend page: {directory / page}")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_api_route("/health", api_health, methods=["GET"])
    app.mount("/api", api)
    app.mount("/", StaticFiles(directory=directory, html=True))
    return app
