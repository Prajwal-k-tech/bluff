import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from web_app import create_app


@pytest.fixture
def exported_app(tmp_path, monkeypatch):
    (tmp_path / "game").mkdir()
    (tmp_path / "index.html").write_text("<h1>Bluff</h1>")
    (tmp_path / "game/index.html").write_text("<h1>Game</h1>")
    monkeypatch.setenv("BLUFF_WEB_DIR", str(tmp_path))
    return create_app()


def test_static_pages_and_api_are_separate(exported_app):
    with TestClient(exported_app) as client:
        assert client.get("/").text == "<h1>Bluff</h1>"
        assert client.get("/game/").text == "<h1>Game</h1>"
        assert client.get("/health").json()["ruleset_id"] == "bluff-fixed-rounds-v2"
        assert client.get("/api/bots").json()["bots"] == ["flagship", "math", "honest", "random"]
        assert client.get("/api/missing").status_code == 404
        assert client.get("/.env").status_code == 404
        assert client.get("/%2e%2e/server_v2.py").status_code == 404


def test_guest_cookie_and_websocket_work_under_mount(exported_app):
    with TestClient(exported_app) as client:
        response = client.post("/api/rooms?bot_name=honest")
        assert response.status_code == 200
        assert client.cookies.get("bluff_guest_v1")
        with client.websocket_connect(f"/api/ws/{response.json()['room_id']}") as socket:
            assert socket.receive_json()["type"] == "game_state"


def test_missing_export_fails_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUFF_WEB_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="Missing exported frontend"):
        create_app()


def test_web_build_copies_only_declared_frontend_inputs():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile.web").read_text()
    assert "COPY frontend/ ./" not in dockerfile
    assert "COPY frontend/src/ ./src/" in dockerfile
    assert "frontend/public/fonts/OFL-CormorantGaramond.txt" in dockerfile
    assert "**/.clerk" in (root / ".dockerignore").read_text().splitlines()
