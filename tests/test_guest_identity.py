"""Guest bearer-cookie parsing and secure-cookie configuration checks."""

import pytest
from fastapi.testclient import TestClient

import server_v2
from guest_identity import COOKIE_NAME, guest_id


@pytest.mark.parametrize("token", [
    None, 42, b"a" * 64, "", "a" * 63, "a" * 65, "A" * 64,
    "a" * 63 + "G", "a" * 32 + "\n" + "a" * 31,
    "a" * 32 + "\x00" + "a" * 31, "a" * 32 + " " + "a" * 31,
])
def test_guest_id_rejects_invalid_token_types_lengths_and_characters(token):
    assert guest_id(token) is None


def test_guest_id_is_stable_and_uuid_scoped():
    token = "0a" * 32
    identity = guest_id(token)
    assert identity == guest_id(token)
    assert identity != token
    assert len(identity) == 36
    assert guest_id("0b" * 32) != identity


def test_samesite_none_requires_secure_cookie(monkeypatch):
    monkeypatch.setenv("BLUFF_COOKIE_SAMESITE", "none")
    monkeypatch.delenv("BLUFF_COOKIE_SECURE", raising=False)
    with TestClient(server_v2.app, base_url="http://localhost") as client:
        response = client.get("/guest")
    assert response.status_code == 503
    assert response.json() == {"detail": "Guest cookie security is not configured"}
    assert "set-cookie" not in response.headers


@pytest.mark.parametrize("base_url", ["https://localhost", "http://localhost"])
def test_secure_guest_cookie_flags_for_https_or_explicit_secure_setting(
        monkeypatch, base_url):
    monkeypatch.setenv("BLUFF_COOKIE_SAMESITE", "none")
    if base_url.startswith("http://"):
        monkeypatch.setenv("BLUFF_COOKIE_SECURE", "1")
    else:
        monkeypatch.delenv("BLUFF_COOKIE_SECURE", raising=False)
    with TestClient(server_v2.app, base_url=base_url) as client:
        response = client.get("/guest")
    assert response.status_code == 200
    cookie = response.headers["set-cookie"].lower()
    assert COOKIE_NAME.lower() in cookie
    assert "httponly" in cookie and "secure" in cookie and "samesite=none" in cookie
