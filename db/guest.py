"""Strict guest-owned storage for preferences and consented game transcripts."""

import json
import uuid

from db import pg

_STATE_SCHEMA = "bluff-fixed-rounds-v2"
_ACTIONS = {"play", "pass", "call", "start", "end"}


def _uuid(value):
    if not isinstance(value, str):
        raise ValueError("id must be a UUID string")
    return uuid.UUID(value)


def _json_object(value, name):
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON data") from exc
    return encoded


def _consent(row, version):
    return (row is not None and row["research_logging"] is True
            and row["consent_version"] == version)


async def get_preferences(user_id: str) -> dict:
    return await pg.get_privacy_preferences(user_id)


async def set_preferences(user_id: str, *, cross_session_profile: bool,
                          research_logging: bool, consent_version: str) -> dict:
    return await pg.update_privacy_preferences(
        user_id, cross_session_profile=cross_session_profile,
        research_logging=research_logging, consent_version=consent_version)


async def delete_data(user_id: str) -> bool:
    user_uuid = _uuid(user_id)
    pool = await pg.get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM game_sessions WHERE player_id=$1", user_uuid)
            await conn.execute("DELETE FROM opponent_models WHERE user_id=$1", user_uuid)
            await conn.execute("DELETE FROM users WHERE id=$1", user_uuid)
    return True


async def start_game(user_id: str, bot_name: str, game_id: str,
                     provenance: dict, consent_version: str) -> bool:
    user_uuid, game_uuid = _uuid(user_id), _uuid(game_id)
    if not isinstance(bot_name, str) or not bot_name or len(bot_name) > 20:
        raise ValueError("invalid bot name")
    if not isinstance(consent_version, str) or not consent_version or len(consent_version) > 64:
        raise ValueError("invalid consent version")
    provenance_json = _json_object(provenance, "provenance")
    pool = await pg.get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    async with pool.acquire() as conn:
        async with conn.transaction():
            consent = await conn.fetchrow(
                "SELECT research_logging, consent_version FROM user_privacy_preferences "
                "WHERE user_id=$1 FOR SHARE", user_uuid)
            if not _consent(consent, consent_version):
                return False
            await conn.execute(
                "INSERT INTO game_sessions (id, player_id, bot_mode, result, "
                "research_consent_version, policy_provenance) "
                "VALUES ($1,$2,$3,'started',$4,$5::jsonb) ON CONFLICT (id) DO NOTHING",
                game_uuid, user_uuid, bot_name, consent_version, provenance_json)
            row = await conn.fetchrow(
                "SELECT player_id, bot_mode, research_consent_version, policy_provenance "
                "FROM game_sessions WHERE id=$1", game_uuid)
    if (row is None or row["player_id"] != user_uuid
            or row["bot_mode"] != bot_name):
        return False
    stored_provenance = row["policy_provenance"]
    if isinstance(stored_provenance, str):
        stored_provenance = json.loads(stored_provenance)
    return (row["research_consent_version"] == consent_version
            and stored_provenance == json.loads(provenance_json))


async def append_event(user_id: str, game_id: str, event_index: int,
                       event: dict) -> bool:
    user_uuid, game_uuid = _uuid(user_id), _uuid(game_id)
    if type(event_index) is not int or event_index < 0:
        raise ValueError("event index must be a nonnegative integer")
    event_json = _json_object(event, "event")
    player_type, action_type = event.get("player_type"), event.get("action_type")
    if (not isinstance(player_type, str) or player_type not in {"human", "bot"}
            or not isinstance(action_type, str) or action_type not in _ACTIONS):
        raise ValueError("event has invalid player_type or action_type")
    pool = await pg.get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    async with pool.acquire() as conn:
        async with conn.transaction():
            consent = await conn.fetchrow(
                "SELECT research_logging, consent_version FROM user_privacy_preferences "
                "WHERE user_id=$1 FOR SHARE", user_uuid)
            if consent is None or consent["research_logging"] is not True:
                return False
            session = await conn.fetchrow(
                "SELECT id, finished_at FROM game_sessions WHERE id=$1 AND player_id=$2 "
                "AND research_consent_version=$3 FOR UPDATE",
                game_uuid, user_uuid, consent["consent_version"])
            if session is None:
                return False
            existing = await conn.fetchrow(
                "SELECT state_vector FROM actions WHERE game_id=$1 AND turn_number=$2 "
                "AND state_schema=$3", game_uuid, event_index, _STATE_SCHEMA)
            if existing is not None:
                stored = existing["state_vector"]
                if isinstance(stored, str):
                    stored = json.loads(stored)
                if stored == event:
                    return True
                raise ValueError("event index already contains different data")
            if session["finished_at"] is not None:
                return False
            await conn.execute(
                "INSERT INTO actions (game_id, turn_number, player_type, action_type, "
                "state_schema, state_vector) VALUES ($1,$2,$3,$4,$5,$6::jsonb)",
                game_uuid, event_index, player_type, action_type, _STATE_SCHEMA, event_json)
    return True


async def finish_game(user_id: str, game_id: str, result: str, num_turns: int,
                      duration_seconds: int) -> bool:
    user_uuid, game_uuid = _uuid(user_id), _uuid(game_id)
    if not isinstance(result, str) or result not in {"win", "loss", "abandoned"}:
        raise ValueError("invalid result")
    for name, value in (("num_turns", num_turns), ("duration_seconds", duration_seconds)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    pool = await pg.get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    async with pool.acquire() as conn:
        async with conn.transaction():
            consent = await conn.fetchrow(
                "SELECT research_logging, consent_version FROM user_privacy_preferences "
                "WHERE user_id=$1 FOR SHARE", user_uuid)
            if consent is None or consent["research_logging"] is not True:
                return False
            session = await conn.fetchrow(
                "SELECT result, num_turns, duration_seconds, finished_at "
                "FROM game_sessions WHERE id=$1 AND player_id=$2 "
                "AND research_consent_version=$3 FOR UPDATE",
                game_uuid, user_uuid, consent["consent_version"])
            if session is None:
                return False
            final = (result, num_turns, duration_seconds)
            if session["finished_at"] is not None:
                if (session["result"], session["num_turns"],
                        session["duration_seconds"]) == final:
                    return True
                raise ValueError("game already finished with different results")
            await conn.execute(
                "UPDATE game_sessions SET result=$3, num_turns=$4, duration_seconds=$5, "
                "finished_at=NOW() WHERE id=$1 AND player_id=$2 "
                "AND research_consent_version=$6",
                game_uuid, user_uuid, result, num_turns, duration_seconds,
                consent["consent_version"])
    return True
