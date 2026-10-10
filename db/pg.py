"""Neon Postgres logging for web games (S3 data pipeline).

Best-effort infrastructure handling logs failures without breaking gameplay.
Profile reads and stale-write conflicts propagate to the server so it can
avoid overwriting learning. Other storage failures return failure sentinels.
The server also runs with no DATABASE_URL set.

Schema: db/schema.sql. Anonymous and non-consenting rooms create no session
rows. Verified Clerk subjects are deterministically mapped to UUIDs for
per-user opponent models; the raw subject is not stored in the logging schema.
"""

import json
import logging
import os
import time
import traceback
import uuid
from typing import Optional

_log = logging.getLogger("bluff.db")

_pool = None
_warned = False
_UNCONDITIONAL_PROFILE_SAVE = object()


class ProfileWriteConflict(RuntimeError):
    """Another completed game updated this profile after it was loaded."""


def _warn_once(msg: str) -> None:
    """P2 F4: loud log on first failure + every subsequent failure."""
    global _warned
    _log.warning(msg)
    if not _warned:
        _warned = True
        print(f"[db] WARNING: {msg}")


async def get_pool():
    """Lazy asyncpg pool, or None when unconfigured/unreachable."""
    global _pool
    if _pool is not None:
        return _pool
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    try:
        import asyncpg
        _pool = await asyncpg.create_pool(url, min_size=1, max_size=2,
                                          command_timeout=10)
        return _pool
    except Exception as e:  # noqa: BLE001 — must never propagate
        _log.error("asyncpg pool creation FAILED: %s\n%s", e, traceback.format_exc())
        _warn_once(f"connect failed ({e}); logging disabled for this process")
        return None


async def log_session_start(bot_mode: str, player_id: str,
                            consent_version: Optional[str] = None):
    """Insert an opted-in research session. Missing consent is a no-op.

    Anonymous and non-consenting games are deliberately not persisted.
    """
    if not player_id or not consent_version:
        return None
    try:
        pool = await get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            async with conn.transaction():
                user_uuid = uuid.UUID(player_id)
                consent = await conn.fetchrow(
                    "SELECT research_logging, consent_version "
                    "FROM user_privacy_preferences WHERE user_id=$1 FOR SHARE",
                    user_uuid,
                )
                if (consent is None or not consent["research_logging"]
                        or consent["consent_version"] != consent_version):
                    return None
                row = await conn.fetchrow(
                    "INSERT INTO game_sessions "
                    "(bot_mode, result, player_id, research_consent_version) "
                    "VALUES ($1, 'started', $2, $3) RETURNING id",
                    bot_mode, user_uuid, consent_version,
                )
            return str(row["id"])
    except Exception as e:  # noqa: BLE001
        _log.error("log_session_start FAILED: %s\n%s", e, traceback.format_exc())
        _warn_once(f"log_session_start failed ({e})")
        return None


async def set_model_loaded(session_id, loaded: bool) -> None:
    """Record whether a persisted opponent model was loaded at session start.

    Ablation switch for the claim-(b) headline experiment (docs/s3-design.md
    §4): win-rate delta between loaded vs cold-start sessions for the same
    user. Best-effort like everything here.
    """
    if session_id is None:
        return
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE game_sessions SET model_loaded = $2 WHERE id = $1",
                session_id, loaded,
            )
    except Exception as e:  # noqa: BLE001
        _log.error("set_model_loaded FAILED (session=%s): %s\n%s", session_id, e, traceback.format_exc())
        _warn_once(f"set_model_loaded failed ({e})")


async def set_session_provenance(session_id, provenance: dict) -> None:
    """Write once; a later call must not silently redefine a session's policy."""
    if session_id is None:
        return
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE game_sessions SET policy_provenance = $2 "
                "WHERE id = $1 AND policy_provenance IS NULL",
                session_id, json.dumps(provenance, allow_nan=False),
            )
    except Exception as e:  # noqa: BLE001 — audit failure must not stop play
        _log.error("set_session_provenance FAILED (session=%s): %s", session_id, e)
        _warn_once(f"set_session_provenance failed ({e})")


async def log_action(session_id, turn_number: int, player_type: str,
                     action_type: str, cards_played=None, claimed_rank=None,
                     was_bluff=None, bluff_called: bool = False,
                     caller_was_right=None, hand_size=None,
                     opponent_hand_size=None, pile_size=None,
                     p_bluff_estimate=None, decision_ms=None,
                     state_vector=None, legal_mask=None, action_index=None,
                     state_schema=None, claim_id=None,
                     selected_action_probability=None,
                     selected_action_probability_kind=None,
                     bluff_head_state_vector=None, bluff_head_schema=None,
                     bluff_head_conflict=None, bluff_head_aux=None) -> None:
    """Insert one actions row. Silent no-op when session_id is None."""
    if session_id is None:
        return
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            # JSONB columns need a JSON *string* — asyncpg rejects Python
            # lists/dicts for jsonb (DataError). Server passes _db_cards()
            # lists; without this every play/call row silently failed.
            cards_json = (json.dumps(cards_played)
                          if cards_played is not None else None)
            state_json = (json.dumps(state_vector)
                          if state_vector is not None else None)
            mask_json = (json.dumps(legal_mask)
                         if legal_mask is not None else None)
            head_state_json = (json.dumps(bluff_head_state_vector)
                               if bluff_head_state_vector is not None else None)
            head_aux_json = (json.dumps(bluff_head_aux)
                             if bluff_head_aux is not None else None)
            await conn.execute(
                "INSERT INTO actions (game_id, turn_number, player_type, "
                "action_type, cards_played, claimed_rank, was_bluff, "
                "bluff_called, caller_was_right, hand_size, "
                "opponent_hand_size, pile_size, p_bluff_estimate, "
                "decision_ms, state_vector, legal_mask, action_index, "
                "state_schema, claim_id, selected_action_probability, "
                "selected_action_probability_kind, bluff_head_state_vector, "
                "bluff_head_schema, bluff_head_conflict, bluff_head_aux) SELECT "
                "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25 "
                "WHERE EXISTS (SELECT 1 FROM game_sessions "
                "WHERE id=$1 AND research_consent_version IS NOT NULL)",
                session_id, turn_number, player_type, action_type,
                cards_json, claimed_rank, was_bluff, bluff_called,
                caller_was_right, hand_size, opponent_hand_size, pile_size,
                p_bluff_estimate, decision_ms, state_json, mask_json,
                action_index, state_schema, uuid.UUID(str(claim_id)) if claim_id is not None else None,
                selected_action_probability, selected_action_probability_kind,
                head_state_json, bluff_head_schema, bluff_head_conflict,
                head_aux_json,
            )
    except Exception as e:  # noqa: BLE001
        _log.error("log_action FAILED (session=%s turn=%s): %s\n%s",
                   session_id, turn_number, e, traceback.format_exc())
        _warn_once(f"log_action failed ({e})")


async def log_session_end(session_id, result: str, num_turns: int,
                          duration_seconds: int) -> None:
    """Mark the session finished. result ∈ win/loss/draw (human POV)."""
    if session_id is None:
        return
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE game_sessions SET result=$2, num_turns=$3, "
                "duration_seconds=$4, finished_at=NOW() WHERE id=$1",
                session_id, result, num_turns, duration_seconds,
            )
    except Exception as e:  # noqa: BLE001
        _log.error("log_session_end FAILED (session=%s): %s\n%s", session_id, e, traceback.format_exc())
        _warn_once(f"log_session_end failed ({e})")


# ---------------------------------------------------------------------------
# Per-user opponent-model persistence (JSONB, keyed by user_id + bot_id)
# ---------------------------------------------------------------------------

async def ensure_user(user_id: str) -> None:
    """Create a session-level user row if it doesn't exist."""
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id, display_name) VALUES ($1, $2) "
                "ON CONFLICT (id) DO NOTHING",
                user_id, f"player-{user_id[:8]}",
            )
    except Exception as e:  # noqa: BLE001
        _log.error("ensure_user FAILED (user=%s): %s\n%s", user_id, e, traceback.format_exc())
        _warn_once(f"ensure_user failed ({e})")


async def get_privacy_preferences(user_id: str) -> dict:
    """Return explicit purposes; a missing row is fail-closed (both off)."""
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT cross_session_profile, research_logging, consent_version "
            "FROM user_privacy_preferences WHERE user_id=$1",
            uuid.UUID(user_id),
        )
    if row is None:
        return {"cross_session_profile": False, "research_logging": False,
                "consent_version": None}
    return {"cross_session_profile": bool(row["cross_session_profile"]),
            "research_logging": bool(row["research_logging"]),
            "consent_version": row["consent_version"]}


async def update_privacy_preferences(user_id: str, *,
                                     cross_session_profile: bool,
                                     research_logging: bool,
                                     consent_version: str) -> dict:
    """Set purpose-specific preferences and delete revoked stored data atomically."""
    if (type(cross_session_profile) is not bool
            or type(research_logging) is not bool
            or not isinstance(consent_version, str) or not consent_version):
        raise ValueError("invalid privacy preference values")
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    user_uuid = uuid.UUID(user_id)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO users (id, display_name) VALUES ($1, $2) "
                "ON CONFLICT (id) DO NOTHING",
                user_uuid, f"player-{user_id[:8]}",
            )
            await conn.execute(
                "INSERT INTO user_privacy_preferences "
                "(user_id, cross_session_profile, research_logging, consent_version) "
                "VALUES ($1, $2, $3, $4) ON CONFLICT (user_id) DO UPDATE SET "
                "cross_session_profile=EXCLUDED.cross_session_profile, "
                "research_logging=EXCLUDED.research_logging, "
                "consent_version=EXCLUDED.consent_version, updated_at=NOW()",
                user_uuid, cross_session_profile, research_logging, consent_version,
            )
            if not cross_session_profile:
                await conn.execute(
                    "DELETE FROM opponent_models WHERE user_id=$1", user_uuid)
            if not research_logging:
                await conn.execute(
                    "DELETE FROM game_sessions WHERE player_id=$1", user_uuid)
    return {"cross_session_profile": cross_session_profile,
            "research_logging": research_logging,
            "consent_version": consent_version}


async def update_profile_preference(user_id: str, *, enabled: bool) -> dict:
    """Change profile consent only; preserve research consent/history/version.

    Existing consent_version also governs historical research logging, so do
    not rewrite it on an existing row as a side effect of this narrow control.
    A new row has research disabled. Profile revocation deletes saved profiles
    atomically with the flag change, using the same row lock as profile saves.
    """
    if type(enabled) is not bool:
        raise ValueError("profile consent must be a boolean")
    pool = await get_pool()
    if pool is None:
        raise RuntimeError("database pool unavailable")
    user_uuid = uuid.UUID(user_id)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO users (id, display_name) VALUES ($1, $2) "
                "ON CONFLICT (id) DO NOTHING",
                user_uuid, f"player-{user_id[:8]}",
            )
            await conn.execute(
                "INSERT INTO user_privacy_preferences "
                "(user_id, cross_session_profile, research_logging, consent_version) "
                "VALUES ($1, $2, FALSE, 'bluff-profile-consent-v1') "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "cross_session_profile=EXCLUDED.cross_session_profile, "
                "updated_at=NOW()",
                user_uuid, enabled,
            )
            if not enabled:
                await conn.execute(
                    "DELETE FROM opponent_models WHERE user_id=$1", user_uuid)
    return {"cross_session_profile": enabled}


async def save_opponent_model(user_id: str, bot_id: str,
                              model_data: dict, *,
                              expected_revision=_UNCONDITIONAL_PROFILE_SAVE) -> bool:
    """Save (or overwrite) the serialized bot model for a (user, bot) pair.

    Increments games_played on each save so callers can gauge data volume.
    Live rooms supply the loaded games_played revision (None for a new row).
    A stale write raises ProfileWriteConflict; infrastructure failures return
    False. A missing or revoked profile preference always blocks the write.
    """
    if not isinstance(model_data, dict):
        raise TypeError("opponent profile must be a JSON object")
    try:
        pool = await get_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            async with conn.transaction():
                consent = await conn.fetchrow(
                    "SELECT cross_session_profile FROM user_privacy_preferences "
                    "WHERE user_id=$1 FOR SHARE", uuid.UUID(user_id))
                if consent is None or not consent["cross_session_profile"]:
                    return False
                if expected_revision is not _UNCONDITIONAL_PROFILE_SAVE:
                    if expected_revision is None:
                        row = await conn.fetchrow(
                            "INSERT INTO opponent_models (user_id, bot_id, model_data, games_played) "
                            "VALUES ($1, $2, $3::jsonb, 1) "
                            "ON CONFLICT (user_id, bot_id) DO NOTHING RETURNING games_played",
                            user_id, bot_id, json.dumps(model_data))
                    else:
                        if type(expected_revision) is not int or expected_revision < 0:
                            raise ValueError("invalid profile revision")
                        row = await conn.fetchrow(
                            "UPDATE opponent_models SET model_data=$3::jsonb, "
                            "games_played=games_played+1, updated_at=NOW() "
                            "WHERE user_id=$1 AND bot_id=$2 AND games_played=$4 RETURNING games_played",
                            user_id, bot_id, json.dumps(model_data), expected_revision)
                    if row is None:
                        raise ProfileWriteConflict("profile changed since game start")
                    return True
                await conn.execute(
                    "INSERT INTO opponent_models "
                    "(user_id, bot_id, model_data, games_played) "
                    "VALUES ($1, $2, $3::jsonb, 1) "
                    "ON CONFLICT (user_id, bot_id) DO UPDATE SET "
                    "model_data = EXCLUDED.model_data, "
                    "games_played = opponent_models.games_played + 1, "
                    "updated_at = NOW()",
                    user_id, bot_id, json.dumps(model_data),
                )
        return True
    except ProfileWriteConflict:
        raise
    except Exception as e:  # noqa: BLE001 — must never propagate
        _log.error("save_opponent_model FAILED (user=%s bot=%s): %s\n%s",
                   user_id, bot_id, e, traceback.format_exc())
        _warn_once(f"save_opponent_model failed ({e})")
        return False


async def load_opponent_model(user_id: str, bot_id: str) -> Optional[dict]:
    """Compatibility reader; live rooms also need the atomic snapshot revision."""
    data, _revision = await load_opponent_model_snapshot(user_id, bot_id)
    return data


async def load_opponent_model_snapshot(user_id: str, bot_id: str) -> tuple[Optional[dict], Optional[int]]:
    """Read model and revision together; (None, None) means a successful miss.

    Storage/read/decoding failures propagate so callers can avoid replacing a
    profile they could not safely read.
    """
    try:
        pool = await get_pool()
        if pool is None:
            raise RuntimeError("database pool unavailable")
        async with pool.acquire() as conn:
            async with conn.transaction():
                consent = await conn.fetchrow(
                    "SELECT cross_session_profile FROM user_privacy_preferences "
                    "WHERE user_id=$1 FOR SHARE", uuid.UUID(user_id))
                if consent is None or not consent["cross_session_profile"]:
                    return None, None
                row = await conn.fetchrow(
                    "SELECT model_data, games_played FROM opponent_models "
                    "WHERE user_id = $1 AND bot_id = $2",
                    user_id, bot_id,
                )
                if row is None:
                    return None, None
                revision = row["games_played"]
                if type(revision) is not int or revision < 0:
                    raise ValueError("invalid stored profile revision")
                model_data = json.loads(row["model_data"])
                if not isinstance(model_data, dict):
                    raise ValueError("stored opponent profile must be a JSON object")
                return model_data, revision
    except Exception as e:  # noqa: BLE001
        _log.error("load_opponent_model FAILED (user=%s bot=%s): %s\n%s",
                   user_id, bot_id, e, traceback.format_exc())
        _warn_once(f"load_opponent_model failed ({e})")
        raise
