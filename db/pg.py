"""Neon Postgres logging for web games (S3 data pipeline).

Best-effort by design: every public function catches ALL exceptions and
returns None on failure. Gameplay must NEVER break because the database is
down, slow, or unconfigured (server runs fine with no DATABASE_URL set).

Schema: db/schema.sql. Anonymous play uses player_id NULL until Clerk token
verification lands (then bind users.clerk_user_id → per-user opponent models).
"""

import json
import os
import time
import traceback
import uuid
from typing import Optional

_pool = None
_warned = False


def _warn_once(msg: str) -> None:
    global _warned
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
        _warn_once(f"connect failed ({e}); logging disabled for this process")
        return None


async def log_session_start(bot_mode: str,
                            player_id: Optional[str] = None):
    """Insert a game_sessions row. Returns session id (str) or None.

    player_id: stable session-level user id (Clerk sub later). NULL keeps
    anonymous rows — every web game is still attributable via session_user_id
    once ensure_user has created the users row.
    """
    try:
        pool = await get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            if player_id:
                row = await conn.fetchrow(
                    "INSERT INTO game_sessions (bot_mode, result, player_id) "
                    "VALUES ($1, 'started', $2) RETURNING id",
                    bot_mode, uuid.UUID(player_id),
                )
            else:
                row = await conn.fetchrow(
                    "INSERT INTO game_sessions (bot_mode, result) "
                    "VALUES ($1, 'started') RETURNING id",
                    bot_mode,
                )
            return str(row["id"])
    except Exception:  # noqa: BLE001
        _warn_once("log_session_start failed\n" + traceback.format_exc(limit=3))
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
    except Exception:  # noqa: BLE001
        _warn_once("set_model_loaded failed\n" + traceback.format_exc(limit=3))


async def log_action(session_id, turn_number: int, player_type: str,
                     action_type: str, cards_played=None, claimed_rank=None,
                     was_bluff=None, bluff_called: bool = False,
                     caller_was_right=None, hand_size=None,
                     opponent_hand_size=None, pile_size=None,
                     p_bluff_estimate=None, decision_ms=None) -> None:
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
            await conn.execute(
                "INSERT INTO actions (game_id, turn_number, player_type, "
                "action_type, cards_played, claimed_rank, was_bluff, "
                "bluff_called, caller_was_right, hand_size, "
                "opponent_hand_size, pile_size, p_bluff_estimate, "
                "decision_ms) VALUES "
                "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",
                session_id, turn_number, player_type, action_type,
                cards_json, claimed_rank, was_bluff, bluff_called,
                caller_was_right, hand_size, opponent_hand_size, pile_size,
                p_bluff_estimate, decision_ms,
            )
    except Exception:  # noqa: BLE001
        _warn_once("log_action failed\n" + traceback.format_exc(limit=3))


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
    except Exception:  # noqa: BLE001
        _warn_once("log_session_end failed\n" + traceback.format_exc(limit=3))


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
                user_id, f"session-{user_id[:8]}",
            )
    except Exception:  # noqa: BLE001
        _warn_once("ensure_user failed\n" + traceback.format_exc(limit=3))


async def save_opponent_model(user_id: str, bot_id: str,
                              model_data: dict) -> None:
    """Save (or overwrite) the serialized bot model for a (user, bot) pair.

    Increments games_played on each save so callers can gauge data volume.
    Best-effort: never propagates exceptions.
    """
    try:
        pool = await get_pool()
        if pool is None:
            return
        async with pool.acquire() as conn:
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
    except Exception:  # noqa: BLE001
        _warn_once("save_opponent_model failed\n"
                    + traceback.format_exc(limit=3))


async def load_opponent_model(user_id: str, bot_id: str) -> Optional[dict]:
    """Load the serialized bot model for a (user, bot) pair, or None."""
    try:
        pool = await get_pool()
        if pool is None:
            return None
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT model_data FROM opponent_models "
                "WHERE user_id = $1 AND bot_id = $2",
                user_id, bot_id,
            )
            if row is None:
                return None
            return json.loads(row["model_data"])
    except Exception:  # noqa: BLE001
        _warn_once("load_opponent_model failed\n"
                    + traceback.format_exc(limit=3))
        return None
