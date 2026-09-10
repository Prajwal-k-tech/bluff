"""End-to-end S3 persistence test — closes solo-push GB open items.

Verifies (in-memory where possible, Neon when DATABASE_URL is set):
  GB-i  log_session_end survives a live DB (finished_at UPDATE lands).
  GB-ii actions rows are written for real web games (log_action wired).
  GB-iii fabricated pass-Actions are safe for the opponent model.
  GB-1  save_opponent_model → load_opponent_model round-trip per (user, bot).
  GB-2  Cross-SESSION persistence: a fresh BayesianBot loaded from a saved
        model behaves measurably differently than a cold-start bot (the
        user's "truly learns over time" requirement, now test-verified).

Run:  python3 -m pytest test_s3_persistence.py -v
      (DB tests auto-skip when DATABASE_URL is unset)
"""

import asyncio
import os
import uuid

import pytest

from bots.bayesian_bot import BayesianBot, OpponentModel, BluffTracker
from cards import Card, Rank
from game import Action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_card(rank: Rank, suit: int = 0) -> Card:
    return Card(rank, suit)


def make_action(player=0, ranks=None, claimed=Rank.SEVEN, bluff=False,
                called=False, right=False) -> Action:
    cards = [make_card(r) for r in (ranks or [claimed])]
    return Action(
        player=player,
        cards_played=cards,
        claimed_rank=claimed,
        was_bluff=bluff,
        bluff_called=called,
        caller_was_right=right,
        pile_size_before=4,
    )


def play_some_bluffy_games(bot: BayesianBot, human_bluff_rate: float = 0.7,
                           games: int = 3) -> None:
    """Simulate observing a bluffy human for `games` hands."""
    for _ in range(games):
        bot.reset()
        for turn in range(12):
            bluff = (turn % 10) / 10 < human_bluff_rate  # mostly bluffs
            act = make_action(player=0, bluff=bluff,
                              called=False, right=False)
            bot.observe_action(act, opponent_hand_size=8)


# ---------------------------------------------------------------------------
# GB-iii: fabricated pass-Actions are safe (no DB needed)
# ---------------------------------------------------------------------------

def test_pass_action_does_not_pollute_rank_models():
    bot = BayesianBot()
    # Simulate 20 passes (fabricated empty-card actions from server.py)
    for _ in range(20):
        bot.observe_action(
            Action(player=0, cards_played=[], claimed_rank=Rank.TWO,
                   was_bluff=False, bluff_called=False,
                   caller_was_right=False, pile_size_before=5),
            opponent_hand_size=10,
        )
    # Rank/claim-size models must be untouched by passes
    rank_stats = bot.model.bluff_by_rank[Rank.TWO]
    assert rank_stats.alpha == 1.0 and rank_stats.beta == 1.0, \
        "pass polluted the rank-TWO model"
    claim_stats = bot.model.bluff_by_claim_size[1]
    assert claim_stats.alpha == 1.0 and claim_stats.beta == 1.0, \
        "pass polluted the claim-size model"
    # call_frequency DID get real evidence (human chose not to call)
    assert bot.model.total_actions_observed == 20
    cf = bot.model.call_frequency
    assert cf.beta >= 20  # 20 no-call updates


def test_real_play_still_updates_all_models():
    bot = BayesianBot()
    bot.observe_action(make_action(bluff=True, called=True, right=True), 9)
    assert bot.model.bluff_by_rank[Rank.SEVEN].alpha == 2.0
    assert bot.model.total_actions_observed == 1
    assert bot.bluff_tracker.total_caught == 0  # opponent's bluff, not ours


# ---------------------------------------------------------------------------
# GB-2 (offline half): serialization round-trip is lossless
# ---------------------------------------------------------------------------

def test_model_roundtrip_lossless():
    bot = BayesianBot()
    play_some_bluffy_games(bot, games=3)
    d = {
        "model": bot.model.to_dict(),
        "bluff_tracker": bot.bluff_tracker.to_dict(),
        "bluff_threshold": bot.bluff_threshold,
        "call_threshold": bot.call_threshold,
    }
    fresh = BayesianBot()
    fresh.model = OpponentModel.from_dict(d["model"])
    fresh.bluff_tracker = BluffTracker.from_dict(d["bluff_tracker"])
    assert fresh.model.total_actions_observed == bot.model.total_actions_observed
    a = fresh.model.estimate_bluff_probability(8, Rank.KING, 2)
    b = bot.model.estimate_bluff_probability(8, Rank.KING, 2)
    assert abs(a - b) < 1e-12
    assert abs(fresh.model.estimate_call_frequency()
               - bot.model.estimate_call_frequency()) < 1e-12


def test_loaded_bot_behaves_differently_than_cold_start():
    """The whole point: persistence changes behavior (GB-2 core claim)."""
    bluffy = BayesianBot()
    play_some_bluffy_games(bluffy, human_bluff_rate=0.8, games=5)
    d = {"model": bluffy.model.to_dict(),
         "bluff_tracker": bluffy.bluff_tracker.to_dict(),
         "bluff_threshold": bluffy.bluff_threshold,
         "call_threshold": bluffy.call_threshold}

    cold = BayesianBot()
    loaded = BayesianBot()
    loaded.model = OpponentModel.from_dict(d["model"])
    loaded.bluff_tracker = BluffTracker.from_dict(d["bluff_tracker"])

    probe = make_action(claimed=Rank.NINE, bluff=False)
    p_cold = cold.model.estimate_bluff_probability(8, Rank.NINE, 1)
    p_loaded = loaded.model.estimate_bluff_probability(8, Rank.NINE, 1)
    assert p_loaded > p_cold + 0.15, (
        f"loaded bot not more suspicious of a bluffy human: "
        f"{p_loaded:.3f} vs cold {p_cold:.3f}")
    # and the probe is unused in the call decision paths — sanity only
    assert probe is not None


# ---------------------------------------------------------------------------
# DB-backed tests (auto-skip without DATABASE_URL)
# ---------------------------------------------------------------------------

has_db = bool(os.environ.get("DATABASE_URL"))

@pytest.mark.skipif(not has_db, reason="DATABASE_URL not set")
def test_full_persistence_cycle_against_neon():
    """GB-i/ii/1/2 against live Neon: end-to-end cross-session cycle."""
    from db import pg

    async def run():
        user = str(uuid.uuid4())
        bot_id = "bayesian"

        # -- Identity first (same order as server.py db_start): ----------
        # game_sessions.player_id has an FK to users.id.
        await pg.ensure_user(user)

        # -- Session 1: bot learns a bluffy human, then saves ----------
        s1 = await pg.log_session_start(bot_id, player_id=user)
        assert s1 is not None, "log_session_start failed against Neon"
        await pg.set_model_loaded(s1, False)

        bot = BayesianBot()
        play_some_bluffy_games(bot, human_bluff_rate=0.8, games=4)
        model_data = {
            "model": bot.model.to_dict(),
            "bluff_tracker": bot.bluff_tracker.to_dict(),
            "bluff_threshold": bot.bluff_threshold,
            "call_threshold": bot.call_threshold,
        }
        await pg.save_opponent_model(user, bot_id, model_data)
        await pg.log_action(s1, 1, "human", "play",
                            cards_played=["9H"], claimed_rank=9,
                            was_bluff=True, decision_ms=1234)
        await pg.log_session_end(s1, "loss", 18, 95)
        # GB-i: finished_at must be set by the end update
        pool = await pg.get_pool()
        row = await pool.fetchrow(
            "SELECT finished_at, model_loaded, player_id FROM game_sessions "
            "WHERE id = $1", uuid.UUID(s1))
        assert row["finished_at"] is not None, "GB-i: finished_at not set"
        assert row["model_loaded"] is False
        assert row["player_id"] is not None, "player_id not bound"
        actions = await pool.fetch(
            "SELECT decision_ms, cards_played FROM actions "
            "WHERE game_id = $1", uuid.UUID(s1))
        assert len(actions) == 1, "GB-ii: action row missing"
        assert actions[0]["decision_ms"] == 1234

        # -- Session 2: fresh bot loads the model (cold vs loaded) ------
        s2 = await pg.log_session_start(bot_id, player_id=user)
        saved = await pg.load_opponent_model(user, bot_id)
        assert saved is not None, "GB-1: save→load round-trip failed"
        fresh = BayesianBot()
        cold = BayesianBot()
        fresh.model = OpponentModel.from_dict(saved["model"])
        fresh.bluff_tracker = BluffTracker.from_dict(saved["bluff_tracker"])
        await pg.set_model_loaded(s2, True)
        await pg.log_session_end(s2, "win", 12, 70)

        p_loaded = fresh.model.estimate_bluff_probability(8, Rank.NINE, 1)
        p_cold = cold.model.estimate_bluff_probability(8, Rank.NINE, 1)
        assert p_loaded > p_cold + 0.15, (
            f"GB-2 cross-session learning failed: loaded={p_loaded:.3f} "
            f"cold={p_cold:.3f}")
        row2 = await pool.fetchrow(
            "SELECT model_loaded FROM game_sessions WHERE id = $1",
            uuid.UUID(s2))
        assert row2["model_loaded"] is True
        return user

    user = asyncio.run(run())
    print(f"persistence cycle verified for user {user}")


@pytest.mark.skipif(not has_db, reason="DATABASE_URL not set")
def test_session_end_survives_missing_session():
    from db import pg
    # Best-effort contract: unknown session id must not raise
    asyncio.run(pg.log_session_end(str(uuid.uuid4()), "win", 5, 30))


def test_hybrid_bot_serialization_roundtrip():
    from bots.hybrid_bot import HybridBot
    bot = HybridBot()
    play_some_bluffy_games(bot, human_bluff_rate=0.75, games=3)
    d = bot.to_dict()
    assert "model" in d
    assert "bluff_tracker" in d
    assert "call_threshold" in d

    loaded = HybridBot.from_dict(d)
    assert loaded.model.total_actions_observed == bot.model.total_actions_observed
    assert abs(loaded.model.estimate_call_frequency() - bot.model.estimate_call_frequency()) < 1e-9
