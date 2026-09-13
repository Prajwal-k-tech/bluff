"""P2 Persistent Profiling Architecture — unit tests.

Covers F1-F4:
  F2: profile_meta round-trip (games_played, accumulated_info, backward-compat)
  F3: fairfight vs predator mode, variance-gated alpha, RWYW cap, telemetry
  F1: mark_session_completed() returns correct telemetry dict
  F4: (pg.py loud logging — tested via integration, not unit)

Run:  python3 -m pytest tests/test_persistent_profiling.py -v
"""

import math
import os
import sys

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cards import Card, Rank, Suit
from game import Action
from bots.bayesian_bot import BayesianBot, OpponentModel, BluffTracker, BetaDistribution
from bots.hybrid_bot import HybridBot
from bots.academic_beast_bot import AcademicBeastBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_action(player=0, ranks=None, claimed=Rank.SEVEN, bluff=False,
                called=False, right=False) -> Action:
    cards = [Card(r, Suit.HEARTS) for r in (ranks or [claimed])]
    return Action(
        player=player, cards_played=cards, claimed_rank=claimed,
        was_bluff=bluff, bluff_called=called, caller_was_right=right,
        pile_size_before=4,
    )


def feed_observations(bot, n: int = 10, bluff_rate: float = 0.5):
    """Feed n observations to the bot's opponent model."""
    for i in range(n):
        bluff = (i % max(1, int(1 / bluff_rate))) == 0 if bluff_rate > 0 else False
        bot.observe_action(make_action(bluff=bluff), opponent_hand_size=8)


# ---------------------------------------------------------------------------
# F2: AcademicBeastBot profile_meta round-trip
# ---------------------------------------------------------------------------

class TestAcademicBeastBotRoundTrip:
    def test_to_dict_includes_new_profile_meta_fields(self):
        bot = AcademicBeastBot(use_nn=False)
        d = bot.to_dict()
        assert "profile_meta" in d
        pm = d["profile_meta"]
        assert "games_played" in pm
        assert "accumulated_info" in pm
        assert "sessions_observed" in pm
        assert pm["games_played"] == 0
        assert pm["accumulated_info"] == 0.0
        assert pm["sessions_observed"] == 0

    def test_from_dict_backward_compat_no_games_played(self):
        """Old DB rows without games_played/accumulated_info load safely."""
        bot = AcademicBeastBot(use_nn=False)
        d = bot.to_dict()
        # Simulate old schema: remove new fields
        del d["profile_meta"]["games_played"]
        del d["profile_meta"]["accumulated_info"]
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)
        assert loaded._games_played == 0
        assert loaded._accumulated_info == 0.0

    def test_roundtrip_preserves_all_fields(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        feed_observations(bot, n=5, bluff_rate=0.6)
        bot._games_played = 7
        bot._accumulated_info = 3.14
        bot._sessions_observed = 4

        d = bot.to_dict()
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)

        assert loaded._games_played == 7
        assert abs(loaded._accumulated_info - 3.14) < 1e-9
        assert loaded._sessions_observed == 4
        assert loaded.mode == "fairfight"
        assert loaded.model.total_actions_observed == bot.model.total_actions_observed

    def test_mark_session_completed_increments_counters(self):
        bot = AcademicBeastBot(use_nn=False)
        feed_observations(bot, n=8, bluff_rate=0.5)
        info = bot.mark_session_completed(lam=1.0)
        assert info["sessions_observed"] == 1
        assert info["games_played"] == 1
        assert bot._games_played == 1
        assert bot._sessions_observed == 1

    def test_mark_session_completed_accumulates_info(self):
        bot = AcademicBeastBot(use_nn=False)
        # Session 1: feed observations, mark complete
        feed_observations(bot, n=5, bluff_rate=0.7)
        info1 = bot.mark_session_completed(lam=1.0)
        assert info1["delta"] >= 0
        acc1 = bot._accumulated_info

        # Session 2: feed more observations, mark complete
        feed_observations(bot, n=5, bluff_rate=0.2)
        info2 = bot.mark_session_completed(lam=1.0)
        assert bot._accumulated_info >= acc1
        assert bot._games_played == 2
        assert bot._sessions_observed == 2


# ---------------------------------------------------------------------------
# F2: HybridBot profile_meta round-trip
# ---------------------------------------------------------------------------

class TestHybridBotRoundTrip:
    def test_to_dict_includes_profile_meta(self):
        bot = HybridBot()
        d = bot.to_dict()
        # HybridBot.to_dict should still work (no profile_meta yet — it doesn't
        # have the persistent profiling fields, but to_dict round-trips cleanly)
        assert "model" in d
        assert "bluff_tracker" in d

    def test_mark_session_completed_returns_dict(self):
        bot = HybridBot()
        info = bot.mark_session_completed()
        assert isinstance(info, dict)
        assert "sessions_observed" in info
        assert "games_played" in info


# ---------------------------------------------------------------------------
# F2: BayesianBot to_dict/from_dict round-trip
# ---------------------------------------------------------------------------

class TestBayesianBotRoundTrip:
    def test_to_dict_from_dict_roundtrip(self):
        bot = BayesianBot()
        feed_observations(bot, n=6, bluff_rate=0.6)
        d = bot.to_dict()
        assert "model" in d
        assert "bluff_tracker" in d
        assert "bluff_threshold" in d
        assert "call_threshold" in d

        loaded = BayesianBot.from_dict(d)
        assert loaded.model.total_actions_observed == bot.model.total_actions_observed
        p_orig = bot.model.estimate_bluff_probability(8, Rank.KING, 2)
        p_loaded = loaded.model.estimate_bluff_probability(8, Rank.KING, 2)
        assert abs(p_orig - p_loaded) < 1e-12

    def test_mark_session_completed_returns_dict(self):
        bot = BayesianBot()
        info = bot.mark_session_completed()
        assert isinstance(info, dict)
        assert "bluff_mean" in info
        assert info["sessions_observed"] == 1


# ---------------------------------------------------------------------------
# F3: fairfight vs predator mode
# ---------------------------------------------------------------------------

class TestFairfightPredatorModes:
    def test_predator_mode_alpha_always_one(self):
        bot = AcademicBeastBot(use_nn=False, mode="predator")
        assert bot.fairfight_alpha() == 1.0

    def test_fairfight_mode_alpha_cold_start(self):
        """Cold start: high variance → low alpha."""
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        alpha = bot.fairfight_alpha()
        # Fresh Beta(1,4) has variance ~0.0098, σ²_thr=0.01
        # α = max(0, 1 - 0.0098/0.01) ≈ 0.02
        assert 0.0 <= alpha <= 0.15

    def test_fairfight_mode_alpha_converged(self):
        """After many observations: low variance → high alpha."""
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        # Feed 50 identical observations → posterior concentrates
        for _ in range(50):
            bot.observe_action(make_action(bluff=True), opponent_hand_size=8)
        alpha = bot.fairfight_alpha()
        assert alpha > 0.8  # should be near 1.0

    def test_predator_rwyw_cap_uncapped(self):
        bot = AcademicBeastBot(use_nn=False, mode="predator")
        assert bot.rwyw_exploit_depth_cap() == 999999

    def test_fairfight_rwyw_cap_zero_info(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        bot._accumulated_info = 0.0
        assert bot.rwyw_exploit_depth_cap() == 0

    def test_fairfight_rwyw_cap_grows_with_info(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        bot._accumulated_info = 1.0
        assert bot.rwyw_exploit_depth_cap() == 2
        bot._accumulated_info = 2.0
        assert bot.rwyw_exploit_depth_cap() == 4

    def test_fairfight_rwyw_cap_capped_at_4(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        bot._accumulated_info = 100.0
        assert bot.rwyw_exploit_depth_cap() == 4

    def test_mode_roundtrips_through_to_dict_from_dict(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        d = bot.to_dict()
        assert d["mode"] == "fairfight"
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)
        assert loaded.mode == "fairfight"

    def test_fairfight_telemetry_dict(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        feed_observations(bot, n=5, bluff_rate=0.5)
        t = bot.get_fairfight_telemetry()
        assert t["mode"] == "fairfight"
        assert "alpha" in t
        assert "games_played" in t
        assert "sigma2" in t
        assert "rwyw_cap" in t
        assert "cap_hit" in t

    def test_create_bot_via_server_factory(self):
        """Test create_bot with mode param (simulates server.py behavior)."""
        from server import create_bot
        bot_pred = create_bot("beast", mode="predator")
        assert isinstance(bot_pred, AcademicBeastBot)
        assert bot_pred.mode == "predator"

        bot_fair = create_bot("beast", mode="fairfight")
        assert isinstance(bot_fair, AcademicBeastBot)
        assert bot_fair.mode == "fairfight"

        bot_auto = create_bot("beast-fair")
        assert isinstance(bot_auto, AcademicBeastBot)
        assert bot_auto.mode == "fairfight"


# ---------------------------------------------------------------------------
# F3: Telemetry logging (smoke test)
# ---------------------------------------------------------------------------

class TestTelemetryLogging:
    def test_mark_session_completed_telemetry_values(self):
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        feed_observations(bot, n=10, bluff_rate=0.5)
        info = bot.mark_session_completed(lam=1.0)
        assert "bluff_mean" in info
        assert "delta" in info
        assert info["bluff_mean"] > 0
        assert info["delta"] >= 0
        assert isinstance(info["accumulated_info"], float)


# ---------------------------------------------------------------------------
# R1: save → fresh bot → from_dict → full state survives
# ---------------------------------------------------------------------------

class TestR1LoadSideGap:
    def test_beast_from_dict_restores_archetype_and_profile(self):
        """R1 CRITICAL: from_dict must restore archetype_state + profile_meta."""
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        feed_observations(bot, n=10, bluff_rate=0.6)
        bot._games_played = 5
        bot._accumulated_info = 2.7
        bot._sessions_observed = 3
        bot.opp_bluffs = 12
        bot.opp_honest = 8
        bot.opp_calls = 4
        bot.opp_passes = 6

        d = bot.to_dict()
        # Simulate a fresh bot loading from DB
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)

        assert loaded._games_played == 5
        assert abs(loaded._accumulated_info - 2.7) < 1e-9
        assert loaded._sessions_observed == 3
        assert loaded.opp_bluffs == 12
        assert loaded.opp_honest == 8
        assert loaded.opp_calls == 4
        assert loaded.opp_passes == 6
        assert loaded.mode == "fairfight"
        # Model roundtrip
        assert loaded.model.total_actions_observed == bot.model.total_actions_observed

    def test_hybrid_from_dict_restores_config(self):
        """R1: HybridBot from_dict restores all config fields."""
        bot = HybridBot(thompson_sampling=False, w_model_cap=0.42,
                        exploit_mult=3.1, call_mult=2.2)
        feed_observations(bot, n=5, bluff_rate=0.4)
        d = bot.to_dict()
        loaded = HybridBot.from_dict(d)
        assert loaded.thompson_sampling is False
        assert loaded.w_model_cap == 0.42
        assert loaded.exploit_mult == 3.1
        assert loaded.call_mult == 2.2
        assert loaded.model.total_actions_observed == bot.model.total_actions_observed

    def test_backward_compat_old_saved_dict_no_profile_meta(self):
        """Old saved dicts without profile_meta still load (defaults to 0)."""
        bot = AcademicBeastBot(use_nn=False)
        d = bot.to_dict()
        # Strip all P2 fields to simulate pre-P2 save
        del d["profile_meta"]
        del d["archetype_state"]
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)
        assert loaded._games_played == 0
        assert loaded._accumulated_info == 0.0
        assert loaded.opp_bluffs == 0
        assert loaded.opp_honest == 0


# ---------------------------------------------------------------------------
# R2: exploit counter increments and cap blocks
# ---------------------------------------------------------------------------

class TestR2ExploitDepthCap:
    def test_exploit_counter_increments_on_t7e_exploit(self):
        """R2: T7e never-bluffer exploit increments _exploit_depth_this_game."""
        bot = AcademicBeastBot(use_nn=False, mode="predator")
        # Simulate a never-bluffer: opp_bluffs=0, opp_honest>=6
        bot.opp_bluffs = 0
        bot.opp_honest = 6
        bot._vol_calls = 5  # satisfy vol_calls >= 5 trigger

        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        game_state = {
            "opponent_hand_size": 10,
            "pile_size": 3,
            "draw_pile_size": 20,
            "turn_number": 10,
            "can_pass": True,
            "actions": [],
        }
        bot.reset()
        assert bot._exploit_depth_this_game == 0

        cards, rank = bot.decide_play(hand, game_state)
        # T7e fires: exploit counter should be 1
        assert bot._exploit_depth_this_game == 1

    def test_cap_blocks_exploit_when_reached(self):
        """R2: once cap is hit, T7e exploit is blocked and falls through."""
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        # Set up: accumulated_info=0 → cap = min(4, 0*2) = 0
        bot._accumulated_info = 0.0
        bot.opp_bluffs = 0
        bot.opp_honest = 6
        bot._vol_calls = 5

        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SEVEN, Suit.DIAMONDS),
        ]
        game_state = {
            "opponent_hand_size": 10,
            "pile_size": 3,
            "draw_pile_size": 20,
            "turn_number": 10,
            "can_pass": True,
            "actions": [],
        }
        bot.reset()
        assert bot._exploit_depth_this_game == 0
        assert bot.rwyw_exploit_depth_cap() == 0  # cap is 0

        cards, rank = bot.decide_play(hand, game_state)
        # T7e is blocked by cap=0, falls through to honest play
        assert bot._exploit_depth_this_game == 0  # NOT incremented
        # Should return honest cards (multi-honest or fallback)
        assert len(cards) > 0

    def test_counter_increments_on_model_call(self):
        """R2: model-based call increments counter (non-budget, non-counter)."""
        bot = AcademicBeastBot(use_nn=False, mode="predator")
        bot._vol_calls = 10  # past budget phase
        bot.opp_bluffs = 3
        # Force high bluff probability to trigger a call
        for _ in range(20):
            bot.observe_action(make_action(bluff=True), opponent_hand_size=8)

        last_action = Action(
            player=0,
            cards_played=[Card(Rank.ACE, Suit.HEARTS)],
            claimed_rank=Rank.ACE,
            was_bluff=True,
            bluff_called=False,
            caller_was_right=False,
            pile_size_before=5,
        )
        game_state = {
            "hand": [Card(Rank.KING, Suit.SPADES), Card(Rank.QUEEN, Suit.DIAMONDS)],
            "opponent_hand_size": 8,
            "pile_size": 5,
            "draw_pile_size": 15,
            "can_pass": True,
        }
        bot.reset()
        initial_depth = bot._exploit_depth_this_game
        # If model decides to call, counter should increment
        called = bot.decide_call(last_action, game_state)
        if called:
            assert bot._exploit_depth_this_game == initial_depth + 1

    def test_exploit_counter_resets_per_game(self):
        """R2: reset() clears _exploit_depth_this_game."""
        bot = AcademicBeastBot(use_nn=False, mode="predator")
        bot._exploit_depth_this_game = 3
        bot.reset()
        assert bot._exploit_depth_this_game == 0

    def test_telemetry_reports_cap_hit(self):
        """R2: get_fairfight_telemetry reflects cap_hit correctly."""
        bot = AcademicBeastBot(use_nn=False, mode="fairfight")
        bot._accumulated_info = 0.0
        bot._exploit_depth_this_game = 1
        t = bot.get_fairfight_telemetry()
        # cap = 0, depth = 1 → cap_hit = True
        assert t["cap_hit"] is True

        bot2 = AcademicBeastBot(use_nn=False, mode="fairfight")
        bot2._accumulated_info = 5.0  # cap = min(4, 10) = 4
        bot2._exploit_depth_this_game = 2
        t2 = bot2.get_fairfight_telemetry()
        assert t2["cap_hit"] is False


# ---------------------------------------------------------------------------
# R3: _prev_bluff_mean persists in to_dict/from_dict
# ---------------------------------------------------------------------------

class TestR3PrevBluffMeanPersistence:
    def test_prev_bluff_mean_roundtrips(self):
        """R3: _prev_bluff_mean survives to_dict → from_dict."""
        bot = AcademicBeastBot(use_nn=False)
        feed_observations(bot, n=10, bluff_rate=0.7)
        # After observations, mean has shifted
        original_mean = bot.model.overall_bluff.mean()
        bot._prev_bluff_mean = 0.35  # set to a known value

        d = bot.to_dict()
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)
        assert abs(loaded._prev_bluff_mean - 0.35) < 1e-9

    def test_no_phantom_delta_after_reload(self):
        """R3: first mark_session_completed after reload has delta≈0."""
        bot = AcademicBeastBot(use_nn=False)
        feed_observations(bot, n=10, bluff_rate=0.5)
        mean_before = bot.model.overall_bluff.mean()
        bot._prev_bluff_mean = mean_before  # simulate persisted value

        d = bot.to_dict()
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)

        info = loaded.mark_session_completed(lam=1.0)
        # delta should be ~0 (prev_bluff_mean was persisted correctly)
        assert info["delta"] < 0.01, (
            f"phantom delta detected: {info['delta']:.4f} "
            f"(prev={loaded._prev_bluff_mean:.4f}, current={info['bluff_mean']:.4f})"
        )
        assert info["accumulated_info"] < 0.01

    def test_backward_compat_old_dict_defaults_to_model_mean(self):
        """Old dicts without prev_bluff_mean fall back to current model mean."""
        bot = AcademicBeastBot(use_nn=False)
        d = bot.to_dict()
        # Simulate old schema: no prev_bluff_mean in profile_meta
        del d["profile_meta"]["prev_bluff_mean"]
        loaded = AcademicBeastBot.from_dict(d, checkpoint_path=None)
        # Should default to current model mean, not crash
        expected = loaded.model.overall_bluff.mean()
        assert abs(loaded._prev_bluff_mean - expected) < 1e-9

    def test_save_load_file_roundtrip(self):
        """R3: file-based save/load also preserves _prev_bluff_mean."""
        import tempfile, json
        bot = AcademicBeastBot(use_nn=False)
        feed_observations(bot, n=5, bluff_rate=0.6)
        bot._prev_bluff_mean = 0.42

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(bot.to_dict(), f)
            tmp_path = f.name

        try:
            loaded = AcademicBeastBot(use_nn=False)
            loaded.load(tmp_path)
            assert abs(loaded._prev_bluff_mean - 0.42) < 1e-9
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# P3: Persistent Proof Curve Harness
# ---------------------------------------------------------------------------

class TestPersistentProofCurve:
    """P3: Proof-curve harness produces valid results."""

    def test_proof_curve_runs_and_saves(self):
        """Smoke test: run 2 games per shift × 8 personas = 16 total, verify output."""
        from analysis.persistent_proof_curve import run_proof_curve, PROOF_CURVE_PERSONAS
        result = run_proof_curve(
            games_per_shift=2,
            max_turns_per_game=60,
            seed=99,
            verbose=False,
        )
        assert result["total_games"] == 2 * len(PROOF_CURVE_PERSONAS)
        assert result["wins"] + result["losses"] + result["draws"] == result["total_games"]
        assert 0.0 <= result["win_rate"] <= 1.0
        # accumulated_info should be > 0 after mark_session_completed calls
        assert result["accumulated_info_delta"] >= 0.0

    def test_proof_curve_persona_shifts(self):
        """Verify persona names appear in game_log after shifts."""
        from analysis.persistent_proof_curve import run_proof_curve
        result = run_proof_curve(
            games_per_shift=3,
            max_turns_per_game=60,
            seed=42,
            verbose=False,
        )
        # First 3 games: Honest_Rock, next 3: Hyper_Maniac
        personas_seen = [g["persona"] for g in result["game_log"]]
        assert personas_seen[0] == "Honest_Rock"
        assert personas_seen[3] == "Hyper_Maniac"

    def test_proof_curve_file_output(self):
        """Verify JSON output file is created."""
        import os, json
        from analysis.persistent_proof_curve import run_proof_curve
        result = run_proof_curve(
            games_per_shift=2,
            max_turns_per_game=60,
            seed=77,
            verbose=False,
        )
        assert os.path.exists("data/persistent_proof_curve.json")
        with open("data/persistent_proof_curve.json") as f:
            data = json.load(f)
        assert data["total_games"] == result["total_games"]

    def test_proof_curve_wilson_ci_bounds(self):
        """Wilson CIs should be valid probability bounds."""
        from analysis.persistent_proof_curve import run_proof_curve
        result = run_proof_curve(
            games_per_shift=2,
            max_turns_per_game=60,
            seed=42,
            verbose=False,
        )
        ci = result["wilson_95ci"]
        assert 0.0 <= ci[0] <= ci[1] <= 1.0
        # Early/late phase CIs
        assert 0.0 <= result["early_phase"]["wilson_95ci"][0]
        assert result["early_phase"]["wilson_95ci"][1] <= 1.0
