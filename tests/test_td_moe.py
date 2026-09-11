"""Unit tests for Temporal Decaying Mixture of Experts (TD-MoE).

Verifies:
1. Mathematical properties of the schedule: w_nn(n) + w_bayes(n) == 1.0, monotonic decay.
2. Half-life decay calibration across different tau values.
3. Integration in AcademicBeastBot and HybridBot.
4. Calling Station bluff suppression as w_bayes matures.
5. Serialization round-trip preserving TD-MoE parameters.
"""

import math
import pytest
from cards import Card, Rank, Suit
from game import Action
from bots.academic_beast_bot import AcademicBeastBot
from bots.hybrid_bot import HybridBot


class TestTDMoESchedule:
    def test_monotonic_decay_and_complementarity(self):
        bot = AcademicBeastBot(decay_tau=8.0, nn_floor=0.15)
        prev_w_nn = 1.05

        for n in range(0, 40):
            bot.model.total_actions_observed = n
            w_nn, w_bayes = bot.get_td_moe_weights()

            # Complementarity
            assert math.isclose(w_nn + w_bayes, 1.0, rel_tol=1e-6)

            # Monotonicity
            assert w_nn <= prev_w_nn + 1e-9
            prev_w_nn = w_nn

            # Bounds
            assert 0.15 - 1e-6 <= w_nn <= 1.0 + 1e-6
            assert 0.0 - 1e-6 <= w_bayes <= 0.85 + 1e-6

        # Initial condition (n=0): pure NN prior dominance
        bot.model.total_actions_observed = 0
        w_nn_0, w_bayes_0 = bot.get_td_moe_weights()
        assert math.isclose(w_nn_0, 1.0, rel_tol=1e-6)
        assert math.isclose(w_bayes_0, 0.0, abs_tol=1e-6)

        # Asymptotic condition (n=50): Bayesian exploitation dominance
        bot.model.total_actions_observed = 50
        w_nn_50, w_bayes_50 = bot.get_td_moe_weights()
        assert math.isclose(w_nn_50, 0.15, abs_tol=0.01)
        assert math.isclose(w_bayes_50, 0.85, abs_tol=0.01)

    def test_half_life_calibration(self):
        """At n = tau, the decayed portion should equal 1 - 1/e ≈ 63.2%."""
        for tau in [4.0, 8.0, 16.0]:
            bot = AcademicBeastBot(decay_tau=tau, nn_floor=0.15)
            bot.model.total_actions_observed = int(tau)
            w_nn, _ = bot.get_td_moe_weights()
            expected_w_nn = 0.15 + 0.85 * math.exp(-1.0)
            assert math.isclose(w_nn, expected_w_nn, rel_tol=1e-3)

    def test_hybrid_bot_schedule_parity(self):
        """Verify HybridBot exhibits identical TD-MoE schedule mechanics."""
        bot = HybridBot(decay_tau=10.0, nn_floor=0.20)
        bot.model.total_actions_observed = 10
        w_nn, w_bayes = bot.get_td_moe_weights()
        assert math.isclose(w_nn + w_bayes, 1.0, rel_tol=1e-6)
        expected_w_nn = 0.20 + 0.80 * math.exp(-1.0)
        assert math.isclose(w_nn, expected_w_nn, rel_tol=1e-3)


class TestTDMoEBotIntegration:
    def test_academic_beast_decide_play(self):
        bot = AcademicBeastBot(decay_tau=8.0)
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS), Card(Rank.KING, Suit.SPADES)]
        game_state = {
            "opponent_hand_size": 10,
            "pile_size": 2,
            "draw_pile_size": 20,
            "turn_number": 2,
            "can_pass": True,
            "actions": [],
        }
        cards, claimed_rank = bot.decide_play(hand, game_state)
        assert len(cards) > 0
        assert isinstance(claimed_rank, Rank)
        for c in cards:
            assert c in hand

    def test_academic_beast_decide_call(self):
        bot = AcademicBeastBot(decay_tau=8.0)
        action = Action(
            player=1,
            cards_played=[Card(Rank.TEN, Suit.HEARTS)],
            claimed_rank=Rank.TEN,
            was_bluff=False,
            bluff_called=False,
            caller_was_right=False,
            pile_size_before=2,
        )
        game_state = {
            "hand": [Card(Rank.ACE, Suit.HEARTS)],
            "opponent_hand_size": 10,
            "pile_size": 3,
            "draw_pile_size": 18,
            "can_pass": True,
            "actions": [action],
        }
        res = bot.decide_call(action, game_state)
        assert isinstance(res, bool)

    def test_calling_station_bluff_suppression_when_mature(self):
        """When w_bayes is mature and opponent is confirmed Calling Station, bot never bluffs."""
        bot = AcademicBeastBot(decay_tau=8.0)
        bot.player_id = 0
        # Feed 12 actions where our play was called by opponent
        for _ in range(12):
            call_act = Action(
                player=0, cards_played=[Card(Rank.ACE, Suit.HEARTS)], claimed_rank=Rank.ACE,
                was_bluff=False, bluff_called=True, caller_was_right=False,
                pile_size_before=1,
            )
            bot.observe_action(call_act, opponent_hand_size=10)

        assert bot.model.total_actions_observed >= 10
        assert bot.opp_calls >= 10
        top_arch, conf = bot.classifier.top_archetype(
            bot.opp_bluffs, bot.opp_honest, bot.opp_calls, bot.opp_passes
        )
        assert top_arch == "Calling_Station"
        assert conf >= 0.50

        _, w_bayes = bot.get_td_moe_weights()
        assert w_bayes > 0.50

        hand = [Card(Rank.TWO, Suit.HEARTS), Card(Rank.THREE, Suit.SPADES)]
        game_state = {
            "opponent_hand_size": 10,
            "pile_size": 4,
            "draw_pile_size": 16,
            "turn_number": 15,
            "can_pass": True,
            "actions": [],
        }
        cards, claimed_rank = bot.decide_play(hand, game_state)
        # Must play honestly
        assert all(c.rank == claimed_rank for c in cards)

    def test_td_moe_serialization_round_trip(self):
        bot = AcademicBeastBot(decay_tau=12.0, nn_floor=0.18)
        saved = bot.to_dict()
        assert saved["decay_tau"] == 12.0
        assert saved["nn_floor"] == 0.18

        restored = AcademicBeastBot.from_dict(saved)
        assert restored.decay_tau == 12.0
        assert restored.nn_floor == 0.18
