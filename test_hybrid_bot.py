"""Unit tests for HybridBot, Thompson Sampling, and Opponent Modeling."""

import os
import pytest
from cards import Card, Rank, Suit
from game import Action, GameState
from bots.bayesian_bot import BetaDistribution, OpponentModel
from bots.hybrid_bot import HybridBot


class TestThompsonSampling:
    def test_beta_sample_bounds(self):
        dist = BetaDistribution(alpha=2.0, beta=5.0)
        for _ in range(50):
            val = dist.sample()
            assert 0.0 <= val <= 1.0

    def test_opponent_model_sampling(self):
        model = OpponentModel()
        for _ in range(20):
            p = model.sample_bluff_probability(hand_size=10, claimed_rank=Rank.ACE, claim_size=2)
            assert 0.0 <= p <= 1.0
            call_f = model.sample_call_frequency()
            assert 0.0 <= call_f <= 1.0

    def test_posterior_variance_preserves_exploration(self):
        dist = BetaDistribution(alpha=1.0, beta=1.0)
        samples = [dist.sample() for _ in range(30)]
        assert min(samples) < 0.3
        assert max(samples) > 0.7
        assert len(set(samples)) == 30


class TestHybridBot:
    def test_hybrid_bot_initialization(self):
        bot = HybridBot()
        assert bot.thompson_sampling is True
        assert bot.model is not None
        assert bot.counter is not None

    def test_decide_play_returns_valid_cards(self):
        bot = HybridBot()
        hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES), Card(Rank.ACE, Suit.DIAMONDS)]
        game_state = {
            "opponent_hand_size": 10,
            "pile_size": 2,
            "draw_pile_size": 20,
            "turn_number": 3,
            "can_pass": True,
            "actions": [],
        }
        cards, claimed_rank = bot.decide_play(hand, game_state)
        assert len(cards) > 0
        assert isinstance(claimed_rank, Rank)
        for c in cards:
            assert c in hand

    def test_decide_call_returns_bool(self):
        bot = HybridBot()
        action = Action(
            player=1,
            cards_played=[Card(Rank.QUEEN, Suit.HEARTS)],
            claimed_rank=Rank.QUEEN,
            was_bluff=False,
            bluff_called=False,
            caller_was_right=False,
            pile_size_before=2,
        )
        game_state = {
            "hand": [Card(Rank.ACE, Suit.HEARTS), Card(Rank.QUEEN, Suit.SPADES)],
            "opponent_hand_size": 10,
            "pile_size": 3,
            "draw_pile_size": 18,
            "can_pass": True,
        }
        call_decision = bot.decide_call(action, game_state)
        assert isinstance(call_decision, bool)

    def test_serialization_round_trip(self):
        bot = HybridBot(thompson_sampling=False)
        bot.bluff_threshold = 0.62
        bot.call_threshold = 0.48
        act = Action(
            player=1,
            cards_played=[Card(Rank.ACE, Suit.HEARTS)],
            claimed_rank=Rank.ACE,
            was_bluff=True,
            bluff_called=True,
            caller_was_right=True,
            pile_size_before=4,
        )
        bot.observe_action(act, opponent_hand_size=8)

        saved_dict = bot.to_dict()
        assert saved_dict["thompson_sampling"] is False
        assert saved_dict["bluff_threshold"] == 0.62

        restored_bot = HybridBot.from_dict(saved_dict)
        assert restored_bot.thompson_sampling is False
        assert restored_bot.bluff_threshold == 0.62
        assert restored_bot.call_threshold == 0.48
        assert restored_bot.model.total_actions_observed == 1

    def test_card_counter_resets_intra_game(self):
        bot = HybridBot()
        bot.counter.update_with_play([Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.SPADES)])
        assert bot.counter.remaining_by_rank[Rank.ACE] == 2
        # Reset must clean the card counter but preserve the learned opponent model
        bot.reset()
        assert bot.counter.remaining_by_rank[Rank.ACE] == 4
