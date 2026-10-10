"""Equivalent visible histories must not expose different hidden worlds."""

import random

import pytest

from bots.base import public_action_for_viewer
from bots.round_observation import round_player_view
from bots.round_policy import (
    RoundAdaptiveBot, RoundHonestBot, RoundMathBot, RoundRandomBot,
)
from cards import Card, Hand, Rank, Suit
from game_v2 import GameState


def hidden_worlds(viewer):
    """Swap a truthful/false claimed card with supply, preserving all counts."""
    truthful = Card(Rank.SEVEN, Suit.HEARTS)
    false = Card(Rank.TWO, Suit.HEARTS)
    rest = [Card(rank, suit) for suit in Suit for rank in Rank
            if Card(rank, suit) not in (truthful, false)]
    games = []
    rng_state = random.getstate()
    try:
        for played, supplied in ((truthful, false), (false, truthful)):
            game = GameState()
            game.hands[viewer] = Hand(rest[:14])
            game.hands[1 - viewer] = Hand([played] + rest[14:27])
            game.draw_pile = [supplied] + rest[27:]
            game.current_player = 1 - viewer
            assert game.play_cards(1 - viewer, [played], Rank.SEVEN)[0]
            cards = (game.hands[0].cards + game.hands[1].cards
                     + game.pile + game.draw_pile)
            assert len(cards) == len(set(cards)) == 52
            assert len(game.draw_pile) == 24
            assert not game.game_over
            games.append(game)
    finally:
        random.setstate(rng_state)
    return games


def visible_snapshot(view):
    # Compare every public action field, including dynamically added metadata,
    # rather than relying on Action object identity or a partial field list.
    return {
        **view,
        "actions": [vars(action) for action in view["actions"]],
        "last_action": vars(view["last_action"]) if view["last_action"] else None,
    }


@pytest.mark.parametrize("viewer", [0, 1])
@pytest.mark.parametrize("bot_type", [
    RoundRandomBot, RoundHonestBot, RoundMathBot, RoundAdaptiveBot,
])
def test_equivalent_hidden_worlds_have_identical_observations_and_decisions(viewer, bot_type):
    games = hidden_worlds(viewer)
    assert [game.last_action.was_bluff for game in games] == [False, True]
    views = [round_player_view(game, viewer) for game in games]
    assert visible_snapshot(views[0]) == visible_snapshot(views[1])
    choices, profiles, telemetry = [], [], []
    for game, view in zip(games, views):
        bot = bot_type(seed=79)
        bot.player_id = viewer
        if isinstance(bot, RoundAdaptiveBot):
            # Fairness must also hold with an already learned profile.
            bot._call_evidence["1|1-2"] = [2, 5]
            bot._bluff_evidence["1|1-2"] = [3, 4]
        before = bot.to_dict()
        bot.observe_action(view["last_action"], game.hands[1 - viewer].size())
        assert bot.to_dict() == before  # unresolved labels are not evidence
        choices.append(bot.choose_action(list(game.hands[viewer].cards), view))
        profiles.append(bot.to_dict())
        telemetry.append(bot.last_decision_telemetry)
    assert choices[0] == choices[1]
    assert profiles[0] == profiles[1]
    assert telemetry[0] == telemetry[1]


@pytest.mark.parametrize("viewer", [0, 1])
def test_legal_challenge_reveal_distinguishes_worlds_and_learns_once(viewer):
    profiles = []
    for game in hidden_worlds(viewer):
        bot = RoundAdaptiveBot(seed=79)
        bot.player_id = viewer
        before = public_action_for_viewer(game.last_action, viewer)
        bot.observe_action(before, game.hands[1 - viewer].size())
        assert bot.to_dict()["bluff_evidence"] == {}
        ok, _, resolved = game.call_bluff(viewer)
        assert ok and resolved.response_was_forced is False
        after = public_action_for_viewer(resolved, viewer)
        assert after.cards_played == resolved.cards_played
        assert after.was_bluff is resolved.was_bluff
        bot.observe_action(after, game.hands[1 - viewer].size())
        profile = bot.to_dict()
        # Both a duplicate reveal and a stale pending copy must be harmless.
        bot.observe_action(public_action_for_viewer(resolved, viewer), 0)
        bot.observe_action(before, 0)
        assert bot.to_dict() == profile
        profiles.append(profile["bluff_evidence"])
    assert profiles == [
        {"1|1-2": {"successes": 0, "failures": 1}},
        {"1|1-2": {"successes": 1, "failures": 0}},
    ]
