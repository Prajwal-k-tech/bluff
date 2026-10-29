from copy import deepcopy

import pytest

from bots.base import public_action_for_viewer, public_bluff_label
from bots.round_card_memory import opponent_rank_bounds
from bots.round_ismcts import information_state_key
from bots.round_observation import round_player_view
from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
from bots.round_private_pickup_evidence import extract_private_pickup_evidence
from bots.round_policy import RoundAdaptiveBot
from cards import Card, Hand, Rank, Suit
from game_v2 import GameState, RULESET_ID


def _card(rank, suit):
    return Card(rank, suit)


def _pile(latest_bluff=False, earlier_bluff=False):
    """Two ordered packets; latest truth alone selects who takes both."""
    game = GameState()
    old = _card(Rank.TWO if earlier_bluff else Rank.SEVEN, Suit.CLUBS)
    latest = _card(Rank.TWO if latest_bluff else Rank.SEVEN, Suit.DIAMONDS)
    game.hands = [Hand([latest, _card(Rank.KING, Suit.HEARTS)]),
                  Hand([old, _card(Rank.QUEEN, Suit.SPADES)])]
    game.current_player = 1
    assert game.play_cards(1, [old], Rank.SEVEN)[0]
    assert game.play_cards(0, [latest], Rank.SEVEN)[0]
    return game, old, latest


@pytest.mark.parametrize("latest_bluff,earlier_bluff", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_challenge_reveals_ordered_packets_to_both_but_latest_alone_resolves(
        latest_bluff, earlier_bluff):
    game, old, latest = _pile(latest_bluff, earlier_bluff)
    assert game.call_bluff(1)[0]

    assert [action.cards_revealed for action in game.actions] == [True, True]
    for viewer in (0, 1):
        view = round_player_view(game, viewer)
        assert [action.cards_played for action in view["actions"]] == [[old], [latest]]
        assert [public_bluff_label(action, viewer) for action in view["actions"]] == [
            earlier_bluff, latest_bluff]
        assert [public_action_for_viewer(action, viewer).cards_played
                for action in game.actions] == [[old], [latest]]

    loser = 0 if latest_bluff else 1
    assert old in game.hands[loser].cards
    assert latest in game.hands[loser].cards
    assert (old in game.hands[1-loser].cards) is False
    assert (latest in game.hands[1-loser].cards) is False
    assert game.actions[0].bluff_called is False
    assert game.actions[1].bluff_called is True
    assert RULESET_ID == "bluff-public-pile-v3"


@pytest.mark.parametrize("truthful", [True, False])
def test_automatic_final_challenge_reveals_whole_ordered_pile(truthful):
    game = GameState()
    old = _card(Rank.SEVEN, Suit.CLUBS)
    final = _card(Rank.SEVEN if truthful else Rank.TWO, Suit.DIAMONDS)
    game.hands = [Hand([final]), Hand([old, _card(Rank.QUEEN, Suit.SPADES)])]
    game.current_player = 1
    assert game.play_cards(1, [old], Rank.SEVEN)[0]
    assert game.play_cards(0, [final], Rank.SEVEN)[0]

    assert [action.cards_revealed for action in game.actions] == [True, True]
    assert game.actions[-1].bluff_called is True
    assert game.actions[-1].was_bluff is (not truthful)
    assert old in game.hands[0 if not truthful else 1].cards
    assert final in game.hands[0 if not truthful else 1].cards


def test_pass_recycles_without_public_reveal_or_identity_leak():
    game, old, latest = _pile(latest_bluff=True)
    game.draw_pile = [_card(Rank.KING, Suit.HEARTS)]
    assert game.pass_turn(1)[0]

    assert all(not getattr(action, "cards_revealed", False) for action in game.actions)
    for viewer in (0, 1):
        actions = round_player_view(game, viewer)["actions"]
        assert actions[0].cards_played == ([old] if viewer == 1 else [None])
        assert actions[1].cards_played == ([latest] if viewer == 0 else [None])


@pytest.mark.parametrize("latest_bluff", [True, False])
def test_rank_memory_assigns_the_entire_revealed_pile_to_latest_result_loser(latest_bluff):
    game, old, latest = _pile(latest_bluff, earlier_bluff=not latest_bluff)
    assert game.call_bluff(1)[0]
    loser = 0 if latest_bluff else 1
    winner = 1 - loser

    bounds, _ = opponent_rank_bounds(round_player_view(game, winner)["actions"], winner)
    assert bounds[Rank.SEVEN] >= 1
    assert bounds[Rank.TWO] >= int(old.rank == Rank.TWO) + int(latest.rank == Rank.TWO)


def test_private_pickup_evidence_accepts_older_public_packets_without_double_counting():
    game, _old, _latest = _pile(latest_bluff=True, earlier_bluff=True)
    assert game.call_bluff(1)[0]
    collector = game.last_private_pickup[0]
    view = round_player_view(game, collector)
    evidence = extract_private_pickup_evidence(view)
    assert [row["action_index"] for row in evidence] == [
        action.action_index for action in game.actions if action.player != collector]
    # Once identities are public, the private channel is still valid and
    # observation de-duplication prevents counting those same packet labels twice.
    bot = RoundPickupAdaptiveBot(1)
    bot.player_id = collector
    bot.observe_player_view(view)
    before = deepcopy(bot.to_dict())
    bot.observe_player_view(view)
    assert bot.to_dict() == before


def test_information_state_key_distinguishes_older_public_card_identities():
    game, _old, _latest = _pile(latest_bluff=False)
    assert game.call_bluff(1)[0]
    view = round_player_view(game, 1)
    hand = view["own_hand"]
    changed = deepcopy(view)
    changed["actions"][0].cards_played = [_card(Rank.NINE, Suit.SPADES)]
    assert information_state_key(hand, view) != information_state_key(hand, changed)


def test_adaptive_public_observation_counts_each_revealed_action_once_not_forced_final():
    game, _old, _latest = _pile(latest_bluff=True)
    assert game.call_bluff(1)[0]
    bot = RoundAdaptiveBot(seed=3)
    bot.player_id = 0
    bot.observe_player_view(round_player_view(game, 0))
    assert RoundAdaptiveBot._counts(bot._bluff_evidence) == (0, 1)
    assert RoundAdaptiveBot._counts(bot._call_evidence) == (1, 0)
    before = bot.to_dict()
    bot.observe_player_view(round_player_view(game, 0))
    assert bot.to_dict() == before

    final = GameState()
    one = _card(Rank.TWO, Suit.HEARTS)
    final.hands = [Hand([one]), Hand([_card(Rank.THREE, Suit.CLUBS)])]
    final.current_player = 0
    assert final.play_cards(0, [one], Rank.TWO)[0]
    observer = RoundAdaptiveBot(seed=4)
    observer.player_id = 1
    observer.observe_player_view(round_player_view(final, 1))
    assert RoundAdaptiveBot._counts(observer._bluff_evidence) == (0, 0)
    assert RoundAdaptiveBot._counts(observer._call_evidence) == (0, 0)
