from bots.base import public_action_for_viewer
from bots.round_honest_hypothesis_memory import honest_count_hypothesis_bounds
from bots.round_observation import round_player_view
from bots.round_policy import RoundAdaptiveBot
from bots.round_public_opponent_model import public_adaptive_from_history
from cards import Card, Hand, Rank, Suit
from game_v2 import GameState


def _c(rank, suit):
    return Card(rank, suit)


def _prior_transfer_history(old_packet_uses_twos):
    """Same challenger knowledge; hidden old packet consumes a different rank."""
    game = GameState()
    sevens = [_c(Rank.SEVEN, suit) for suit in Suit]
    eights = [_c(Rank.EIGHT, Suit.HEARTS), _c(Rank.EIGHT, Suit.DIAMONDS)]
    twos = [_c(Rank.TWO, Suit.CLUBS), _c(Rank.TWO, Suit.SPADES),
            _c(Rank.TWO, Suit.DIAMONDS)]
    kings = [_c(Rank.KING, Suit.HEARTS), _c(Rank.KING, Suit.DIAMONDS)]
    game.hands = [Hand(), Hand(sevens + eights + twos + kings)]
    game.draw_pile = [_c(Rank.ACE, Suit.CLUBS)]
    game.current_player = 1

    # These wrong challenges transfer four Sevens, two Eights and one Two to
    # player 0. Player 1 therefore knows all seven identities in player 0's hand.
    for cards, rank in ((sevens, Rank.SEVEN), (eights, Rank.EIGHT),
                        ([twos[0]], Rank.TWO)):
        assert game.play_cards(1, cards, rank)[0]
        assert game.call_bluff(0)[0]

    # Identical public sequence in both variants through the first action of
    # the decisive round. The target's own known-card inventory is unchanged.
    assert game.play_cards(1, [twos[1]], Rank.TWO)[0]
    old = [twos[0]] if old_packet_uses_twos else [sevens[0]]
    assert game.play_cards(0, old, Rank.TWO)[0]
    assert game.play_cards(1, [twos[2]], Rank.TWO)[0]
    # The latest claim is the same one-card false Two in each history.
    assert game.play_cards(0, [sevens[1]], Rank.TWO)[0]
    return game


def _visible_action_signature(view):
    return [(
        action.player, action.action_index, action.round_number,
        tuple(action.cards_played), action.claimed_rank, action.was_bluff,
        action.bluff_called, action.response_kind, action.response_player,
        action.response_was_forced,
    ) for action in view["actions"]]


def test_retrospective_old_packet_identity_does_not_change_prechallenge_knowledge():
    # Variant A's older packet spends one of the known Sevens; variant B spends
    # a known Two. At the time player 1 sees the older packet, both are hidden.
    seven_old = _prior_transfer_history(old_packet_uses_twos=False)
    two_old = _prior_transfer_history(old_packet_uses_twos=True)
    before_seven = round_player_view(seven_old, 1)
    before_two = round_player_view(two_old, 1)
    assert _visible_action_signature(before_seven) == _visible_action_signature(before_two)
    assert before_seven["hand_size"] == before_two["hand_size"]
    assert before_seven["opponent_hand_size"] == before_two["opponent_hand_size"]
    assert before_seven["actions"][-3].cards_played == [None]
    assert before_two["actions"][-3].cards_played == [None]

    assert seven_old.call_bluff(1)[0]
    assert two_old.call_bluff(1)[0]
    after_seven = honest_count_hypothesis_bounds(round_player_view(seven_old, 0))
    after_two = honest_count_hypothesis_bounds(round_player_view(two_old, 0))

    # The actual old identities are public now, but cannot be used to claim
    # player 1 knew an exact remaining rank count before deciding to challenge.
    assert after_seven["compatible"] == after_two["compatible"]
    assert after_seven["compatible"] is True
    assert after_seven["conditional_bounds"] == {Rank.TWO: 2}
    assert after_seven["conditional_bounds"] == after_two["conditional_bounds"]


def test_public_adaptive_history_uses_only_currently_exposed_opponent_labels():
    game = GameState()
    old = _c(Rank.SEVEN, Suit.CLUBS)
    middle = _c(Rank.SEVEN, Suit.HEARTS)
    latest = _c(Rank.TWO, Suit.DIAMONDS)
    game.hands = [Hand([old, latest, _c(Rank.FIVE, Suit.HEARTS)]),
                  Hand([middle, _c(Rank.SIX, Suit.CLUBS)])]
    game.current_player = 0
    assert game.play_cards(0, [old], Rank.SEVEN)[0]
    assert game.play_cards(1, [middle], Rank.SEVEN)[0]
    assert game.play_cards(0, [latest], Rank.SEVEN)[0]

    before = [public_action_for_viewer(action, 0) for action in game.actions]
    hidden = public_adaptive_from_history(before, opponent_id=1, seed=7)
    assert RoundAdaptiveBot._counts(hidden._bluff_evidence) == (0, 0)

    # Falsifying the redacted root-owned truth does not create evidence.
    forged_hidden = [public_action_for_viewer(action, 0) for action in game.actions]
    forged_hidden[0].was_bluff = True
    unchanged = public_adaptive_from_history(forged_hidden, opponent_id=1, seed=7)
    assert unchanged.to_dict() == hidden.to_dict()

    assert game.call_bluff(1)[0]
    after = [public_action_for_viewer(action, 0) for action in game.actions]
    learned = public_adaptive_from_history(after, opponent_id=1, seed=7)
    assert [action.cards_revealed for action in after] == [True, True, True]
    assert RoundAdaptiveBot._counts(learned._bluff_evidence) == (1, 1)
