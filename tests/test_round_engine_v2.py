from collections import Counter

import pytest

import game_v2
from cards import Card, Hand, Rank, Suit
from game_v2 import GameState, RULESET_ID


def c(rank, suit):
    return Card(rank, suit)


def fixture_state(hand0, hand1, *, current=0, draw=None):
    state = GameState()
    state.hands = [Hand(hand0), Hand(hand1)]
    state.current_player = current
    state.draw_pile = list(draw or [])
    return state


def cards_in_play(state):
    return [card for hand in state.hands for card in hand.cards] + state.pile + state.draw_pile


def test_deal_uses_52_unique_cards_and_pass_recycles_before_drawing():
    state = GameState()
    state.deal(14, random_start=False)
    all_cards = cards_in_play(state)
    assert RULESET_ID == "bluff-fixed-rounds-v2"
    assert [hand.size() for hand in state.hands] == [14, 14]
    assert len(state.draw_pile) == 24
    assert len(all_cards) == len(set(all_cards)) == 52
    assert state.current_player == 0

    lead = state.hands[0].cards[0]
    assert state.play_cards(0, [lead], Rank.SEVEN)[0]
    recycled = state.pile[0]
    draw_before = list(state.draw_pile)
    shuffled = []

    def observe_shuffle(cards):
        shuffled.extend(cards)

    original_shuffle = game_v2.random.shuffle
    game_v2.random.shuffle = observe_shuffle
    try:
        ok, _ = state.pass_turn(1)
    finally:
        game_v2.random.shuffle = original_shuffle

    assert ok
    assert shuffled == draw_before + [recycled]
    assert draw_before[0] in state.hands[1].cards
    assert len(state.draw_pile) == len(draw_before)
    assert len(state.draw_pile) >= 24
    assert state.pile == []
    assert state.current_player == 0
    assert state.round_number == 2
    assert state.active_rank is None
    assert state.pending_claim is None
    assert state.round_start_action_index == len(state.actions)
    assert state.current_round_actions == []
    assert state.actions[0].response_was_call is False
    assert state.actions[0].response_was_forced is False
    assert state.actions[0].response_player == 1
    assert state.actions[0].response_hand_size_before == 14
    assert len(cards_in_play(state)) == len(set(cards_in_play(state))) == 52


def test_any_rank_leads_round_and_all_later_plays_keep_that_rank():
    first = c(Rank.TWO, Suit.HEARTS)
    first_spare = c(Rank.THREE, Suit.HEARTS)
    second = c(Rank.SEVEN, Suit.CLUBS)
    second_spare = c(Rank.FOUR, Suit.HEARTS)
    state = fixture_state(
        [first, first_spare], [second, second_spare],
        draw=[c(Rank.NINE, Suit.DIAMONDS)],
    )

    assert state.active_rank is None
    assert state.play_cards(0, [first], Rank.TWO)[0]
    first_claim = state.last_action
    assert state.current_player == 1
    assert state.active_rank == Rank.TWO
    assert state.can_play(1)
    assert state.can_pass(1)
    assert state.can_call_bluff(1)

    ok, _ = state.play_cards(1, [second], Rank.SEVEN)
    assert not ok
    assert state.pending_claim is first_claim
    assert state.hands[1].cards == sorted([second, second_spare])

    assert state.play_cards(1, [second], Rank.TWO)[0]
    assert first_claim.response_was_call is False
    assert first_claim.response_was_forced is False
    assert first_claim.response_player == 1
    assert first_claim.response_hand_size_before == 2
    assert state.last_action is state.pending_claim
    assert state.last_action.player == 1
    assert state.current_player == 0
    assert not state.can_call_bluff(1)
    assert state.can_call_bluff(0)


def test_invalid_plays_are_atomic_and_duplicate_cards_are_rejected():
    a = c(Rank.SEVEN, Suit.HEARTS)
    b = c(Rank.TWO, Suit.HEARTS)
    state = fixture_state([a, b], [c(Rank.KING, Suit.HEARTS)])
    state.active_rank = Rank.SEVEN

    def snapshot():
        return (
            tuple(tuple(hand.cards) for hand in state.hands), tuple(state.pile),
            tuple(state.draw_pile), state.current_player, state.turn_count,
            tuple(state.actions), tuple(state.cards_played), state.active_rank,
            state.pending_claim,
        )

    invalid_plays = [
        (1, [a], Rank.SEVEN),
        (0, [], Rank.SEVEN),
        (0, [a, a], Rank.SEVEN),
        (0, [a], 99),
        (0, [a], Rank.EIGHT),
        (0, [c(Rank.ACE, Suit.SPADES)], Rank.SEVEN),
        (0, [Card(99, Suit.SPADES)], Rank.SEVEN),
        (0, [a, b, c(Rank.THREE, Suit.CLUBS), c(Rank.FOUR, Suit.CLUBS),
             c(Rank.FIVE, Suit.CLUBS)], Rank.SEVEN),
    ]
    for args in invalid_plays:
        before = snapshot()
        assert state.play_cards(*args)[0] is False
        assert snapshot() == before

    with pytest.raises(ValueError):
        GameState(3)


def test_only_latest_opponent_packet_can_be_challenged_and_prior_cards_stay_hidden():
    prior = c(Rank.TWO, Suit.HEARTS)
    latest_true = c(Rank.SEVEN, Suit.DIAMONDS)
    latest_false = c(Rank.TWO, Suit.CLUBS)
    state = fixture_state(
        [prior, c(Rank.NINE, Suit.HEARTS)],
        [latest_true, latest_false, c(Rank.JACK, Suit.HEARTS)],
    )

    assert state.play_cards(0, [prior], Rank.SEVEN)[0]
    old_claim = state.last_action
    assert state.play_cards(1, [latest_true, latest_false], Rank.SEVEN)[0]
    latest_claim = state.last_action
    assert latest_claim.was_bluff  # any mismatch makes a mixed packet false
    assert old_claim.response_was_call is False
    assert state.latest_pending_claim is latest_claim
    assert state.current_round_actions == [old_claim, latest_claim]
    assert state.can_call_bluff(0)
    assert not state.can_call_bluff(1)

    ok, message, resolved = state.call_bluff(0)
    assert ok and resolved is latest_claim
    assert str(latest_true) in message and str(latest_false) in message
    assert str(prior) not in message
    assert len(state.hands[1].cards) == 4  # spare + both latest cards + prior packet
    assert state.pile == []
    assert state.current_player == 0  # successful challenger leads
    assert state.round_number == 2
    assert state.active_rank is None
    assert state.pending_claim is None
    assert latest_claim.bluff_called
    assert latest_claim.caller_was_right
    assert latest_claim.response_was_call is True
    assert latest_claim.response_was_forced is False
    assert latest_claim.response_player == 0
    assert latest_claim.response_hand_size_before == 1
    assert state.turn_count == 3
    assert state.round_start_action_index == len(state.actions)
    assert state.current_round_actions == []


def test_challenging_a_truthful_packet_gives_the_whole_pile_to_caller():
    first = c(Rank.SEVEN, Suit.HEARTS)
    second = c(Rank.SEVEN, Suit.DIAMONDS)
    state = fixture_state(
        [first, c(Rank.FOUR, Suit.HEARTS)],
        [second, c(Rank.FIVE, Suit.HEARTS)],
    )
    assert state.play_cards(0, [first], Rank.SEVEN)[0]
    assert state.play_cards(1, [second], Rank.SEVEN)[0]
    latest = state.last_action

    ok, _, resolved = state.call_bluff(0)
    assert ok and resolved is latest
    assert not latest.was_bluff
    assert not latest.caller_was_right
    assert latest.response_player == 0
    assert latest.response_hand_size_before == 1
    assert latest.player_hand_size_before == 2
    assert state.hands[0].size() == 3
    assert state.pile == []
    assert state.current_player == 1  # truthful claimant leads
    assert state.round_number == 2


def test_final_mixed_packet_is_forced_challenged_and_returns_whole_pile():
    earlier = c(Rank.SEVEN, Suit.HEARTS)
    final_true = c(Rank.SEVEN, Suit.DIAMONDS)
    final_false = c(Rank.TWO, Suit.CLUBS)
    state = fixture_state(
        [final_true, final_false],
        [earlier, c(Rank.FOUR, Suit.HEARTS)],
        current=1,
    )

    assert state.play_cards(1, [earlier], Rank.SEVEN)[0]
    turns_before_final = state.turn_count
    ok, message = state.play_cards(0, [final_true, final_false], Rank.SEVEN)
    assert ok and "caught" in message
    final_action = state.last_action
    assert final_action.was_bluff
    assert final_action.bluff_called
    assert final_action.caller_was_right
    assert final_action.response_was_call is True
    assert final_action.response_was_forced is True
    assert final_action.response_player == 1
    assert final_action.response_hand_size_before == 1
    assert final_action.player_hand_size_before == 2
    assert state.hands[0].size() == 3
    assert state.pile == []
    assert not state.game_over
    assert state.winner is None
    assert state.current_player == 1
    assert state.round_number == 2
    assert state.turn_count == turns_before_final + 1
    assert state.round_start_action_index == len(state.actions)


def test_truthful_final_packet_wins_with_forced_public_response_metadata():
    final = c(Rank.KING, Suit.CLUBS)
    state = fixture_state([final], [c(Rank.TWO, Suit.DIAMONDS)])
    assert state.play_cards(0, [final], Rank.KING)[0]
    action = state.last_action

    assert state.game_over
    assert state.winner == 0
    assert action.bluff_called
    assert not action.caller_was_right
    assert action.response_was_call is True
    assert action.response_was_forced is True
    assert state.current_player == 1
    assert state.pile == []
    assert final in state.hands[1].cards


@pytest.mark.parametrize("actor", [True, False, 0.0, 1.0, -1, 2, "0"])
def test_malformed_actor_does_not_alias_a_valid_player(actor):
    state = GameState()
    state.deal(random_start=False)
    before = Counter(cards_in_play(state))
    assert not state.can_play(actor)
    assert not state.can_pass(actor)
    assert not state.can_call_bluff(actor)
    assert not state.play_cards(actor, state.hands[0].cards[:1], Rank.KING)[0]
    assert not state.pass_turn(actor)[0]
    assert not state.call_bluff(actor)[0]
    assert Counter(cards_in_play(state)) == before
    assert state.turn_count == 0


def test_responses_are_actor_aware_and_empty_draw_only_disables_passing():
    lead = c(Rank.ACE, Suit.HEARTS)
    state = fixture_state([lead, c(Rank.TWO, Suit.HEARTS)], [c(Rank.THREE, Suit.HEARTS)])
    assert state.play_cards(0, [lead], Rank.ACE)[0]
    assert not state.can_call_bluff(0)
    assert not state.can_pass(0)
    assert state.can_call_bluff(1)

    state.draw_pile.clear()
    assert not state.can_pass(1)
    assert state.can_call_bluff(1)
    assert state.can_play(1)
    assert not state.pass_turn(1)[0]
    assert state.call_bluff(0)[0] is False


def test_no_gameplay_turn_cap():
    lead = c(Rank.QUEEN, Suit.CLUBS)
    state = fixture_state([lead, c(Rank.TWO, Suit.CLUBS)], [c(Rank.THREE, Suit.CLUBS)])
    state.turn_count = 10_000
    assert state.play_cards(0, [lead], Rank.QUEEN)[0]
    assert not state.game_over
    assert state.turn_count == 10_001


def test_physical_card_conservation_after_challenge_and_pass():
    state = GameState()
    state.deal(14, random_start=False)
    expected = Counter(cards_in_play(state))
    first = state.hands[0].cards[0]
    assert state.play_cards(0, [first], Rank.QUEEN)[0]
    assert state.pass_turn(1)[0]
    assert Counter(cards_in_play(state)) == expected

    current = state.current_player
    played = state.hands[current].cards[0]
    assert state.play_cards(current, [played], Rank.TWO)[0]
    caller = state.current_player
    assert state.call_bluff(caller)[0]
    assert Counter(cards_in_play(state)) == expected

