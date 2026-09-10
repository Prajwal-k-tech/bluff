"""Engine-level unit tests for game.py.

Covers:
  - forced-call-when-empty (can_pass() False + pass rejected/forced when draw pile is 0)
  - passer-draw attribution (pass_turn(passer=X) draws to X, not current_player)
  - pile-transfer correctness on call_bluff (winner/loser pile assignment)
  - 100-turn draw rule

Runnable standalone:  python test_engine.py
        or via pytest: pytest test_engine.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cards import Card, Deck, Hand, Rank, Suit
from game import GameState, Action


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_card(rank: Rank, suit: Suit = Suit.SPADES) -> Card:
    return Card(rank, suit)


def _make_game_empty_draw() -> GameState:
    """Deal a normal game, then empty the draw pile."""
    g = GameState(num_players=2)
    g.deal(14)
    g.draw_pile.clear()
    return g


def _setup_call_scenario(g: GameState) -> None:
    """Put the game into a state where the other player can call or pass."""
    # Ensure there is an action to respond to
    hand = g.get_hand(g.current_player)
    card = hand.cards[0]
    g.play_cards(g.current_player, [card], card.rank)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestForcedCallWhenEmpty:
    """When draw pile is empty, can_pass() must return False and callers
    must be forced to call_bluff instead of passing."""

    def test_can_pass_false_when_empty(self):
        g = _make_game_empty_draw()
        assert g.can_pass() is False

    def test_can_pass_true_when_cards_remain(self):
        g = GameState(num_players=2)
        g.deal(14)
        assert g.can_pass() is True
        assert len(g.draw_pile) == 24  # 52 - 14*2

    def test_pass_turn_succeeds_even_when_empty(self):
        """pass_turn() itself does NOT block — callers should gate on
        can_pass() first, but pass_turn handles empty pile gracefully."""
        g = _make_game_empty_draw()
        _setup_call_scenario(g)
        other = 1 - g.current_player
        ok, msg = g.pass_turn(passer=other)
        assert ok is True
        assert "empty" in msg.lower() or "no card" in msg.lower()

    def test_call_bluff_works_when_empty(self):
        """Even with empty draw pile, call_bluff must succeed if there
        is an action to challenge."""
        g = _make_game_empty_draw()
        _setup_call_scenario(g)
        other = 1 - g.current_player
        ok, msg, act = g.call_bluff(other)
        assert ok is True

    def test_can_call_bluff_true_when_empty_and_actions_exist(self):
        g = _make_game_empty_draw()
        _setup_call_scenario(g)
        assert g.can_call_bluff() is True


class TestPasserDrawAttribution:
    """pass_turn(passer=X) draws the card to X's hand, not to the
    current_player's hand (which may differ from the passer)."""

    def test_card_goes_to_passer(self):
        g = GameState(num_players=2)
        # Pin starter: this test exercises passer logic, not the deal coin
        # flip (game-rules.md §1).
        g.deal(14, random_start=False)
        g.current_player = 0
        assert len(g.draw_pile) > 0

        # Put a known card on top of the draw pile
        sentinel = _make_card(Rank.ACE, Suit.HEARTS)
        g.draw_pile[0] = sentinel

        passer = 1  # not necessarily the current_player
        old_hand = g.get_hand(passer).size()
        old_other = g.get_hand(0).size()

        _setup_call_scenario(g)
        # Now the other player (1) is the passer

        # Play cards first so there's an action to respond to
        _setup_call_scenario(g)
        other = 1 - g.current_player
        ok, msg = g.pass_turn(passer=other)
        assert ok

        # Card went to the passer
        assert g.get_hand(other).size() == old_hand + 1

    def test_passer_not_current_player(self):
        """Demonstrate the passer parameter matters: if current_player is 0
        but we pass passer=1, the draw goes to player 1."""
        g = GameState(num_players=2)
        # Pin seat 0: this test exercises passer logic, not the deal coin
        # flip (game-rules.md §1). Deal deterministically, then set starter.
        g.deal(14, random_start=False)
        g.current_player = 0
        sentinel = _make_card(Rank.KING, Suit.CLUBS)
        g.draw_pile[0] = sentinel

        # Ensure current_player is 0, play a card so there is an action
        assert g.current_player == 0
        hand0 = g.get_hand(0)
        card = hand0.cards[0]
        g.play_cards(0, [card], card.rank)

        # Now pass with passer=1 (the other player)
        p1_before = g.get_hand(1).size()
        p0_before = g.get_hand(0).size()
        ok, msg = g.pass_turn(passer=1)
        assert ok
        # Card should go to player 1 (passer), NOT player 0 (current_player)
        assert g.get_hand(1).size() == p1_before + 1
        assert g.get_hand(0).size() == p0_before

    def test_turn_advances_after_pass(self):
        g = GameState(num_players=2)
        g.deal(14)
        initial_player = g.current_player
        _setup_call_scenario(g)
        other = 1 - g.current_player
        g.pass_turn(passer=other)
        # current_player should have toggled
        assert g.current_player != initial_player


class TestPileTransferOnCallBluff:
    """call_bluff correctly assigns the pile to bluffer or caller per
    game-rules.md §3."""

    def test_bluff_caught_bluffer_takes_pile(self):
        """When the last play was a bluff, the bluffer takes the pile."""
        g = GameState(num_players=2)
        # Pin starter: pile logic under test, not the deal coin flip.
        g.deal(14, random_start=False)
        g.current_player = 0

        # Player 0 will play a bluff: claim ACE but play a non-ACE
        hand0 = g.get_hand(0)
        # Find a non-ACE card
        non_ace = None
        for c in hand0.cards:
            if c.rank != Rank.ACE:
                non_ace = c
                break
        assert non_ace is not None, "Need a non-ACE in hand0"

        g.play_cards(0, [non_ace], Rank.ACE)  # this is a bluff
        assert g.get_pile_size() == 1

        p0_before = g.get_hand(0).size()

        # Player 1 calls bluff
        ok, msg, act = g.call_bluff(1)
        assert ok
        assert act.was_bluff is True
        assert act.bluff_called is True

        # Bluffer (player 0) takes the pile
        pile_cards = 1  # the one card we played
        assert g.get_hand(0).size() == p0_before + pile_cards
        assert g.get_pile_size() == 0  # pile cleared

    def test_wrong_caller_takes_pile(self):
        """When the caller is wrong, the caller takes the pile."""
        g = GameState(num_players=2)
        # Pin starter: pile logic under test, not the deal coin flip.
        g.deal(14, random_start=False)
        g.current_player = 0

        hand0 = g.get_hand(0)
        # Find an ACE to play honestly
        ace = None
        for c in hand0.cards:
            if c.rank == Rank.ACE:
                ace = c
                break
        if ace is None:
            import pytest
            pytest.skip("no ACE dealt")

        g.play_cards(0, [ace], Rank.ACE)  # honest play
        assert g.get_pile_size() == 1

        p1_before = g.get_hand(1).size()

        ok, msg, act = g.call_bluff(1)
        assert ok
        assert act.was_bluff is False
        assert act.caller_was_right is False

        # Caller (player 1) takes the pile
        assert g.get_hand(1).size() == p1_before + 1
        assert g.get_pile_size() == 0

    def test_pile_cleared_after_call(self):
        """Regardless of who was right, the pile is cleared."""
        g = GameState(num_players=2)
        g.deal(14)
        _setup_call_scenario(g)
        pile_before = g.get_pile_size()
        other = 1 - g.current_player
        g.call_bluff(other)
        assert g.get_pile_size() == 0

    def test_cannot_call_bluff_twice(self):
        g = GameState(num_players=2)
        g.deal(14)
        _setup_call_scenario(g)
        other = 1 - g.current_player
        g.call_bluff(other)
        # After call_bluff, turn advances — original "other" is now current_player.
        # Use the *new* other player to attempt the second call.
        other2 = 1 - g.current_player
        ok2, msg2, _ = g.call_bluff(other2)
        assert ok2 is False
        assert "already called" in msg2.lower()


class TestDealCoinFlip:
    """LOCKED game-rules.md §1: first player is a 50/50 coin flip."""

    def test_random_start_off_pins_seat_zero(self):
        g = GameState(num_players=2)
        g.deal(14, random_start=False)
        assert g.current_player == 0

    def test_coin_flip_produces_both_starters(self):
        starters = set()
        for _ in range(200):
            g = GameState(num_players=2)
            g.deal(14)
            assert g.current_player in (0, 1)
            starters.add(g.current_player)
        assert starters == {0, 1}, "200 deals never produced a starter — flip broken"


class TestHundredTurnDraw:
    """100-turn draw rule: after 100 turns without a winner, the game
    ends in a draw (winner is None, game_over is True)."""

    def test_draw_after_max_turns(self):
        g = GameState(num_players=2)
        g.deal(14)

        # Drive the game to turn 100 by alternating plays and passes/calls
        for _ in range(200):  # generous upper bound
            if g.game_over:
                break
            current = g.current_player
            other = 1 - current

            # Play a card (keeps the game flowing)
            hand = g.get_hand(current)
            if hand.size() == 0:
                break
            card = hand.cards[0]
            g.play_cards(current, [card], card.rank)

            if g.game_over:
                break

            # Other player passes
            if g.can_pass():
                g.pass_turn(passer=other)
            else:
                g.call_bluff(other)

            if g.turn_count >= 100:
                break

        # The engine MUST enforce the 100-turn draw rule
        assert g.game_over is True, (
            f"Game should be over after 100 turns, but game_over={g.game_over}, "
            f"turn_count={g.turn_count}"
        )
        assert g.winner is None, (
            f"Draw should have winner=None, but winner={g.winner}"
        )

    def test_turn_count_increments(self):
        g = GameState(num_players=2)
        g.deal(14)
        initial = g.turn_count
        _setup_call_scenario(g)
        other = 1 - g.current_player
        g.pass_turn(passer=other)
        assert g.turn_count == initial + 1

    def test_call_bluff_increments_turn(self):
        g = GameState(num_players=2)
        g.deal(14)
        _setup_call_scenario(g)
        initial = g.turn_count
        other = 1 - g.current_player
        g.call_bluff(other)
        assert g.turn_count == initial + 1


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
