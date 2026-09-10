"""Game state and turn logic for 2-player Bluff."""

import random
from typing import List, Optional, Tuple
from cards import Card, Deck, Hand, Rank


class Action:
    """A single turn action."""
    def __init__(self, player: int, cards_played: List[Card],
                 claimed_rank: Rank, was_bluff: bool,
                 bluff_called: bool, caller_was_right: bool,
                 pile_size_before: int):
        self.player = player                # who played (0 or 1)
        self.cards_played = cards_played    # actual cards (hidden from opponent)
        self.claimed_rank = claimed_rank    # what they said
        self.was_bluff = was_bluff          # were they lying?
        self.bluff_called = bluff_called    # did opponent call?
        self.caller_was_right = caller_was_right  # was the call correct?
        self.pile_size_before = pile_size_before

    def __repr__(self) -> str:
        cards_str = ", ".join(str(c) for c in self.cards_played)
        bluff_tag = " [BLUFF]" if self.was_bluff else ""
        call_tag = ""
        if self.bluff_called:
            call_tag = f" → CALLED {'✓' if self.caller_was_right else '✗'}"
        return f"P{self.player} played {cards_str} as {self.claimed_rank.display()}{bluff_tag}{call_tag}"


class GameState:
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.deck = Deck()
        self.hands: List[Hand] = [Hand() for _ in range(num_players)]
        self.pile: List[Card] = []
        self.draw_pile: List[Card] = []  # explicit draw pile (remaining deck after deal)
        self.current_player: int = 0
        self.turn_count: int = 0
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.actions: List[Action] = []  # full game history
        self.cards_played: List[Card] = []  # all cards that left hands (for card counting)

    def deal(self, cards_per_player: int = 14, random_start: bool = True):
        """Deal cards to all players. Remaining cards form the draw pile.

        LOCKED game-rules.md §1: the first player is a 50/50 coin flip.
        Pass random_start=False only for deterministic unit tests.
        """
        for i in range(self.num_players):
            cards = self.deck.deal(cards_per_player)
            self.hands[i].add(cards)
        # Remaining deck becomes the draw pile
        self.draw_pile = self.deck.deal(self.deck.remaining())
        if random_start:
            self.current_player = random.randrange(self.num_players)

    def get_hand(self, player: int) -> Hand:
        return self.hands[player]

    def get_pile_size(self) -> int:
        return len(self.pile)

    def get_cards_played(self) -> List[Card]:
        """All cards that have been played (for card counting)."""
        return list(self.cards_played)

    def get_remaining_deck_count(self) -> int:
        """Cards not in any hand or pile."""
        return self.deck.remaining()

    def play_cards(self, player: int, cards: List[Card],
                   claimed_rank: Rank) -> Tuple[bool, str]:
        """
        Player plays cards face-down, claiming they are claimed_rank.
        Returns (success, message).
        """
        if self.game_over:
            return False, "Game is over."
        if player != self.current_player:
            return False, "Not your turn."
        if len(cards) == 0:
            return False, "Must play at least 1 card."
        if len(cards) > 4:
            return False, "Cannot play more than 4 cards."

        # Remove cards from hand
        if not self.hands[player].remove(cards):
            return False, "You don't have those cards."

        # Add cards to pile (face-down, so pile doesn't reveal them yet)
        self.pile.extend(cards)

        # Track all played cards
        self.cards_played.extend(cards)

        # Record action
        was_bluff = not all(c.rank == claimed_rank for c in cards)
        action = Action(
            player=player,
            cards_played=list(cards),
            claimed_rank=claimed_rank,
            was_bluff=was_bluff,
            bluff_called=False,
            caller_was_right=False,
            pile_size_before=len(self.pile) - len(cards)
        )
        self.actions.append(action)

        # Check if player emptied hand
        if not self.hands[player].has_cards():
            self.game_over = True
            self.winner = player
            return True, f"Player {player} wins! Hand is empty."

        return True, f"Played {len(cards)} card(s) as {claimed_rank.display()}."

    def can_pass(self) -> bool:
        """Check if current player can pass. Cannot pass when draw pile is empty."""
        return len(self.draw_pile) > 0

    def can_call_bluff(self) -> bool:
        """Check if current player can call bluff."""
        if self.game_over:
            return False
        if not self.actions:
            return False
        last_action = self.actions[-1]
        return not last_action.bluff_called

    def call_bluff(self, caller: int) -> Tuple[bool, str, Optional[Action]]:
        """
        Caller challenges the last play.
        Returns (success, message, last_action).
        """
        if self.game_over:
            return False, "Game is over.", None
        if caller == self.current_player:
            return False, "You can't call bluff on your own play.", None
        if not self.actions:
            return False, "Nothing to call bluff on.", None

        last_action = self.actions[-1]
        if last_action.bluff_called:
            return False, "Bluff already called on this play.", None

        last_action.bluff_called = True
        last_action.caller_was_right = last_action.was_bluff

        # Reveal cards
        if last_action.was_bluff:
            # Bluffer lied → they take the pile
            bluffer = last_action.player
            pile_copy = list(self.pile)  # copy before clearing
            self.hands[bluffer].add(pile_copy)
            result_msg = (f"BLUFF CAUGHT! {last_action.cards_played} were NOT "
                         f"{last_action.claimed_rank.display()}. "
                         f"Player {bluffer} takes {len(self.pile)} cards.")
        else:
            # Caller wrong → they take the pile
            pile_copy = list(self.pile)  # copy before clearing
            self.hands[caller].add(pile_copy)
            result_msg = (f"WRONG CALL! {last_action.cards_played} were indeed "
                         f"{last_action.claimed_rank.display()}. "
                         f"Player {caller} takes {len(self.pile)} cards.")

        self.pile.clear()
        self._next_turn()

        return True, result_msg, last_action

    def pass_turn(self, passer: Optional[int] = None) -> Tuple[bool, str]:
        """Pass without calling bluff. Draw 1 card from draw pile.

        Args:
            passer: The player who is passing. Defaults to the current player.
                IMPORTANT: when the *responder* passes on the last play, pass
                their index here — the draw must go to them, not to the
                player who just played.
        """
        if self.game_over:
            return False, "Game is over."
        if not self.actions:
            return False, "Nothing to pass on."

        last_action = self.actions[-1]
        if last_action.bluff_called:
            return False, "Bluff already called."

        # Draw 1 card from draw pile (to the passer, not necessarily current)
        target = passer if passer is not None else self.current_player
        if len(self.draw_pile) > 0:
            drawn = self.draw_pile.pop(0)
            self.hands[target].add([drawn])
            msg = f"Passed. Drew 1 card ({drawn})."
        else:
            msg = "Passed. Draw pile empty, no card drawn."

        self._next_turn()
        return True, msg

    def _next_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        self.turn_count += 1
        # §4/§6: Round limit — game is a draw if no winner after 100 turns
        if self.turn_count >= 100:
            self.game_over = True
            # winner stays None → draw

    def display_state(self, viewer: int):
        """Display game state from a player's perspective."""
        print(f"\n{'='*50}")
        print(f"Turn {self.turn_count} | Pile: {len(self.pile)} cards | Draw pile: {len(self.draw_pile)} cards")
        print(f"Your hand ({self.hands[viewer].size()} cards): {self.hands[viewer].display()}")
        other = 1 - viewer
        print(f"Opponent: {self.hands[other].size()} cards")
        print(f"{'='*50}")
