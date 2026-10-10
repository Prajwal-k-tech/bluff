"""Fixed-rank-round two-player Bluff engine."""

import random
from typing import List, Optional, Tuple

from cards import Card, Deck, Hand, Rank, Suit
from game import Action

RULESET_ID = "bluff-fixed-rounds-v2"


class GameState:
    """Game state for fixed-rank rounds with play, pass, and challenge responses."""

    def __init__(self, num_players: int = 2):
        if num_players != 2:
            raise ValueError("bluff-fixed-rounds-v2 requires exactly two players")
        self.num_players = 2
        self.deck = Deck()
        self.hands: List[Hand] = [Hand(), Hand()]
        self.pile: List[Card] = []
        self.draw_pile: List[Card] = []
        self.current_player = 0
        self.turn_count = 0
        self.game_over = False
        self.winner: Optional[int] = None
        self.actions: List[Action] = []
        self.cards_played: List[Card] = []
        self.round_number = 1
        self.round_start_action_index = 0
        self.active_rank: Optional[Rank] = None
        self.pending_claim: Optional[Action] = None

    @property
    def last_action(self) -> Optional[Action]:
        return self.actions[-1] if self.actions else None

    @property
    def latest_pending_claim(self) -> Optional[Action]:
        return self.pending_claim

    @property
    def current_round_actions(self) -> List[Action]:
        return self.actions[self.round_start_action_index:]

    def deal(self, cards_per_player: int = 14, random_start: bool = True):
        if type(cards_per_player) is not int or not 1 <= cards_per_player <= 26:
            raise ValueError("cards_per_player must be between 1 and 26")
        self.deck.reset()
        self.hands = [Hand(), Hand()]
        self.pile.clear()
        self.actions.clear()
        self.cards_played.clear()
        self.turn_count = 0
        self.game_over = False
        self.winner = None
        self.round_number = 1
        self.round_start_action_index = 0
        self.active_rank = None
        self.pending_claim = None
        for hand in self.hands:
            hand.add(self.deck.deal(cards_per_player))
        self.draw_pile = self.deck.deal(self.deck.remaining())
        self.current_player = random.randrange(2) if random_start else 0

    def get_hand(self, player: int) -> Hand:
        return self.hands[player]

    def get_pile_size(self) -> int:
        return len(self.pile)

    def get_cards_played(self) -> List[Card]:
        return list(self.cards_played)

    def can_play(self, player: Optional[int] = None) -> bool:
        actor = self.current_player if player is None else player
        return (type(actor) is int and actor in (0, 1) and not self.game_over
                and actor == self.current_player and bool(self.hands[actor].cards))

    def can_pass(self, player: Optional[int] = None) -> bool:
        actor = self.current_player if player is None else player
        return (type(actor) is int and actor in (0, 1) and not self.game_over
                and self.pending_claim is not None and actor == self.current_player
                and self.pending_claim.player != actor and bool(self.draw_pile))

    def can_call_bluff(self, player: Optional[int] = None) -> bool:
        actor = self.current_player if player is None else player
        return (type(actor) is int and actor in (0, 1)
                and not self.game_over and self.pending_claim is not None
                and actor == self.current_player
                and self.pending_claim.player != actor)

    def play_cards(self, player: int, cards: List[Card], claimed_rank: Rank) -> Tuple[bool, str]:
        if self.game_over:
            return False, "Game is over."
        if type(player) is not int or player not in (0, 1) or player != self.current_player:
            return False, "Not your turn."
        if not isinstance(cards, (list, tuple)) or not 1 <= len(cards) <= 4:
            return False, "Play between 1 and 4 cards."
        if not isinstance(claimed_rank, Rank):
            return False, "Invalid claimed rank."
        if any(not isinstance(card, Card) or not isinstance(card.rank, Rank)
               or not isinstance(card.suit, Suit) for card in cards):
            return False, "Invalid card."
        if len(set(cards)) != len(cards):
            return False, "Cannot play duplicate cards."
        if any(card not in self.hands[player].cards for card in cards):
            return False, "You don't have those cards."
        if self.active_rank is not None and claimed_rank != self.active_rank:
            return False, f"This round is fixed at {self.active_rank.display()}."

        if self.pending_claim is not None:
            self.pending_claim.response_was_call = False
            self.pending_claim.response_was_forced = False
            self.pending_claim.response_player = player
            self.pending_claim.response_hand_size_before = self.hands[player].size()
            self.pending_claim.response_kind = "play"

        was_bluff = any(card.rank != claimed_rank for card in cards)
        pile_size_before = len(self.pile)
        player_hand_size_before = self.hands[player].size()
        self.hands[player].remove(cards)
        self.pile.extend(cards)
        self.cards_played.extend(cards)
        action = Action(player, list(cards), claimed_rank, was_bluff, False, False,
                        pile_size_before)
        action.response_was_call = None
        action.response_was_forced = None
        action.response_player = None
        action.response_hand_size_before = None
        action.player_hand_size_before = player_hand_size_before
        # Public, deal-local identity survives detached observation copies.
        action.action_index = len(self.actions)
        action.round_number = self.round_number
        action.response_kind = None
        self.actions.append(action)
        self.pending_claim = action
        if self.active_rank is None:
            self.active_rank = claimed_rank
        self.turn_count += 1
        self.current_player = 1 - player

        if not self.hands[player].cards:
            self._resolve_final_claim(action, caller=self.current_player)
            if self.game_over:
                return True, f"Final claim was truthful. Player {player} wins."
            return True, f"Final bluff caught. Player {player} takes the pile."
        return True, f"Played {len(cards)} card(s) as {claimed_rank.display()}."

    def pass_turn(self, passer: Optional[int] = None) -> Tuple[bool, str]:
        actor = self.current_player if passer is None else passer
        if self.game_over:
            return False, "Game is over."
        if type(actor) is not int or actor not in (0, 1) or actor != self.current_player:
            return False, "Not your turn."
        if self.pending_claim is None:
            return False, "Nothing to pass on."
        if self.pending_claim.player == actor:
            return False, "Cannot pass on your own claim."
        if not self.draw_pile:
            return False, "Draw pile is empty."

        claim = self.pending_claim
        claim.response_was_call = False
        claim.response_was_forced = False
        claim.response_player = actor
        claim.response_hand_size_before = self.hands[actor].size()
        claim.response_kind = "pass"
        self.draw_pile.extend(self.pile)
        self.pile.clear()
        random.shuffle(self.draw_pile)
        drawn = self.draw_pile.pop(0)
        self.hands[actor].add([drawn])
        self.pending_claim = None
        self.active_rank = None
        self.round_number += 1
        self.round_start_action_index = len(self.actions)
        self.current_player = 1 - actor
        self.turn_count += 1
        return True, f"Passed. Drew 1 card; Player {self.current_player} starts round {self.round_number}."

    def call_bluff(self, caller: int) -> Tuple[bool, str, Optional[Action]]:
        if self.game_over:
            return False, "Game is over.", None
        if type(caller) is not int or caller not in (0, 1) or caller != self.current_player:
            return False, "Not your turn.", None
        claim = self.pending_claim
        if claim is None:
            return False, "Nothing to challenge.", None
        if claim.player == caller:
            return False, "Cannot challenge your own claim.", None

        return True, self._resolve_challenge(claim, caller, forced=False), claim

    def _resolve_final_claim(self, claim: Action, caller: int) -> None:
        claim.response_kind = "challenge"
        claim.response_was_call = True
        claim.response_was_forced = True
        claim.response_player = caller
        claim.response_hand_size_before = self.hands[caller].size()
        claim.bluff_called = True
        claim.caller_was_right = claim.was_bluff
        self.pending_claim = None
        self.active_rank = None
        if claim.was_bluff:
            self.hands[claim.player].add(self.pile)
            self.pile.clear()
            self.round_number += 1
            self.round_start_action_index = len(self.actions)
            self.current_player = caller
        else:
            # Automatic final challenges follow the same whole-pile transfer
            # rule even though a truthful claimant wins immediately afterward.
            self.hands[caller].add(self.pile)
            self.pile.clear()
            self.round_start_action_index = len(self.actions)
            self.game_over = True
            self.winner = claim.player

    def _resolve_challenge(self, claim: Action, caller: int, forced: bool) -> str:
        claim.response_kind = "challenge"
        claim.response_was_call = True
        claim.response_was_forced = forced
        claim.response_player = caller
        claim.response_hand_size_before = self.hands[caller].size()
        claim.bluff_called = True
        claim.caller_was_right = claim.was_bluff

        loser = claim.player if claim.was_bluff else caller
        leader = caller if claim.was_bluff else claim.player
        packet = ", ".join(str(card) for card in claim.cards_played)
        pile_size = len(self.pile)
        self.hands[loser].add(self.pile)
        self.pile.clear()
        self.pending_claim = None
        self.active_rank = None
        self.round_number += 1
        self.round_start_action_index = len(self.actions)
        self.current_player = leader
        self.turn_count += 1
        outcome = "bluff caught" if claim.was_bluff else "wrong challenge"
        return (f"{outcome}: latest packet [{packet}] claimed "
                f"{claim.claimed_rank.display()}; Player {loser} takes {pile_size} cards. "
                f"Player {leader} starts round {self.round_number}.")
