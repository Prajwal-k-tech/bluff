"""BotInterface — abstract base class for all Bluff bots.

Every bot (Random, Honest, CardCount, Adaptive, NN, Hybrid) implements this
interface. The game engine and web adapter only depend on this contract,
never on concrete bot implementations.

Design principle: the game engine calls decide_play() and decide_call().
The bot calls observe_action() to learn from every action it sees.
save/load persist learned state (opponent models, NN weights) across sessions.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from cards import Card, Rank
from game import Action


class BotInterface(ABC):
    """Abstract base class for all Bluff bots."""

    @abstractmethod
    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        """Decide which cards to play and what rank to claim.

        Args:
            hand: Current cards in hand (sorted).
            game_state: Dict with:
                - opponent_hand_size (int)
                - pile_size (int)
                - draw_pile_size (int)
                - turn_number (int)
                - last_action (Action or None)
                - cards_played (List[Card]) — all revealed cards

        Returns:
            (cards_to_play, claimed_rank)
            Must play 1-4 cards. All cards must be from hand.
        """
        pass

    @abstractmethod
    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """Decide whether to call bluff on the opponent's last play.

        Args:
            last_action: The action to evaluate (opponent's most recent play).
            game_state: Dict with:
                - hand_size (int) — our hand size
                - opponent_hand_size (int)
                - pile_size (int)
                - draw_pile_size (int)
                - turn_number (int)
                - cards_played (List[Card]) — all revealed cards

        Returns:
            True to call bluff, False to pass.
        """
        pass

    @abstractmethod
    def observe_action(self, action: Action, opponent_hand_size: int):
        """Observe any action (own or opponent's) for learning.

        Called after every play, call, or pass. Bots use this to update
        card counting, opponent models, etc.

        Args:
            action: The action that just occurred.
            opponent_hand_size: Opponent's hand size after this action.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist learned state to disk.

        Args:
            path: File path to save to (e.g., 'models/bayesian_user123.json').
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load learned state from disk.

        Args:
            path: File path to load from.
        """
        pass

    def reset(self):
        """Reset bot state for a new game. Override if needed."""
        pass


def build_game_state(
    hand_size: int,
    opponent_hand_size: int,
    pile_size: int,
    draw_pile_size: int,
    turn_number: int,
    last_action: Optional[Action] = None,
    cards_played: Optional[List[Card]] = None,
) -> dict:
    """Helper to build the game_state dict passed to bots.

    Used by the game engine and web adapter to construct a consistent
    state dictionary from the current GameState.
    """
    return {
        "hand_size": hand_size,
        "opponent_hand_size": opponent_hand_size,
        "pile_size": pile_size,
        "draw_pile_size": draw_pile_size,
        "turn_number": turn_number,
        "last_action": last_action,
        "cards_played": cards_played or [],
    }
