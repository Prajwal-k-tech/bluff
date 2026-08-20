"""RandomBot — zero-intelligence baseline.

Plays uniformly at random among legal moves. Calls bluff 50% of the time.
No learning, no strategy. Pure baseline for comparison.
"""

import random
from typing import List, Tuple
from cards import Card, Rank
from game import Action
from bots.base import BotInterface


class RandomBot(BotInterface):
    """Plays random legal moves. Calls bluff 50% of the time."""

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        # Pick a random rank to claim
        claimed_rank = random.choice(list(Rank))

        # Pick 1-4 cards from hand (must be ≤ hand size)
        max_play = min(4, len(hand))
        num_cards = random.randint(1, max_play)
        cards = random.sample(hand, num_cards)

        return (cards, claimed_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        return random.random() < 0.5

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass  # No learning

    def save(self, path: str):
        pass  # Nothing to save

    def load(self, path: str):
        pass  # Nothing to load
