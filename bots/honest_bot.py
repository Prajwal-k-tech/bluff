"""HonestBot — pure baseline. Never bluffs.

Plays cards it actually has, claims the correct rank. Calls bluff using
simple probability. No learning, no strategy beyond honest play.

This is the control condition: "what happens if you never lie?"
"""

from typing import List, Tuple, Dict
from cards import Card, Rank
from game import Action
from bots.base import BotInterface


class HonestBot(BotInterface):
    """Never bluffs. Pure honest baseline."""

    def __init__(self):
        self.cards_seen: Dict[Rank, int] = {rank: 4 for rank in Rank}

    def reset(self):
        self.cards_seen = {rank: 4 for rank in Rank}

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, int] = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        # Play the rank we have most of (dump hand fast, honestly)
        best_rank = max(rank_counts.keys(), key=lambda r: rank_counts[r])
        count = min(rank_counts[best_rank], 4)
        cards = [c for c in hand if c.rank == best_rank][:count]

        return (cards, best_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """Call bluff using simple probability."""
        claimed_rank = last_action.claimed_rank
        claim_size = len(last_action.cards_played)

        remaining_rank = self.cards_seen.get(claimed_rank, 4)
        remaining_total = sum(self.cards_seen.values())

        if remaining_total == 0 or remaining_rank < claim_size:
            return True

        p_honest = remaining_rank / remaining_total
        for i in range(1, claim_size):
            p_honest *= (remaining_rank - i) / (remaining_total - i)

        p_bluff = 1.0 - p_honest
        return p_bluff > 0.5

    def observe_action(self, action: Action, opponent_hand_size: int):
        if action.bluff_called and action.caller_was_right:
            for card in action.cards_played:
                self.cards_seen[card.rank] = max(0, self.cards_seen.get(card.rank, 4) - 1)
        elif not action.was_bluff:
            for card in action.cards_played:
                self.cards_seen[card.rank] = max(0, self.cards_seen.get(card.rank, 4) - 1)

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
