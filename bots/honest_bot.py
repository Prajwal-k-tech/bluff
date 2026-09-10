"""HonestBot — pure baseline. Never bluffs.

Plays cards it actually has, claims the correct rank. Calls bluff using
simple probability. No learning, no strategy beyond honest play.

This is the control condition: "what happens if you never lie?"
"""

from typing import List, Tuple, Dict
from cards import Card, Rank
from game import Action
from bots.base import (BotInterface, pool_by_rank, pending_total,
                       bluff_probability)
from bots.prob import Hypergeometric


class HonestBot(BotInterface):
    """Never bluffs. Pure honest baseline."""

    def reset(self):
        pass  # pool model is derived from the public action history each call

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
        """Call bluff using the shared pool-model P(bluff).

        P(bluff) = P(the opponent's before-hand held fewer than claim_size
        copies of the claimed rank), drawn from the pool of unseen cards:
        pool[r] = 4 − our copies of r − unresolved pile claims of r
        (docs/bot-modes.md "pool model"). Derived fresh from the public
        action history, so it self-heals when pile cards recycle into hands
        after a call.
        """
        p_bluff = bluff_probability(
            game_state.get("hand", []),
            game_state.get("pending_claims", {}),
            last_action,
            game_state.get("opponent_hand_size", 14),
        )
        return p_bluff > 0.6  # conservative threshold (docs/bot-modes.md)

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass  # pool model derives from action history; nothing to accumulate

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
