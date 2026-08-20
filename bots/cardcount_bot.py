"""CardCountBot — math-driven player with strategic bluffs.

Personality: "Analytical human" — uses exact hypergeometric for both
playing and calling. Bluffs when the expected value is positive.

Differentiation from HonestBot:
- Bluffs are math-driven, not desperation-driven
- Uses exact hypergeometric for bluff evaluation
- Higher bluff frequency when math says it's good
- More aggressive caller (lower threshold)
"""

import random
from typing import List, Tuple, Dict
from cards import Card, Rank
from game import Action
from bots.base import BotInterface
from bots.prob import Hypergeometric


class CardCountBot(BotInterface):
    """Rational agent. Uses exact math for both bluffing and calling."""

    def __init__(self):
        self.remaining_by_rank: Dict[Rank, int] = {rank: 4 for rank in Rank}
        self.total_remaining: int = 52
        self.bluff_count = 0
        self.call_count = 0

    def reset(self):
        self.remaining_by_rank = {rank: 4 for rank in Rank}
        self.total_remaining = 52

    def _update_counts(self, cards: List[Card]):
        """Remove cards from our tracking."""
        for card in cards:
            if self.remaining_by_rank[card.rank] > 0:
                self.remaining_by_rank[card.rank] -= 1
                self.total_remaining -= 1

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, int] = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pile_size = game_state.get("pile_size", 0)

        # Score every possible (rank, claim_size) combo
        best_score = -1
        best_rank = None
        best_cards: List[Card] = []

        for rank in Rank:
            have = rank_counts.get(rank, 0)
            for claim_size in range(1, min(4, hand_size) + 1):
                if claim_size > hand_size:
                    continue

                is_honest = claim_size <= have

                if is_honest:
                    # Honest play: score by how many cards we dump
                    score = claim_size * 0.3
                    # Bonus if few of this rank remain (opponent can't verify)
                    remaining = self.remaining_by_rank.get(rank, 0)
                    if remaining <= 1:
                        score += 0.5
                else:
                    # Bluff: only if math says it's good
                    remaining_rank = self.remaining_by_rank.get(rank, 0)
                    remaining_total = self.total_remaining - hand_size

                    if remaining_total <= 0:
                        score = 0.0
                    elif claim_size > remaining_rank:
                        # Impossible to have this many — risky bluff
                        score = 0.1
                    else:
                        # P(opponent can't disprove)
                        p_fewer = Hypergeometric.cdf(
                            claim_size - 1, remaining_total,
                            remaining_rank, opp_hand_size
                        )
                        score = p_fewer * 0.4  # Bluff quality

                    self.bluff_count += 1

                if score > best_score:
                    best_score = score
                    best_rank = rank
                    if is_honest:
                        best_cards = [c for c in hand if c.rank == rank][:claim_size]
                    else:
                        best_cards = hand[:claim_size]

        assert best_rank is not None
        return (best_cards, best_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """Call bluff using exact hypergeometric probability."""
        claimed_rank = last_action.claimed_rank
        claim_size = len(last_action.cards_played)

        our_hand_size = game_state.get("hand_size", 0)
        opp_hand_size = game_state.get("opponent_hand_size", 0)

        remaining_rank = self.remaining_by_rank.get(claimed_rank, 0)
        remaining_total = self.total_remaining - our_hand_size

        if remaining_total <= 0:
            return True

        if claim_size > remaining_rank:
            return True

        p_fewer = Hypergeometric.cdf(
            claim_size - 1,
            remaining_total,
            remaining_rank,
            opp_hand_size,
        )
        p_bluff = 1.0 - p_fewer

        # Aggressive caller: lower threshold than HonestBot
        self.call_count += 1
        return p_bluff > 0.4

    def observe_action(self, action: Action, opponent_hand_size: int):
        """Update card counts based on observed action."""
        if action.bluff_called and action.caller_was_right:
            self._update_counts(action.cards_played)
        elif action.bluff_called and not action.caller_was_right:
            self._update_counts(action.cards_played)
        elif not action.bluff_called and not action.was_bluff:
            self._update_counts(action.cards_played)

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
