"""CardCountBot — math-driven player with strategic bluffs.

Personality: "Analytical human" — uses exact hypergeometric for both
playing and calling. Bluffs when the expected value is positive.

Differentiation from HonestBot:
- Bluffs are math-driven, not desperation-driven
- Uses exact hypergeometric for bluff evaluation
- Higher bluff frequency when math says it's good
- More aggressive caller (lower threshold)

Card accounting uses the pool model (docs/bot-modes.md): unresolved pile
claims are tracked from the public action history, so counts self-heal when
pile cards recycle into hands after a call. The old monotone "cards seen"
counters hit zero for ranks still in play, which caused eternal call-wars.
"""

from typing import List, Tuple, Dict
from cards import Card, Rank
from game import Action
from bots.base import (BotInterface, pool_by_rank, pending_total,
                       bluff_probability, claim_plausibility)


class CardCountBot(BotInterface):
    """Rational agent. Uses exact math for both bluffing and calling."""

    def reset(self):
        pass  # pool model is derived from the public action history each call

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, int] = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pending = game_state.get("pending_claims", {})
        pool = pool_by_rank(hand, pending)

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
                    # Bonus if few copies remain unseen (hard to disprove)
                    if pool.get(rank, 0) <= 1:
                        score += 0.5
                else:
                    # Bluff: P(opponent can't disprove the claim)
                    p_survive = claim_plausibility(
                        hand, pending, rank, claim_size, opp_hand_size
                    )
                    score = p_survive * 0.4  # Bluff quality

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
        """Call bluff using the shared pool-model P(bluff).

        Aggressive caller: lower threshold than HonestBot.
        """
        p_bluff = bluff_probability(
            game_state.get("hand", []),
            game_state.get("pending_claims", {}),
            last_action,
            game_state.get("opponent_hand_size", 0),
        )
        return p_bluff > 0.4

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass  # pool model derives from action history; nothing to accumulate

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
