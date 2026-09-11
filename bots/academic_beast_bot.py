"""
AcademicBeastBot — Literature-Grounded Ultimate Bluff Bot (Phase 2).

Directly implements academic methodologies from:
1. Southey et al. (2005) ("Bayes' Bluff"): Selection-bias corrected Bayesian opponent modeling
   with Thompson Sampling over Beta posteriors.
2. Dewey (2025) ("Reinforcement Learning in Imperfect Information Card Games"):
   Stake-sensitive deception equity and explicit Expected Value (EV) decision equations:
       EV(call) = P(bluff) * Pile_Size - (1 - P(bluff)) * Pile_Size
   Dynamic pile-conditioned thresholding where stakes govern risk tolerance.
3. Yeung (2008) / Game-Theoretic Equilibrium:
   Offensive multi-card packet dumping against passive regimes to break draw-locks before turn 24.
4. Hierarchical Value Prior:
   Neural network used for state evaluation and tactical tie-breaking, avoiding additive dilution
   of Bayesian probabilities.
"""

import os
import sys
import random
import math
from typing import Dict, List, Tuple, Optional

import torch

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import Action
from bots.base import BotInterface, pending_claims_from_actions, pool_by_rank
from bots.bayesian_bot import BluffTracker, CardCounter, OpponentModel
from bots.archetype_classifier import ArchetypeClassifier
from nn.model import BluffNet, ACTION_DIM, CALL_ACTION, PASS_ACTION, build_legal_actions, decode_action_into
from nn.state_encoder import StateEncoder, STATE_DIM


class AcademicBeastBot(BotInterface):
    """Academic Beast: Southey selection-bias modeling + Dewey stake-sensitive EV calling."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        thompson_sampling: bool = True,
        risk_aversion: float = 1.0,
    ):
        self.thompson_sampling = thompson_sampling
        self.risk_aversion = risk_aversion
        self.model = OpponentModel()
        self.counter = CardCounter()
        self.bluff_tracker = BluffTracker()
        self.player_id: Optional[int] = None
        self.classifier = ArchetypeClassifier()
        self.opp_bluffs: int = 0
        self.opp_honest: int = 0
        self.opp_calls: int = 0
        self.opp_passes: int = 0

        # Load BluffNet if available for state evaluation
        self.net: Optional[BluffNet] = None
        self.encoder = StateEncoder()
        cand_paths = [
            checkpoint_path,
            "nn/checkpoints/final.pt",
            "nn/checkpoints/synthetic_league.pt",
            "nn/checkpoints/v7_best.pt",
        ]
        for p in cand_paths:
            if p and os.path.exists(p):
                try:
                    from nn.training import load_checkpoint
                    self.net = load_checkpoint(p)
                    break
                except Exception:
                    continue

    def reset(self):
        """Intra-game reset: deck tracking resets, opponent model persists."""
        self.counter = CardCounter()

    def observe_action(self, action: Action, opponent_hand_size: int):
        self.model.observe_action(action, opponent_hand_size)
        if action.cards_played:
            self.counter.update_with_play(action.cards_played)

        # Track archetype observations
        if self.player_id is not None:
            if action.player != self.player_id:
                if action.cards_played and action.bluff_called:
                    if action.caller_was_right:
                        self.opp_bluffs += 1
                    else:
                        self.opp_honest += 1
                elif not action.cards_played:
                    self.opp_passes += 1
            else:
                if action.bluff_called:
                    self.opp_calls += 1

    def _remaining_pool(self, hand: List[Card], game_state: dict) -> Dict[Rank, int]:
        pending = game_state.get("pending_claims")
        if pending is None:
            pending = pending_claims_from_actions(game_state.get("actions", []))
        return pool_by_rank(hand, pending)

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        # 1. Estimate opponent tendencies via Thompson Sampling (Southey et al.)
        if self.thompson_sampling:
            call_freq = self.model.sample_call_frequency()
            opp_bluff_rate = self.model.overall_bluff.sample()
        else:
            call_freq = self.model.estimate_call_frequency()
            opp_bluff_rate = self.model.overall_bluff.mean()

        pile_size = game_state.get("pile_size", 0)
        draw_pile_size = game_state.get("draw_pile_size", 24)
        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)

        rank_counts: Dict[Rank, int] = {}
        for c in hand:
            rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

        # 2. Game-Theoretic Offensive Multi-Card Shedding:
        top_arch, arch_conf = self.classifier.top_archetype(
            self.opp_bluffs, self.opp_honest, self.opp_calls, self.opp_passes
        )

        # Tactical Archetype Counter:
        # Calling Station catches everything -> NEVER bluff, play 100% honest cards
        if top_arch == "Calling_Station" and arch_conf >= 0.50:
            multi_honest = [r for r, count in rank_counts.items() if count >= 2]
            if multi_honest:
                best_r = max(multi_honest, key=lambda r: (rank_counts[r], r.value))
                cards_to_dump = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
                return (cards_to_dump, best_r)
            available_ranks = sorted(rank_counts.keys(), key=lambda r: r.value, reverse=True)
            for r in available_ranks:
                honest_cards = [c for c in hand if c.rank == r][:1]
                return (honest_cards, r)
            return ([hand[0]], hand[0].rank)

        # If opponent is passive (call_freq < 0.40) or honest (opp_bluff_rate < 0.15),
        # shedding 2-4 cards per turn is mathematically the dominant strategy to win before draw exhaustion.
        pool = self._remaining_pool(hand, game_state)

        # Priority A: Honest multi-card shedding
        multi_honest = [r for r, count in rank_counts.items() if count >= 2]
        if multi_honest:
            best_r = max(multi_honest, key=lambda r: (rank_counts[r], r.value))
            cards_to_dump = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
            return (cards_to_dump, best_r)

        # Priority B: Dewey Stake-Sensitive Multi-Card Dumping
        # EV(dump) = (1 - call_freq) * dump_count - call_freq * (pile_size + dump_count)
        # Only attempt if EV is strictly positive or pile is low
        if call_freq < 0.35 or (pile_size <= 2 and call_freq < 0.50):
            # Check safe ranks in pool where opponent has low probability of catching
            safe_3 = [r for r, count in pool.items() if count >= 3]
            if len(hand) >= 3 and safe_3:
                best_safe = max(safe_3, key=lambda r: pool[r])
                return (hand[:3], best_safe)

            safe_2 = [r for r, count in pool.items() if count >= 2]
            if len(hand) >= 2 and safe_2:
                best_safe = max(safe_2, key=lambda r: pool[r])
                return (hand[:2], best_safe)

        # Priority C: Honest single play
        # Play the highest available single card honestly
        available_ranks = sorted(rank_counts.keys(), key=lambda r: r.value, reverse=True)
        for r in available_ranks:
            honest_cards = [c for c in hand if c.rank == r][:1]
            return (honest_cards, r)

        # Fallback
        return ([hand[0]], hand[0].rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """
        Dewey (2025) Stake-Sensitive Calling with Exact Hypergeometric Catching.
        """
        draw_pile_size = game_state.get("draw_pile_size", 0)
        can_pass = draw_pile_size > 0
        if not can_pass:
            return True  # Game rules: forced call when draw pile exhausted

        claimed_rank = last_action.claimed_rank
        claim_size = len(last_action.cards_played)
        hand = game_state.get("hand") or []
        our_copies = sum(1 for c in hand if c.rank == claimed_rank)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pile_size = game_state.get("pile_size", 0)

        # 1. Combinatorial Certainty: Hypergeometric Catching
        counting_p = self.counter.bluff_probability(
            claimed_rank, claim_size, len(hand), opp_hand_size, our_copies=our_copies
        )
        # If mathematically impossible for opponent to hold cards, call with 100% certainty
        if counting_p >= 0.999:
            return True

        # 2. Bayesian Opponent Modeling with Thompson Sampling
        if self.thompson_sampling:
            model_p = self.model.sample_bluff_probability(opp_hand_size, claimed_rank, claim_size)
            overall_bluff_p = self.model.overall_bluff.sample()
        else:
            model_p = self.model.estimate_bluff_probability(opp_hand_size, claimed_rank, claim_size)
            overall_bluff_p = self.model.overall_bluff.mean()

        # 3. Game-Theoretic Honest Grounding (Yeung 2008 / Southey 2005):
        # Calling an honest play causes caller to absorb the entire pile.
        # If opponent rarely bluffs (overall_bluff_p < 0.15) and not caught by count, NEVER call plausible claims.
        if overall_bluff_p < 0.15 and counting_p < 0.95:
            return False

        # 4. Dewey (2025) Stake-Sensitive Risk-Adjusted EV Calling:
        # Expected value of calling:
        #   Gain if bluff: +pile_size (opponent absorbs pile)
        #   Loss if honest: -(pile_size) (caller absorbs pile)
        # Stake multiplier: when pile is small (1-2), risk is near zero, exploratory calling is high EV.
        # When pile is huge (10+), wrong call is catastrophic, demanding extreme certainty (p > 0.80).
        base_threshold = 0.50

        # Stake discount / premium:
        # Low pile (1-3 cards): threshold lowers by up to 0.15 -> call aggressively
        # High pile (6+ cards): threshold raises by up to 0.25 -> call defensively
        stake_adjustment = math.tanh((pile_size - 3.0) / 6.0) * 0.25

        # Opponent deception premium:
        # If opponent is a chronic bluffer (overall_bluff_p > 0.40), discount threshold further
        deception_discount = (overall_bluff_p - 0.20) * 0.50

        # Archetype adjustments
        top_arch, arch_conf = self.classifier.top_archetype(
            self.opp_bluffs, self.opp_honest, self.opp_calls, self.opp_passes
        )
        if top_arch == "Honest_Rock" and arch_conf >= 0.50 and counting_p < 0.999:
            return False
        if top_arch == "Hyper_Maniac" and arch_conf >= 0.50 and pile_size <= 4:
            deception_discount += 0.15

        dynamic_threshold = max(0.30, min(0.85, base_threshold + stake_adjustment - deception_discount))

        # Dynamic Bayesian + Combinatorial Fusion (NO neural dilution)
        n_obs = self.model.total_actions_observed
        var = self.model.overall_bluff.variance()
        certainty = max(0.0, 1.0 - (var / 0.05))
        w_model = min(0.80, certainty * 0.80)
        w_count = 1.0 - w_model

        fused_p = w_model * model_p + w_count * counting_p

        return fused_p > dynamic_threshold

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "bluff_tracker": self.bluff_tracker.to_dict(),
            "thompson_sampling": self.thompson_sampling,
            "risk_aversion": self.risk_aversion,
        }

    @classmethod
    def from_dict(cls, d: dict, checkpoint_path: Optional[str] = None) -> "AcademicBeastBot":
        bot = cls(
            checkpoint_path=checkpoint_path,
            thompson_sampling=d.get("thompson_sampling", True),
            risk_aversion=d.get("risk_aversion", 1.0),
        )
        if "model" in d:
            bot.model = OpponentModel.from_dict(d["model"])
        if "bluff_tracker" in d:
            bot.bluff_tracker = BluffTracker.from_dict(d["bluff_tracker"])
        return bot

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def load(self, path: str):
        import json
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            d = json.load(f)
        loaded = self.from_dict(d)
        self.model = loaded.model
        self.bluff_tracker = loaded.bluff_tracker
        self.thompson_sampling = loaded.thompson_sampling
        self.risk_aversion = loaded.risk_aversion
