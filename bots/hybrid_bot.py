"""HybridBot — PPO Policy Priors + Bayesian Opponent Modeling + Card Counting (Phase 4).

Combines:
1. PPO Neural Network (BluffNet) for strategic base policy distributions over legal actions.
2. Bayesian Opponent Modeling (Beta distributions) for tracking opponent bluffing & calling tendencies.
3. Hypergeometric Card Counting (with base.py v2 shrinkage) for exact combinatorial tracking of unseen cards.
4. Thompson Sampling & Three-Way Dynamic Fusion for calibrated exploitability-safe responses.
5. S3 Serialization (to_dict/from_dict) for persistent cross-session human adaptation.

Design Note on Card Counting vs Opponent Model:
- CardCounter is strictly intra-game deck state (tracking unseen cards from the 52-card deal).
  It intentionally resets between games upon reshuffle via reset().
- OpponentModel tracks cross-session persistent human traits (bluff/call frequencies).
  It is serialized across games and sessions via to_dict()/from_dict() into PostgreSQL.

Decision Fusion & Thompson Sampling:
- Per Southey et al. (2005) and Doshi-Velez et al., point-estimate Bayesian Belief Revision (BBR)
  weighted averaging is prone to lag and collapse against non-stationary human strategies.
- When `thompson_sampling=True`, the bot draws posterior samples from OpponentModel's Beta
  distributions (overall bluff, bluff-by-hand-size, bluff-by-rank, and call frequency).
  This preserves epistemic uncertainty during decision fusion and dynamic thresholding.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from bots.base import (
    BotInterface,
    pending_claims_from_actions,
    pool_by_rank,
)
from bots.bayesian_bot import BluffTracker, CardCounter, OpponentModel
from cards import Card, Rank
from game import Action
from nn.model import (
    ACTION_DIM,
    CALL_ACTION,
    PASS_ACTION,
    BluffNet,
    build_legal_actions,
    decode_action_into,
)
from nn.state_encoder import (
    POPULATION_PRIOR,
    STATE_DIM,
    StateEncoder,
    opponent_signals_from_actions,
)

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nn", "checkpoints", "final.pt",
)


class HybridBot(BotInterface):
    """Hybrid AI combining PPO policy priors with Bayesian adaptation."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        thompson_sampling: bool = True,
        w_model_cap: float = 0.75,
        exploit_mult: float = 2.5,
        call_mult: float = 1.8,
        variance_scaled: bool = True,
    ):
        path = checkpoint_path or os.environ.get("BLUFF_NN_CHECKPOINT", DEFAULT_CHECKPOINT)
        if not os.path.exists(path):
            candidates = ["final.pt", "v7_best.pt", "v7_latest.pt", "v5.pt"]
            for cand in candidates:
                alt = os.path.join(os.path.dirname(DEFAULT_CHECKPOINT), cand)
                if os.path.exists(alt):
                    path = alt
                    break

        self.checkpoint_path = path
        self.thompson_sampling = thompson_sampling
        self.w_model_cap = w_model_cap
        self.exploit_mult = exploit_mult
        self.call_mult = call_mult
        self.variance_scaled = variance_scaled
        self.net: Optional[BluffNet] = None
        self.encoder = StateEncoder()
        self.player_id: Optional[int] = None

        if os.path.exists(path):
            try:
                from nn.training import load_checkpoint
                self.net = load_checkpoint(path)
                if self.net.shared[0].in_features != STATE_DIM:
                    print(f"[HybridBot] WARNING: Checkpoint dim {self.net.shared[0].in_features} != {STATE_DIM}; fallback active.")
                    self.net = None
            except Exception as e:
                print(f"[HybridBot] WARNING: Could not load checkpoint from {path}: {e}")
                self.net = None

        self.model = OpponentModel()
        self.counter = CardCounter()
        self.bluff_tracker = BluffTracker()
        self.bluff_threshold = 0.55
        self.call_threshold = 0.50

    def reset(self):
        self.counter = CardCounter()

    def _remaining_counts(self, hand: List[Card], game_state: dict) -> dict:
        pending = game_state.get("pending_claims")
        if pending is None:
            pending = pending_claims_from_actions(game_state.get("actions", []))
        return pool_by_rank(hand, pending)

    def _context(self, hand: List[Card], game_state: dict) -> dict:
        actions = game_state.get("actions") or []
        viewer = getattr(self, "player_id", None)
        if actions and viewer is not None:
            signals = opponent_signals_from_actions(actions, viewer)
        else:
            keys = ("opponent_call_rate", "opponent_bluff_revealed",
                    "my_bluff_rate", "my_bluff_success_rate")
            signals = dict(zip(keys, POPULATION_PRIOR))

        return {
            "opponent_hand_size": game_state.get("opponent_hand_size", 14),
            "pile_size": game_state.get("pile_size", 0),
            "draw_pile_size": game_state.get("draw_pile_size", 24),
            "turn_number": game_state.get("turn_number", 0),
            "can_pass": game_state.get("can_pass", True),
            "last_action": game_state.get("last_action"),
            "cards_remaining": self._remaining_counts(hand, game_state),
            **signals,
        }

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        if self.thompson_sampling:
            call_freq = self.model.sample_call_frequency()
            overall_bluff_p = self.model.overall_bluff.sample()
        else:
            call_freq = self.model.estimate_call_frequency()
            opp_bluff_rate = getattr(self.model, "overall_bluff", None)
            overall_bluff_p = opp_bluff_rate.mean() if opp_bluff_rate else 0.20
        confidence = min(self.w_model_cap, self.model.total_actions_observed / 25.0)

        # Multi-card shedding when facing passive or honest opponents (call_freq < 0.35 or overall_bluff_p < 0.15).
        # In Cheat, shedding >= 2 cards per turn is the only way to reduce hand size
        # and win before the draw-pile exhausts at turn 24.
        if call_freq < 0.35 or overall_bluff_p < 0.15:
            rank_counts: Dict[Rank, int] = {}
            for c in hand:
                rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

            # Honest multi-card plays first
            multi = [r for r, count in rank_counts.items() if count >= 2]
            if multi:
                best_r = max(multi, key=lambda r: (rank_counts[r], r.value))
                matching_cards = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
                return (matching_cards, best_r)

            # Plausible multi-card shedding via unseen pool
            pool = self._remaining_counts(hand, game_state)
            safe_3 = [r for r, count in pool.items() if count >= 3]
            if len(hand) >= 3 and safe_3:
                best_safe = max(safe_3, key=lambda r: pool[r])
                return (hand[:3], best_safe)

            safe_2 = [r for r, count in pool.items() if count >= 2]
            if len(hand) >= 2 and safe_2:
                best_safe = max(safe_2, key=lambda r: pool[r])
                return (hand[:2], best_safe)

        if self.net is not None:
            state = self.encoder.encode(hand, self._context(hand, game_state))
            mask = build_legal_actions(hand, can_call=False, can_pass=False, respond_only=False)
            device = next(self.net.parameters()).device
            state = state.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                logits, _ = self.net.forward(state)
                masked_logits = logits.clone()
                masked_logits[~mask] = float("-inf")
                probs = F.softmax(masked_logits, dim=-1).cpu()

            rank_counts = {}
            for c in hand:
                rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

            for action_idx in range(ACTION_DIM):
                if not mask[action_idx]:
                    continue
                decoded = decode_action_into(action_idx, hand)
                if decoded is None:
                    continue
                played_cards, claimed_rank = decoded
                is_honest = all(c.rank == claimed_rank for c in played_cards)

                if is_honest:
                    if call_freq > 0.50:
                        probs[action_idx] *= (1.0 + confidence * (call_freq - 0.50) * self.exploit_mult)
                else:
                    if call_freq > 0.50:
                        probs[action_idx] *= max(0.02, 1.0 - confidence * (call_freq - 0.50) * (self.exploit_mult * 1.2))
                    elif call_freq < 0.35:
                        probs[action_idx] *= (1.0 + confidence * (0.35 - call_freq) * (self.exploit_mult * 1.33))

            total_p = probs.sum()
            if total_p > 0:
                probs = probs / total_p
                sampled_idx = int(torch.multinomial(probs, 1).item())
                result = decode_action_into(sampled_idx, hand)
                if result is not None:
                    return result

        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        pile_size = game_state.get("pile_size", 0)

        should_bluff = self.bluff_tracker.should_bluff(
            hand_size, opp_hand_size, pile_size, call_freq
        )

        rank_counts = {}
        for card in hand:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        if should_bluff:
            return self._heuristic_bluff_play(hand, rank_counts, game_state)
        return self._heuristic_honest_play(hand, rank_counts)

    def _heuristic_honest_play(self, hand: List[Card], rank_counts: Dict[Rank, int]) -> Tuple[List[Card], Rank]:
        best_rank = max(rank_counts, key=lambda r: (rank_counts[r], r.value))
        matching_cards = [c for c in hand if c.rank == best_rank][:4]
        return (matching_cards, best_rank)

    def _heuristic_bluff_play(self, hand: List[Card], rank_counts: Dict[Rank, int],
                              game_state: dict) -> Tuple[List[Card], Rank]:
        hand_size = len(hand)
        opp_hand_size = game_state.get("opponent_hand_size", 10)
        singles = [r for r, count in rank_counts.items() if count == 1]
        doubles = [r for r, count in rank_counts.items() if count == 2]

        if doubles:
            target_rank = doubles[0]
            honest_cards = [c for c in hand if c.rank == target_rank]
            other_cards = [c for c in hand if c.rank != target_rank]
            if other_cards:
                return (honest_cards + other_cards[:1], target_rank)

        if singles:
            dump_card = [c for c in hand if c.rank == singles[0]][:1]
            plausible_ranks = [r for r in Rank if r not in rank_counts]
            if plausible_ranks:
                best_rank = max(
                    plausible_ranks,
                    key=lambda r: self.counter.claim_plausibility(r, 1, hand_size, opp_hand_size)
                )
                return (dump_card, best_rank)

        return self._heuristic_honest_play(hand, rank_counts)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        draw_pile_size = game_state.get("draw_pile_size", 0)
        can_pass = draw_pile_size > 0

        if not can_pass:
            return True

        claimed_rank = last_action.claimed_rank
        claim_size = len(last_action.cards_played)
        hand = game_state.get("hand") or []
        our_copies = sum(1 for c in hand if c.rank == claimed_rank)
        opp_hand_size = game_state.get("opponent_hand_size", 10)

        counting_p = self.counter.bluff_probability(
            claimed_rank, claim_size, len(hand), opp_hand_size, our_copies=our_copies
        )
        if counting_p >= 0.999:
            return True

        if self.thompson_sampling:
            model_p = self.model.sample_bluff_probability(
                opp_hand_size, claimed_rank, claim_size
            )
            overall_p = self.model.overall_bluff.sample()
        else:
            model_p = self.model.estimate_bluff_probability(
                opp_hand_size, claimed_rank, claim_size
            )
            opp_bluff_rate = getattr(self.model, "overall_bluff", None)
            overall_p = opp_bluff_rate.mean() if opp_bluff_rate else model_p

        # Game-Theoretic Threshold Adaptation (Yeung 2008 / Southey et al. 2005):
        # Calling an honest play causes caller to absorb the pile (disastrous penalty).
        # If opponent rarely bluffs (overall_p < 0.15), calling plausible claims has
        # strictly negative expected value. Only call if mathematically caught (counting_p >= 0.95).
        if overall_p < 0.15 and counting_p < 0.95:
            return False

        nn_p = 0.50
        if self.net is not None:
            state = self.encoder.encode(hand, self._context(hand, game_state))
            mask = build_legal_actions(hand, can_call=True, can_pass=can_pass, respond_only=True)
            device = next(self.net.parameters()).device
            state = state.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                logits, _ = self.net.forward(state)
                call_logit = logits[CALL_ACTION].item()
                pass_logit = logits[PASS_ACTION].item()
                exp_call = max(1e-6, float(torch.exp(torch.tensor(call_logit))))
                exp_pass = max(1e-6, float(torch.exp(torch.tensor(pass_logit))))
                raw_nn = exp_call / (exp_call + exp_pass)
                if overall_p < 0.25:
                    nn_p = raw_nn * (overall_p / 0.25)
                else:
                    nn_p = raw_nn

        n_obs = self.model.total_actions_observed
        if self.variance_scaled:
            var = self.model.overall_bluff.variance()
            certainty = max(0.0, 1.0 - (var / 0.05))
            w_model = min(self.w_model_cap, certainty * self.w_model_cap)
        else:
            w_model = min(self.w_model_cap, n_obs / 25.0)

        w_count = 0.35 * (1.0 - w_model * 0.4)
        w_nn = max(0.10, 1.0 - w_model - w_count)

        fused_bluff_prob = (
            w_nn * nn_p +
            w_model * model_p +
            w_count * counting_p
        )

        # Dynamic calling threshold:
        # Chronic bluffers (Random) -> threshold drops (call aggressively)
        # Honest/conservative players -> threshold rises up to 0.85
        call_thresh = max(0.35, min(0.85, 0.70 - (overall_p - 0.20) * self.call_mult))
        return fused_bluff_prob > call_thresh

    def observe_action(self, action: Action, opponent_hand_size: int):
        self.model.observe_action(action, opponent_hand_size)
        if action.cards_played:
            self.counter.update_with_play(action.cards_played)

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "bluff_tracker": self.bluff_tracker.to_dict(),
            "bluff_threshold": self.bluff_threshold,
            "call_threshold": self.call_threshold,
            "thompson_sampling": self.thompson_sampling,
            "w_model_cap": self.w_model_cap,
            "exploit_mult": self.exploit_mult,
            "call_mult": self.call_mult,
            "variance_scaled": self.variance_scaled,
        }

    @classmethod
    def from_dict(cls, d: dict, checkpoint_path: Optional[str] = None) -> "HybridBot":
        thompson = d.get("thompson_sampling", True)
        w_cap = d.get("w_model_cap", 0.75)
        exploit_m = d.get("exploit_mult", 2.5)
        call_m = d.get("call_mult", 1.8)
        var_sc = d.get("variance_scaled", True)
        bot = cls(
            checkpoint_path=checkpoint_path,
            thompson_sampling=thompson,
            w_model_cap=w_cap,
            exploit_mult=exploit_m,
            call_mult=call_m,
            variance_scaled=var_sc,
        )
        if "model" in d:
            bot.model = OpponentModel.from_dict(d["model"])
        if "bluff_tracker" in d:
            bot.bluff_tracker = BluffTracker.from_dict(d["bluff_tracker"])
        if "bluff_threshold" in d:
            bot.bluff_threshold = d["bluff_threshold"]
        if "call_threshold" in d:
            bot.call_threshold = d["call_threshold"]
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
        loaded = self.from_dict(d, checkpoint_path=self.checkpoint_path)
        self.model = loaded.model
        self.bluff_tracker = loaded.bluff_tracker
        self.bluff_threshold = loaded.bluff_threshold
        self.call_threshold = loaded.call_threshold
        self.thompson_sampling = loaded.thompson_sampling
        self.w_model_cap = loaded.w_model_cap
        self.exploit_mult = loaded.exploit_mult
        self.call_mult = loaded.call_mult
        self.variance_scaled = loaded.variance_scaled
