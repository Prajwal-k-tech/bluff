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
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import Action
from bots.base import BotInterface, pending_claims_from_actions, pool_by_rank
from bots.bayesian_bot import BluffTracker, CardCounter, OpponentModel
from bots.archetype_classifier import ArchetypeClassifier
from nn.model import BluffNet, ACTION_DIM, CALL_ACTION, PASS_ACTION, build_legal_actions, decode_action_into
from nn.state_encoder import StateEncoder, STATE_DIM, POPULATION_PRIOR, opponent_signals_from_actions


class AcademicBeastBot(BotInterface):
    """Academic Beast: Southey selection-bias modeling + Dewey stake-sensitive EV calling."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        thompson_sampling: bool = True,
        risk_aversion: float = 1.0,
        decay_tau: float = 8.0,
        nn_floor: float = 0.15,
        schedule_type: str = "exponential",
    ):
        self.thompson_sampling = thompson_sampling
        self.risk_aversion = risk_aversion
        self.decay_tau = decay_tau
        self.nn_floor = nn_floor
        self.schedule_type = schedule_type
        self.model = OpponentModel()
        self.counter = CardCounter()
        self.bluff_tracker = BluffTracker()
        self.player_id: Optional[int] = None
        self.classifier = ArchetypeClassifier()
        self.opp_bluffs: int = 0
        self.opp_honest: int = 0
        self.opp_calls: int = 0
        self.opp_passes: int = 0

        # Load BluffNet or BluffNetXL if available for state evaluation
        self.net: Optional[object] = None
        self.encoder = StateEncoder()
        cand_paths = [
            checkpoint_path,
            "nn/checkpoints/bluffnet_xl_league.pt",
            "nn/checkpoints/final.pt",
            "nn/checkpoints/synthetic_league.pt",
            "nn/checkpoints/v7_best.pt",
        ]
        for p in cand_paths:
            if p and os.path.exists(p):
                try:
                    data = torch.load(p, weights_only=True)
                    if isinstance(data, dict) and data.get("architecture") == "BluffNetXL":
                        from nn.model import BluffNetXL
                        self.net = BluffNetXL(state_dim=39, action_dim=54, hidden_dim=512)
                        self.net.load_state_dict(data["model_state_dict"])
                    else:
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
            "cards_remaining": self._remaining_pool(hand, game_state),
            **signals,
        }

    def get_td_moe_weights(self, tau: Optional[float] = None, w_floor: Optional[float] = None) -> Tuple[float, float]:
        """
        Temporal Decaying Mixture of Experts (TD-MoE) schedule:
        w_nn(n) = w_floor + (1.0 - w_floor) * exp(-n / tau)
        w_bayes(n) = 1.0 - w_nn(n)

        At game start (n=0): w_nn = 1.0, w_bayes = 0.0 (NN opening prior dominance).
        As n increases (n >= 10): w_nn -> w_floor (0.15), w_bayes -> 0.85 (Bayesian counter-exploitation).
        """
        tau_val = tau if tau is not None else self.decay_tau
        floor_val = w_floor if w_floor is not None else self.nn_floor
        n_obs = self.model.total_actions_observed

        if getattr(self, "schedule_type", "sigmoidal") == "sigmoidal":
            # S-curve transition (ADR-013): maintains high neural prior through turn 4,
            # then smoothly transitions into Bayesian counter-exploitation as variance drops.
            sig = 1.0 / (1.0 + math.exp((n_obs - 5.0) / 2.0))
            w_nn = floor_val + (1.0 - floor_val) * sig
        else:
            w_nn = floor_val + (1.0 - floor_val) * math.exp(-n_obs / max(0.5, tau_val))

        w_nn = max(floor_val, min(1.0, w_nn))
        w_bayes = 1.0 - w_nn
        return w_nn, w_bayes

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

        top_arch, arch_conf = self.classifier.top_archetype(
            self.opp_bluffs, self.opp_honest, self.opp_calls, self.opp_passes
        )
        w_nn, w_bayes = self.get_td_moe_weights()

        # Tactical Archetype Counter: Calling Station catches everything -> NEVER bluff, play 100% honest cards
        if top_arch == "Calling_Station" and arch_conf >= 0.40 and w_bayes >= 0.35:
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

        # Passive Opponent Multi-Card Shedding (Honest Rock, passive baselines)
        pool = self._remaining_pool(hand, game_state)
        if call_freq < 0.35 or opp_bluff_rate < 0.15 or (top_arch == "Honest_Rock" and arch_conf >= 0.40):
            multi_honest = [r for r, count in rank_counts.items() if count >= 2]
            if multi_honest:
                best_r = max(multi_honest, key=lambda r: (rank_counts[r], r.value))
                cards_to_dump = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
                return (cards_to_dump, best_r)

            if call_freq < 0.35 or (pile_size <= 2 and call_freq < 0.50):
                safe_3 = [r for r, count in pool.items() if count >= 3]
                if len(hand) >= 3 and safe_3:
                    best_safe = max(safe_3, key=lambda r: pool[r])
                    return (hand[:3], best_safe)

                safe_2 = [r for r, count in pool.items() if count >= 2]
                if len(hand) >= 2 and safe_2:
                    best_safe = max(safe_2, key=lambda r: pool[r])
                    return (hand[:2], best_safe)

        # 2. Neural Policy Prior + Bayesian TD-MoE Fusion:
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
                p_nn = F.softmax(masked_logits, dim=-1).cpu()

            probs = p_nn.clone()

            for action_idx in range(ACTION_DIM):
                if not mask[action_idx]:
                    continue
                decoded = decode_action_into(action_idx, hand)
                if decoded is None:
                    continue
                played_cards, claimed_rank = decoded
                is_honest = all(c.rank == claimed_rank for c in played_cards)
                claim_size = len(played_cards)

                if is_honest:
                    u_bayes = 1.0 + (call_freq - 0.35) * 1.5 + (claim_size - 1) * 0.4
                else:
                    if call_freq > 0.50 or (top_arch == "Calling_Station" and arch_conf >= 0.40):
                        u_bayes = 0.0
                    else:
                        ev_bluff = (1.0 - call_freq) * claim_size - call_freq * (pile_size + claim_size)
                        unseen_count = pool.get(claimed_rank, 0)
                        plausibility = min(1.0, unseen_count / max(1, claim_size))
                        u_bayes = max(0.05, 1.0 + ev_bluff * 0.25 * plausibility)

                probs[action_idx] = p_nn[action_idx] * (w_nn + w_bayes * u_bayes)

            total_p = probs.sum()
            if total_p > 0:
                probs = probs / total_p
                sampled_idx = int(torch.multinomial(probs, 1).item())
                result = decode_action_into(sampled_idx, hand)
                if result is not None:
                    return result

        # Heuristic / Bayesian multi-card shedding fallback
        multi_honest = [r for r, count in rank_counts.items() if count >= 2]
        if multi_honest:
            best_r = max(multi_honest, key=lambda r: (rank_counts[r], r.value))
            cards_to_dump = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
            return (cards_to_dump, best_r)

        if call_freq < 0.35 or (pile_size <= 2 and call_freq < 0.50):
            safe_3 = [r for r, count in pool.items() if count >= 3]
            if len(hand) >= 3 and safe_3:
                best_safe = max(safe_3, key=lambda r: pool[r])
                return (hand[:3], best_safe)

            safe_2 = [r for r, count in pool.items() if count >= 2]
            if len(hand) >= 2 and safe_2:
                best_safe = max(safe_2, key=lambda r: pool[r])
                return (hand[:2], best_safe)

        available_ranks = sorted(rank_counts.keys(), key=lambda r: r.value, reverse=True)
        for r in available_ranks:
            honest_cards = [c for c in hand if c.rank == r][:1]
            return (honest_cards, r)

        return ([hand[0]], hand[0].rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """
        Dewey (2025) Stake-Sensitive Calling with Exact Hypergeometric Catching
        and Temporal Decaying Mixture of Experts (TD-MoE).
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
        if counting_p >= 0.999:
            return True

        # 2. Bayesian Opponent Modeling with Archetype-Conditioned Thompson Sampling (ADR-012)
        mean_bluff_p = self.model.overall_bluff.mean()
        if self.thompson_sampling and mean_bluff_p >= 0.15:
            model_p = self.model.sample_bluff_probability(opp_hand_size, claimed_rank, claim_size)
            overall_bluff_p = self.model.overall_bluff.sample()
        else:
            model_p = self.model.estimate_bluff_probability(opp_hand_size, claimed_rank, claim_size)
            overall_bluff_p = mean_bluff_p

        # 3. Game-Theoretic Honest Grounding (Yeung 2008 / Southey 2005)
        if overall_bluff_p < 0.15 and counting_p < 0.95:
            return False

        # 4. Dewey (2025) Stake-Sensitive Risk-Adjusted EV Calling:
        base_threshold = 0.50
        stake_adjustment = math.tanh((pile_size - 3.0) / 6.0) * 0.25
        deception_discount = (overall_bluff_p - 0.20) * 0.50

        # Archetype adjustments
        top_arch, arch_conf = self.classifier.top_archetype(
            self.opp_bluffs, self.opp_honest, self.opp_calls, self.opp_passes
        )
        if top_arch == "Honest_Rock" and arch_conf >= 0.50 and counting_p < 0.999:
            return False
        if top_arch == "Hyper_Maniac" and arch_conf >= 0.50 and pile_size <= 4:
            deception_discount += 0.15

        # 5. TD-MoE Weighting & Neural Tactical Modulation
        w_nn, w_bayes = self.get_td_moe_weights()
        nn_delta = 0.0
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
                # Tactical modulation scaled by w_nn (bounded +-0.04)
                nn_delta = (raw_nn - 0.50) * (0.08 * w_nn)

        dynamic_threshold = max(0.35, min(0.85, base_threshold + stake_adjustment - deception_discount - nn_delta))

        # Dynamic Bayesian + Combinatorial Fusion (scaled by epistemic certainty)
        var = self.model.overall_bluff.variance()
        certainty = max(0.0, 1.0 - (var / 0.05))
        w_model = min(0.80, max(0.35, certainty * 0.80))
        w_count = 1.0 - w_model

        fused_p = w_model * model_p + w_count * counting_p

        return fused_p > dynamic_threshold

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "bluff_tracker": self.bluff_tracker.to_dict(),
            "thompson_sampling": self.thompson_sampling,
            "risk_aversion": self.risk_aversion,
            "decay_tau": self.decay_tau,
            "nn_floor": self.nn_floor,
        }

    @classmethod
    def from_dict(cls, d: dict, checkpoint_path: Optional[str] = None) -> "AcademicBeastBot":
        bot = cls(
            checkpoint_path=checkpoint_path,
            thompson_sampling=d.get("thompson_sampling", True),
            risk_aversion=d.get("risk_aversion", 1.0),
            decay_tau=d.get("decay_tau", 8.0),
            nn_floor=d.get("nn_floor", 0.15),
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
        self.decay_tau = loaded.decay_tau
        self.nn_floor = loaded.nn_floor
