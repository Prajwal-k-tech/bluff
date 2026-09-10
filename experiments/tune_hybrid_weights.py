"""
Bayesian Weight Optimization & Dynamic Exploitation Experiment.
Compares four distinct Bayesian-to-NN weighting regimes for HybridBot:
1. Baseline (Conservative): w_model_cap = 0.40, exploit_mult = 1.5, call_mult = 1.2
2. Moderate Bayesian:       w_model_cap = 0.60, exploit_mult = 2.2, call_mult = 1.6
3. "Bluff Beast" (Dominant): w_model_cap = 0.75, exploit_mult = 3.0, call_mult = 2.0 (aggressive exploitation)
4. Dynamic Variance-Scaled:  w_model dynamically scales with inverse variance of Beta posterior

Evaluates each regime over 100 games per matchup (400 games per regime, 1,600 total across 4 regimes)
with strict 50/50 seat alternation against Honest, CardCount, Bayesian, and Random opponents.
"""

import sys
import os
import time
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn.functional as F

from cards import Card, Rank
from game import Action
from bots.hybrid_bot import HybridBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.random_bot import RandomBot
from test_bots import play_bot_vs_bot
from nn.model import ACTION_DIM, CALL_ACTION, PASS_ACTION, build_legal_actions, decode_action_into

REGIMES = [
    {
        "name": "Baseline (0.40)",
        "w_model_cap": 0.40,
        "exploit_mult": 1.5,
        "call_mult": 1.2,
        "variance_scaled": False,
    },
    {
        "name": "Moderate (0.60)",
        "w_model_cap": 0.60,
        "exploit_mult": 2.2,
        "call_mult": 1.6,
        "variance_scaled": False,
    },
    {
        "name": "Bluff Beast (0.75)",
        "w_model_cap": 0.75,
        "exploit_mult": 3.0,
        "call_mult": 2.0,
        "variance_scaled": False,
    },
    {
        "name": "Variance-Adaptive",
        "w_model_cap": 0.80,
        "exploit_mult": 2.5,
        "call_mult": 1.8,
        "variance_scaled": True,
    },
]

class ParameterizedHybridBot(HybridBot):
    """Hybrid bot parameterized with variable Bayesian weight caps and exploitation multipliers."""

    def __init__(
        self,
        w_model_cap: float = 0.40,
        exploit_mult: float = 1.5,
        call_mult: float = 1.2,
        variance_scaled: bool = False,
        checkpoint_path: Optional[str] = None,
        thompson_sampling: bool = True,
    ):
        super().__init__(checkpoint_path=checkpoint_path, thompson_sampling=thompson_sampling)
        self.w_model_cap = w_model_cap
        self.exploit_mult = exploit_mult
        self.call_mult = call_mult
        self.variance_scaled = variance_scaled

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

        confidence = min(self.w_model_cap, self.model.total_actions_observed / 30.0)

        # Multi-card shedding when facing passive or honest opponents
        if call_freq < 0.35 or overall_bluff_p < 0.15:
            rank_counts: Dict[Rank, int] = {}
            for c in hand:
                rank_counts[c.rank] = rank_counts.get(c.rank, 0) + 1

            multi = [r for r, count in rank_counts.items() if count >= 2]
            if multi:
                best_r = max(multi, key=lambda r: (rank_counts[r], r.value))
                matching_cards = [c for c in hand if c.rank == best_r][:min(4, rank_counts[best_r])]
                return (matching_cards, best_r)

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

        # Game-theoretic threshold guard
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

        call_thresh = max(0.35, min(0.85, 0.70 - (overall_p - 0.20) * self.call_mult))
        return fused_bluff_prob > call_thresh


def run_optimization_experiment():
    print("=================================================================")
    print("      Bayesian Weight Optimization & Tuning Experiment           ")
    print("=================================================================")

    opponent_classes = [
        ("HonestBot", HonestBot),
        ("CardCountBot", CardCountBot),
        ("BayesianBot", BayesianBot),
        ("RandomBot", RandomBot),
    ]

    games_per_opponent = 100  # 50 seats as p0, 50 seats as p1 = 100 games * 4 opps = 400 games per regime
    results = {}

    for regime in REGIMES:
        r_name = regime["name"]
        print(f"\nEvaluating Regime: {r_name} (w_cap={regime['w_model_cap']}, exploit_mult={regime['exploit_mult']}, call_mult={regime['call_mult']})...")

        total_wins = 0
        total_losses = 0
        total_draws = 0
        opp_breakdown = {}

        t0 = time.time()
        for opp_name, OppClass in opponent_classes:
            w, l, d = 0, 0, 0
            for g in range(games_per_opponent):
                seat = 0 if g % 2 == 0 else 1
                bot = ParameterizedHybridBot(
                    w_model_cap=regime["w_model_cap"],
                    exploit_mult=regime["exploit_mult"],
                    call_mult=regime["call_mult"],
                    variance_scaled=regime["variance_scaled"],
                )
                opp_instance = OppClass()

                if seat == 0:
                    winner = play_bot_vs_bot(bot, opp_instance, max_turns=100)
                    if winner == 0:
                        w += 1
                    elif winner == 1:
                        l += 1
                    else:
                        d += 1
                else:
                    winner = play_bot_vs_bot(opp_instance, bot, max_turns=100)
                    if winner == 1:
                        w += 1
                    elif winner == 0:
                        l += 1
                    else:
                        d += 1

            total_wins += w
            total_losses += l
            total_draws += d
            opp_breakdown[opp_name] = (w, l, d)
            print(f"   vs {opp_name:12s}: {w:2d}W - {l:2d}L - {d:2d}D")

        duration = time.time() - t0
        net_score = total_wins - total_losses
        total_games = total_wins + total_losses + total_draws
        win_rate = (total_wins / total_games) * 100.0 if total_games > 0 else 0.0
        results[r_name] = {
            "wins": total_wins,
            "losses": total_losses,
            "draws": total_draws,
            "net": net_score,
            "win_rate": win_rate,
            "opp_breakdown": opp_breakdown,
            "duration": duration,
        }
        print(f"   => TOTAL: {total_wins}W - {total_losses}L - {total_draws}D | Net Score: {net_score:+d} | Win Rate: {win_rate:.1f}% ({duration:.1f}s)")

    print("\n=================================================================")
    print("                     Optimization Summary Table                  ")
    print("=================================================================")
    print(f"{'Regime':<22s} | {'Wins':<5s} | {'Losses':<6s} | {'Draws':<5s} | {'Net':<6s} | {'Win Rate':<8s}")
    print("-" * 65)
    for r_name, data in results.items():
        print(f"{r_name:<22s} | {data['wins']:<5d} | {data['losses']:<6d} | {data['draws']:<5d} | {data['net']:<+6d} | {data['win_rate']:>7.1f}%")
    print("=================================================================")

    # Determine best regime
    best_regime = max(results.items(), key=lambda item: (item[1]["net"], item[1]["wins"]))
    print(f"\nOptimal Regime: {best_regime[0]} with Net Score: {best_regime[1]['net']:+d} (Win Rate: {best_regime[1]['win_rate']:.1f}%)")

    return results

if __name__ == "__main__":
    run_optimization_experiment()
