"""
Empirical Component Ablation: Isolating the Neural Network's Value vs Pure Bayesian Reasoning.

Directly tests the research question:
"Does the Neural Network add any benefit over pure Bayesian opponent modeling and card counting,
or is a pure Bayesian / heuristic architecture superior?"

Compares 4 distinct decision architectures:
1. Full_Hybrid:      NN Policy + Bayesian Opponent Modeling + Card Counting
2. NoNN_Hybrid:      Bayesian Opponent Modeling + Card Counting Only (w_nn = 0, no neural net)
3. Pure_Bayesian:    Standard BayesianBot (Beta distributions without PPO)
4. Pure_NN:          PureNNBot (PPO policy network alone without Bayesian adaptation)

Evaluates each across 6 diverse opponent types (180 games per architecture, 720 games total)
with strict 50/50 seat alternation:
- HonestBot
- CardCountBot
- BayesianBot
- RandomBot
- Calling_Station (Persona: p_b=0.10, p_c=0.85)
- Hyper_Maniac    (Persona: p_b=0.65, p_c=0.60)
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import Action
from bots.base import BotInterface
from bots.hybrid_bot import HybridBot
from bots.bayesian_bot import BayesianBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.random_bot import RandomBot
from bots.pure_nn_bot import PureNNBot
from test_bots import play_bot_vs_bot
from experiments.synthetic_population_eval import PersonaBot


class NoNNHybridBot(HybridBot):
    """HybridBot with the Neural Network completely ablated (w_nn = 0)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Force neural net to None so no neural features or logits are computed
        self.net = None

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

        # Game-theoretic guard: never call plausible claims against honest players
        if overall_p < 0.15 and counting_p < 0.95:
            return False

        # Two-way Bayesian fusion (w_nn = 0)
        n_obs = self.model.total_actions_observed
        if self.variance_scaled:
            var = self.model.overall_bluff.variance()
            certainty = max(0.0, 1.0 - (var / 0.05))
            w_model = min(self.w_model_cap, certainty * self.w_model_cap)
        else:
            w_model = min(self.w_model_cap, n_obs / 25.0)

        w_count = 1.0 - w_model

        fused_bluff_prob = w_model * model_p + w_count * counting_p

        call_thresh = max(0.35, min(0.85, 0.70 - (overall_p - 0.20) * self.call_mult))
        return fused_bluff_prob > call_thresh


def run_ablation_experiment(games_per_pairing: int = 30):
    print("================================================================================")
    print("      Empirical Component Ablation: Neural Network vs Pure Bayesian             ")
    print("================================================================================")

    # Architectures under test
    test_architectures = [
        ("Full_Hybrid (NN+Bayes+Count)", lambda: HybridBot(thompson_sampling=True)),
        ("NoNN_Hybrid (Bayes+Count)",    lambda: NoNNHybridBot(thompson_sampling=True)),
        ("Pure_Bayesian",                lambda: BayesianBot()),
        ("Pure_NN",                      lambda: PureNNBot()),
    ]

    # Benchmark opponents
    opponents = [
        ("HonestBot",        lambda: HonestBot()),
        ("CardCountBot",     lambda: CardCountBot()),
        ("BayesianBot",      lambda: BayesianBot()),
        ("RandomBot",        lambda: RandomBot()),
        ("Calling_Station",  lambda: PersonaBot("Calling_Station", 0.10, 0.85)),
        ("Hyper_Maniac",     lambda: PersonaBot("Hyper_Maniac", 0.65, 0.60)),
    ]

    summary_results = {}
    total_start = time.time()

    for arch_name, arch_factory in test_architectures:
        print(f"\nEvaluating Architecture: {arch_name}...")
        t0 = time.time()

        grand_w, grand_l, grand_d = 0, 0, 0
        opp_records = {}

        for opp_name, opp_factory in opponents:
            w, l, d = 0, 0, 0
            for g in range(games_per_pairing):
                seat = 0 if g % 2 == 0 else 1
                bot = arch_factory()
                opp = opp_factory()

                if seat == 0:
                    winner = play_bot_vs_bot(bot, opp, max_turns=100)
                    if winner == 0:
                        w += 1
                    elif winner == 1:
                        l += 1
                    else:
                        d += 1
                else:
                    winner = play_bot_vs_bot(opp, bot, max_turns=100)
                    if winner == 1:
                        w += 1
                    elif winner == 0:
                        l += 1
                    else:
                        d += 1

            grand_w += w
            grand_l += l
            grand_d += d
            opp_records[opp_name] = (w, l, d)
            print(f"   vs {opp_name:16s}: {w:2d}W - {l:2d}L - {d:2d}D")

        net = grand_w - grand_l
        total_games = grand_w + grand_l + grand_d
        win_pct = (grand_w / total_games) * 100.0 if total_games > 0 else 0.0
        loss_pct = (grand_l / total_games) * 100.0 if total_games > 0 else 0.0
        duration = time.time() - t0

        summary_results[arch_name] = {
            "wins": grand_w,
            "losses": grand_l,
            "draws": grand_d,
            "net": net,
            "win_rate": win_pct,
            "loss_rate": loss_pct,
            "opp_breakdown": opp_records,
            "duration": duration,
        }

        print(f"   => TOTAL: {grand_w}W - {grand_l}L - {grand_d}D | Net: {net:+d} | Win: {win_pct:.1f}% | Loss: {loss_pct:.1f}% ({duration:.1f}s)")

    print("\n" + "=" * 90)
    print("                     Ablation Comparative Summary Table                         ")
    print("=" * 90)
    print(f"{'Architecture':<30s} | {'Wins':<5s} | {'Losses':<6s} | {'Draws':<5s} | {'Net':<6s} | {'Win Rate':<8s} | {'Loss Rate':<9s}")
    print("-" * 90)
    for name, r in summary_results.items():
        print(f"{name:<30s} | {r['wins']:<5d} | {r['losses']:<6d} | {r['draws']:<5d} | {r['net']:<+6d} | {r['win_rate']:>7.1f}% | {r['loss_rate']:>8.1f}%")
    print("=" * 90)

    # Save to JSON
    out_path = "data/ablation_nn_vs_bayesian.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"Results exported to {out_path}")

    return summary_results


if __name__ == "__main__":
    run_ablation_experiment(games_per_pairing=30)
