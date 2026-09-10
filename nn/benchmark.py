"""Benchmark a trained PureNN checkpoint against the 4 rule-based baselines.

Paper-grade output (criterion ADR 2026-09-10: W/L/D + shed-rate reporting):
    - W/L/D per matchup (draw-heavy rows are parity, not weakness — the
      draw-lock analysis showed honest mirrors never resolve under the
      100-turn cap)
    - Wilson 95% score CIs on win rate (citable intervals, not point rows)
    - bluff rate per matchup (calibration vs Dewey 2025 / Yeung 2008)
    - mean opponent hand at game end (shed-race margin metric)

Usage:
    python -m nn.benchmark --checkpoint nn/checkpoints/v7_best.pt
    python -m nn.benchmark --checkpoint nn/checkpoints/v7_best.pt --games 200 \
        --markdown >> docs/benchmarks.md
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot

from nn.training import NNPlayer, RulePlayer, play_training_game, load_checkpoint
from nn.state_encoder import StateEncoder

BASELINES = [
    ("Random", RandomBot),
    ("Honest", HonestBot),
    ("CardCount", CardCountBot),
    ("Bayesian", BayesianBot),
]


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (citable CI)."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def benchmark(net, num_games: int, stochastic: bool = False):
    """Returns {name: stats-dict} for the NN vs each baseline.

    stats keys: wins/draws/losses, bluff_rate, mean_opp_hand_end,
    win_rate, ci_low, ci_high.

    stochastic=True samples from the policy instead of argmax — the v4
    forensics showed deterministic eval can collapse to always-call while
    the sampled policy still plays well (deployment-relevant; stochastic
    deployment also bluffs non-exploitably, per solo-push 3 votes).
    """
    enc = StateEncoder()
    results = {}
    for name, cls in BASELINES:
        wins = draws = losses = 0
        plays = bluffs = 0
        opp_hand_end_sum = 0
        decided = 0
        for i in range(num_games):
            # Alternate seats; NN always first arg (agent), seat varies
            agent_seat = i % 2
            game = None
            _, winner, stats = play_training_game(
                NNPlayer(net, enc), RulePlayer(cls(), name),
                agent_seat=agent_seat, deterministic=not stochastic)
            if winner is None or winner == -1:
                draws += 1
            elif winner == agent_seat:
                wins += 1
                decided += 1
            else:
                losses += 1
                decided += 1
            plays += stats["agent_plays"]
            bluffs += stats["agent_bluffs"]
            # Hand-at-end comes from the engine's final state via stats
            # extension (see play_training_game stats dict).
            opp_hand_end_sum += stats.get("opp_hand_end", 0)
        lo, hi = wilson_ci(wins, num_games)
        results[name] = {
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / num_games,
            "ci_low": lo, "ci_high": hi,
            "bluff_rate": bluffs / max(1, plays),
            "mean_opp_hand_end": opp_hand_end_sum / num_games,
            "decided": decided,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PureNN vs baselines")
    parser.add_argument("--checkpoint", default="nn/checkpoints/final.pt")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per matchup (default 100)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed Python RNG (deck shuffles + rule-bot "
                             "draws) for reproducible benchmarks.")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample from the policy instead of argmax "
                             "(v4 forensics: deterministic eval can collapse "
                             "while the sampled policy is fine)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    net = load_checkpoint(args.checkpoint)
    results = benchmark(net, args.games, stochastic=args.stochastic)

    out = sys.stdout
    out.write(f"\nPureNN ({args.checkpoint}) vs baselines — "
              f"{args.games} games each, deterministic policy, seats "
              f"alternated, Wilson 95% CI\n\n")
    if args.markdown:
        out.write("| Matchup | W/L/D | Win rate [95% CI] | Bluff rate | "
                  "Mean opp hand @ end |\n|---|---|---|---|---|\n")
        for name, s in results.items():
            out.write(
                f"| PureNN vs {name} | {s['wins']}-{s['losses']}-"
                f"{s['draws']} | {s['win_rate']:.0%} "
                f"[{s['ci_low']:.0%}, {s['ci_high']:.0%}] | "
                f"{s['bluff_rate']:.0%} | {s['mean_opp_hand_end']:.1f} |\n")
    else:
        out.write(f"  {'Matchup':<22} {'W/L/D':>12} {'WinRate[CI]':>22} "
                  f"{'Bluff':>6} {'OppHand':>8}\n")
        out.write("  " + "-" * 74 + "\n")
        for name, s in results.items():
            wld = f"{s['wins']}-{s['losses']}-{s['draws']}"
            ci = (f"{s['win_rate']:.0%} "
                  f"[{s['ci_low']:.0%},{s['ci_high']:.0%}]")
            out.write(f"  {'PureNN vs ' + name:<22} {wld:>12} {ci:>22} "
                      f"{s['bluff_rate']:>5.0%} {s['mean_opp_hand_end']:>8.1f}\n")
    out.write("\n")


if __name__ == "__main__":
    main()
