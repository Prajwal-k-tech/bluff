"""Benchmark a trained PureNN checkpoint against the 4 rule-based baselines.

Produces the Phase 3 benchmark table (docs/bot-modes.md §4) and the numbers
for research claims #1-#2:
    - win rate vs each baseline (100 games each by default, seats alternated)
    - bluff rate per matchup (calibration vs Dewey 2025 / Yeung 2008)
    - baseline-vs-baseline reference rows via test_bots.py --log + report.py

Usage:
    python -m nn.benchmark --checkpoint nn/checkpoints/v4.pt
    python -m nn.benchmark --checkpoint nn/checkpoints/final.pt --games 200 \
        --markdown >> docs/benchmarks.md
"""

import argparse
import os
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


def benchmark(net, num_games: int):
    """Returns {name: (win_rate, bluff_rate)} for the NN vs each baseline."""
    enc = StateEncoder()
    results = {}
    for name, cls in BASELINES:
        wins = plays = bluffs = 0
        for i in range(num_games):
            # Alternate seats; NN always first arg (agent), seat varies
            _, winner, stats = play_training_game(
                NNPlayer(net, enc), RulePlayer(cls(), name),
                agent_seat=i % 2, deterministic=True)
            if winner == i % 2:
                wins += 1
            plays += stats["agent_plays"]
            bluffs += stats["agent_bluffs"]
        results[name] = (wins / num_games, bluffs / max(1, plays))
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark PureNN vs baselines")
    parser.add_argument("--checkpoint", default="nn/checkpoints/final.pt")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per matchup (default 100)")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    net = load_checkpoint(args.checkpoint)
    results = benchmark(net, args.games)

    out = sys.stdout
    out.write(f"\nPureNN ({args.checkpoint}) vs baselines — "
              f"{args.games} games each, deterministic policy\n\n")
    if args.markdown:
        out.write("| Matchup | NN win rate | NN bluff rate |\n|---|---|---|\n")
        for name, (wr, br) in results.items():
            out.write(f"| PureNN vs {name} | {wr:.0%} | {br:.0%} |\n")
    else:
        out.write(f"  {'Matchup':<22} {'WinRate':>8} {'BluffRate':>10}\n")
        out.write("  " + "-" * 42 + "\n")
        for name, (wr, br) in results.items():
            out.write(f"  {'PureNN vs ' + name:<22} {wr:>7.0%} {br:>9.0%}\n")
    out.write("\n")


if __name__ == "__main__":
    main()
