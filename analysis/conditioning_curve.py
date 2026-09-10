"""Conditioning-curve analysis — evidence for research claim #2.

Question: does the trained policy's *behavior* condition on the opponent's
observed public signals (call rate, revealed-bluff rate), or is it
opponent-blind?

Method: play the NN against each baseline archetype and record, per decision
phase:
  - the conditioning features the encoder saw (opponent call rate so far,
    opponent revealed-bluff rate so far),
  - the action taken (play-side bluff yes/no, respond-side call yes/no).

Output: per-opponent aggregate table (the "conditioning curve") showing the
policy's call rate / bluff rate as a function of who it faces — the NN-side
half of the claim-#2 figure. The rule-bot side (their true bluff rates) comes
from analysis/report.py on the same games.

Usage:
    python -m analysis.conditioning_curve --checkpoint nn/checkpoints/v6_best.pt
    python -m analysis.conditioning_curve --checkpoint ... --games 50 --markdown
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


def measure(net, num_games: int):
    """Returns {name: {...}} per-opponent behavioral aggregates."""
    enc = StateEncoder()
    results = {}
    for name, cls in BASELINES:
        agg = {
            "wins": 0, "plays": 0, "bluffs": 0,
            "responds": 0, "calls": 0,
            "sig_call_rate_sum": 0.0, "sig_revealed_bluff_sum": 0.0,
            "n_signal_observations": 0,
        }
        for i in range(num_games):
            agent = NNPlayer(net, enc, trace_signals=True)
            agent_seat = i % 2
            _, winner, stats = play_training_game(
                agent, RulePlayer(cls(), name), agent_seat=agent_seat,
                deterministic=True)
            if winner == agent_seat:
                agg["wins"] += 1
            agg["plays"] += stats["agent_plays"]
            agg["bluffs"] += stats["agent_bluffs"]
            agg["responds"] += stats.get("agent_responds", 0)
            agg["calls"] += stats.get("agent_calls", 0)
            # Signals the policy actually observed (averaged over respond
            # decisions where signals were defined) — recorded by NNPlayer
            # when trace_signals is enabled.
            for sig in getattr(agent, "signal_trace", []):
                agg["sig_call_rate_sum"] += sig["opp_call_rate"]
                agg["sig_revealed_bluff_sum"] += sig["opp_revealed_bluff"]
                agg["n_signal_observations"] += 1
        n = max(1, num_games)
        nsig = max(1, agg["n_signal_observations"])
        results[name] = {
            "win_rate": agg["wins"] / n,
            "bluff_rate": agg["bluffs"] / max(1, agg["plays"]),
            "call_rate": agg["calls"] / max(1, agg["responds"]),
            "observed_opp_call_rate": agg["sig_call_rate_sum"] / nsig,
            "observed_opp_bluff_rate": agg["sig_revealed_bluff_sum"] / nsig,
        }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Claim #2 conditioning-curve analysis")
    parser.add_argument("--checkpoint", default="nn/checkpoints/final.pt")
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    net = load_checkpoint(args.checkpoint)
    results = measure(net, args.games)

    out = sys.stdout
    out.write(f"\nConditioning curve — {args.checkpoint} "
              f"({args.games} games/matchup, deterministic)\n")
    out.write("Policy behavior vs opponent archetype (claim #2 evidence)\n\n")
    if args.markdown:
        out.write("| Opponent | NN win | NN bluff rate | NN call rate | "
                  "obs opp call-rate sig | obs opp bluff sig |\n")
        out.write("|---|---|---|---|---|---|\n")
        for name, r in results.items():
            out.write(f"| {name} | {r['win_rate']:.0%} | "
                      f"{r['bluff_rate']:.0%} | {r['call_rate']:.0%} | "
                      f"{r['observed_opp_call_rate']:.2f} | "
                      f"{r['observed_opp_bluff_rate']:.2f} |\n")
    else:
        hdr = (f"  {'Opponent':<12} {'Win':>5} {'Bluff':>6} {'Call':>6} "
               f"{'oppCallSig':>11} {'oppBluffSig':>12}")
        out.write(hdr + "\n  " + "-" * (len(hdr) - 2) + "\n")
        for name, r in results.items():
            out.write(f"  {name:<12} {r['win_rate']:>5.0%} "
                      f"{r['bluff_rate']:>6.0%} {r['call_rate']:>6.0%} "
                      f"{r['observed_opp_call_rate']:>11.2f} "
                      f"{r['observed_opp_bluff_rate']:>12.2f}\n")
    out.write("\nReading: policy is signal-conditioned iff its call/bluff "
              "columns VARY across opponents in the direction of the observed "
              "signals (call more vs frequent bluffers, bluff less vs "
              "callers). v4 (constant priors) shows ~no variation — the "
              "ablation control.\n")


if __name__ == "__main__":
    main()
