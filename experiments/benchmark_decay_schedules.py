"""
Benchmark Study: Decaying Mixture of Experts (TD-MoE) Schedule Comparison.
Compares 4 Temporal Weighting Transition Functions:
1. Exponential Decay: w_nn(n) = w_floor + (1 - w_floor) * exp(-n / tau)
2. Sigmoidal Transition: w_nn(n) = w_floor + (1 - w_floor) / (1 + exp((n - 6) / 2))
3. Linear Piecewise: w_nn(n) = max(w_floor, 1.0 - n / 12)
4. Static 50/50 Control: w_nn(n) = 0.50

Evaluates 5 distinct opponent archetypes:
- HonestBot
- CardCountBot
- BayesianBot
- RandomBot
- HyperManiac (from synthetic population)

N=100 games per matchup (2,000 games total), 50/50 seat alternation.
Outputs: data/benchmark_decay_schedules.json
"""

import os
import sys
import math
import json
import time
import argparse
from typing import Dict, Tuple, List, Optional

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import GameState, Action
from bots.base import BotInterface, build_game_state
from bots.honest_bot import HonestBot
from bots.random_bot import RandomBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from test_bots import play_bot_vs_bot


class DecayingScheduleBeastBot(AcademicBeastBot):
    """AcademicBeastBot parameterized with custom decay transition schedule."""

    def __init__(self, schedule_type: str = "exponential", **kwargs):
        super().__init__(**kwargs)
        self.schedule_type = schedule_type

    def get_td_moe_weights(self, tau: Optional[float] = None, w_floor: Optional[float] = None) -> Tuple[float, float]:
        floor_val = w_floor if w_floor is not None else self.nn_floor
        n_obs = self.model.total_actions_observed

        if self.schedule_type == "exponential":
            tau_val = tau if tau is not None else self.decay_tau
            w_nn = floor_val + (1.0 - floor_val) * math.exp(-n_obs / max(0.5, tau_val))
        elif self.schedule_type == "sigmoidal":
            # S-curve transition centered at turn 5, slope scale 2.0
            sig = 1.0 / (1.0 + math.exp((n_obs - 5.0) / 2.0))
            w_nn = floor_val + (1.0 - floor_val) * sig
        elif self.schedule_type == "linear":
            # Linear decay to floor over 12 turns
            decay = max(0.0, 1.0 - n_obs / 12.0)
            w_nn = floor_val + (1.0 - floor_val) * decay
        elif self.schedule_type == "static":
            w_nn = 0.50
        else:
            w_nn = 0.50

        w_nn = max(floor_val, min(1.0, w_nn))
        w_bayes = 1.0 - w_nn
        return w_nn, w_bayes


def run_decay_benchmark(num_games: int = 100, seed: int = 20260911):
    print("=" * 75)
    print(f"  TD-MoE DECAY SCHEDULE BENCHMARK ({num_games} games/matchup, Seed={seed})")
    print("=" * 75)

    schedules = ["exponential", "sigmoidal", "linear", "static"]
    opponents = [
        ("Honest", HonestBot),
        ("CardCount", CardCountBot),
        ("Bayesian", BayesianBot),
        ("Random", RandomBot),
        ("HyperManiac", lambda: PersonaBot("Hyper_Maniac", true_bluff_rate=0.60, true_call_rate=0.35)),
    ]

    results = {}
    t_start = time.time()

    for sched in schedules:
        print(f"\n>>> Testing Schedule: {sched} ...")
        sched_res = {
            "schedule": sched,
            "total_wins": 0,
            "total_losses": 0,
            "total_draws": 0,
            "matchups": {},
        }

        for opp_name, opp_factory in opponents:
            w = 0
            l = 0
            d = 0

            for g in range(num_games):
                # 50/50 seat alternation
                agent = DecayingScheduleBeastBot(schedule_type=sched)
                opp = opp_factory()

                if g % 2 == 0:
                    res = play_bot_vs_bot(agent, opp)
                    if res == 0:
                        w += 1
                    elif res == 1:
                        l += 1
                    else:
                        d += 1
                else:
                    res = play_bot_vs_bot(opp, agent)
                    if res == 1:
                        w += 1
                    elif res == 0:
                        l += 1
                    else:
                        d += 1

            sched_res["total_wins"] += w
            sched_res["total_losses"] += l
            sched_res["total_draws"] += d
            sched_res["matchups"][opp_name] = {"w": w, "l": l, "d": d}
            print(f"   vs {opp_name:<12}: {w:>2}W - {l:>2}L - {d:>2}D (Win: {w/num_games*100:.1f}%, Loss: {l/num_games*100:.1f}%)")

        tot = len(opponents) * num_games
        sched_res["win_rate"] = sched_res["total_wins"] / tot
        sched_res["loss_rate"] = sched_res["total_losses"] / tot
        sched_res["draw_rate"] = sched_res["total_draws"] / tot

        print(f"  >> Overall {sched:<12}: {sched_res['total_wins']}W - {sched_res['total_losses']}L - {sched_res['total_draws']}D | "
              f"Win: {sched_res['win_rate']*100:.1f}% | Loss: {sched_res['loss_rate']*100:.2f}%")
        results[sched] = sched_res

    out_file = "data/benchmark_decay_schedules.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Benchmark complete in {time.time() - t_start:.1f}s. Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    run_decay_benchmark(num_games=args.games, seed=args.seed)
