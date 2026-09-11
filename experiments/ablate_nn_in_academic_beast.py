"""
Factorial Ablation Study: BluffNet-XL vs No-NN in Academic Beast Bot.
Directly evaluates the quantitative value-add of deep neural policy priors
(BluffNet-XL, 1.35M parameters) when integrated with Bayesian opponent modeling,
Dewey (2025) stake-sensitive EV calling, and Archetype-Conditioned Thompson Sampling.

Evaluates 2 conditions across all 20 parameterized synthetic personas:
  - Condition A: Full Academic Beast (BluffNet-XL + Sigmoidal TD-MoE)
  - Condition B: No-NN Academic Beast (use_nn=False: Bayesian + Combinatorial + Dewey EV)

Parameters: N=100 games/matchup x 20 personas x 2 conditions = 4,000 games total.
Strict 50/50 seat alternation, paired seeds per matchup.
"""

import sys
import os
import json
import time
import math
import random
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import GameState, Action
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from test_bots import play_bot_vs_bot


def wilson_ci(wins: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, (center - margin) * 100), min(100.0, (center + margin) * 100)


def evaluate_condition(
    use_nn: bool,
    games_per_persona: int = 100,
    master_seed: int = 20260911,
    condition_name: str = "BluffNet-XL"
) -> Tuple[List[dict], dict]:
    results = []
    total_w = 0
    total_l = 0
    total_d = 0
    t0 = time.time()

    print(f"\nEvaluating Condition: {condition_name} ({len(SYNTHETIC_POPULATION)} personas x {games_per_persona} games)")

    for idx, persona in enumerate(SYNTHETIC_POPULATION, 1):
        w = 0
        l = 0
        d = 0
        bluff_errors = []
        call_errors = []

        for g in range(games_per_persona):
            game_seed = master_seed + idx * 7919 + g
            random.seed(game_seed)
            np.random.seed(game_seed)
            beast_seat = g % 2  # Strict 50/50 seat alternation

            beast = AcademicBeastBot(
                checkpoint_path="nn/checkpoints/bluffnet_xl_league.pt" if use_nn else None,
                schedule_type="sigmoidal",
                thompson_sampling=True,
                use_nn=use_nn,
            )
            beast.player_id = beast_seat

            opp = PersonaBot(
                name=persona.name,
                true_bluff_rate=persona.true_bluff_rate,
                true_call_rate=persona.true_call_rate,
                bluff_size_pref=persona.bluff_size_pref,
                honest_dump_multi=persona.honest_dump_multi,
            )
            opp.player_id = 1 - beast_seat

            bot_a, bot_b = (beast, opp) if beast_seat == 0 else (opp, beast)
            winner = play_bot_vs_bot(bot_a, bot_b)

            if winner is None or winner == -1:
                d += 1
            elif (winner == 0 and beast_seat == 0) or (winner == 1 and beast_seat == 1):
                w += 1
            else:
                l += 1

            est_bluff = beast.model.overall_bluff.mean()
            est_call = beast.model.estimate_call_frequency()
            bluff_errors.append(abs(est_bluff - persona.true_bluff_rate))
            call_errors.append(abs(est_call - persona.true_call_rate))

        total_w += w
        total_l += l
        total_d += d

        wr = (w / games_per_persona) * 100
        lr = (l / games_per_persona) * 100
        dr = (d / games_per_persona) * 100
        w_lo, w_hi = wilson_ci(w, games_per_persona)
        mae_b = float(np.mean(bluff_errors))
        mae_c = float(np.mean(call_errors))

        results.append({
            "persona": persona.name,
            "condition": condition_name,
            "use_nn": use_nn,
            "true_bluff_rate": persona.true_bluff_rate,
            "true_call_rate": persona.true_call_rate,
            "games": games_per_persona,
            "wins": w,
            "losses": l,
            "draws": d,
            "win_rate": wr,
            "loss_rate": lr,
            "draw_rate": dr,
            "net_score": w - l,
            "wilson_ci_95": [w_lo, w_hi],
            "mae_bluff": mae_b,
            "mae_call": mae_c,
        })
        print(f"  [{idx:2d}/20] {persona.name:<24} | W: {w:3d} L: {l:2d} D: {d:3d} | WR: {wr:5.1f}% [{w_lo:4.1f}%, {w_hi:4.1f}%] | Net: {w-l:+4d}")

    elapsed = time.time() - t0
    n_total = len(SYNTHETIC_POPULATION) * games_per_persona
    tot_wr = (total_w / n_total) * 100
    tot_lr = (total_l / n_total) * 100
    tot_dr = (total_d / n_total) * 100
    tot_lo, tot_hi = wilson_ci(total_w, n_total)

    aggregate = {
        "condition": condition_name,
        "use_nn": use_nn,
        "total_games": n_total,
        "total_wins": total_w,
        "total_losses": total_l,
        "total_draws": total_d,
        "win_rate": tot_wr,
        "loss_rate": tot_lr,
        "draw_rate": tot_dr,
        "net_score": total_w - total_l,
        "wilson_ci_95": [tot_lo, tot_hi],
        "elapsed_seconds": elapsed,
    }
    print(f"\nCondition {condition_name} Total: W: {total_w} L: {total_l} D: {total_d} | WR: {tot_wr:.2f}% | LR: {tot_lr:.2f}% | Elapsed: {elapsed:.1f}s")
    return results, aggregate


def run_ablation_study(games_per_persona: int = 100, master_seed: int = 20260911):
    print("=" * 85)
    print("BLUFFNET-XL VS NO-NN FACTORIAL ABLATION STUDY ACROSS 20 SYNTHETIC PERSONAS")
    print(f"Total Games: {len(SYNTHETIC_POPULATION)} x {games_per_persona} x 2 = {len(SYNTHETIC_POPULATION) * games_per_persona * 2}")
    print("Master Seed:", master_seed, "(Paired matches per seed)")
    print("=" * 85)

    res_xl, agg_xl = evaluate_condition(
        use_nn=True,
        games_per_persona=games_per_persona,
        master_seed=master_seed,
        condition_name="BluffNet-XL"
    )

    res_nonn, agg_nonn = evaluate_condition(
        use_nn=False,
        games_per_persona=games_per_persona,
        master_seed=master_seed,
        condition_name="No-NN (Pure Bayesian/Combinatorial)"
    )

    # Paired comparisons
    comparisons = []
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD PAIRED ABLATION COMPARISON (BluffNet-XL vs No-NN)")
    print(f"{'Persona':<25} | {'BluffNet-XL WR':>14} | {'No-NN WR':>14} | {'Delta WR':>10} | {'Losses (XL/No)':>14}")
    print("-" * 95)

    delta_wr_list = []
    for r_xl, r_nonn in zip(res_xl, res_nonn):
        d_wr = r_xl["win_rate"] - r_nonn["win_rate"]
        delta_wr_list.append(d_wr)
        sign = "+" if d_wr > 0 else ""
        print(f"{r_xl['persona']:<25} | {r_xl['win_rate']:>13.1f}% | {r_nonn['win_rate']:>13.1f}% | {sign}{d_wr:>9.1f}% | {r_xl['losses']:>5d} / {r_nonn['losses']:<5d}")
        comparisons.append({
            "persona": r_xl["persona"],
            "xl_win_rate": r_xl["win_rate"],
            "nonn_win_rate": r_nonn["win_rate"],
            "delta_win_rate": d_wr,
            "xl_losses": r_xl["losses"],
            "nonn_losses": r_nonn["losses"],
            "delta_losses": r_xl["losses"] - r_nonn["losses"],
        })

    print("-" * 95)
    d_total_wr = agg_xl["win_rate"] - agg_nonn["win_rate"]
    d_total_net = agg_xl["net_score"] - agg_nonn["net_score"]
    sign = "+" if d_total_wr > 0 else ""
    print(f"{'AGGREGATE TOTAL':<25} | {agg_xl['win_rate']:>13.2f}% | {agg_nonn['win_rate']:>13.2f}% | {sign}{d_total_wr:>9.2f}% | {agg_xl['total_losses']:>5d} / {agg_nonn['total_losses']:<5d}")
    print(f"Net Score Advantage: {agg_xl['net_score']} vs {agg_nonn['net_score']} (Delta: {d_total_net:+d})")

    output_payload = {
        "metadata": {
            "study": "BluffNet-XL vs No-NN Factorial Ablation Across 20 Personas",
            "games_per_persona": games_per_persona,
            "total_games": len(SYNTHETIC_POPULATION) * games_per_persona * 2,
            "master_seed": master_seed,
            "date": "2026-09-11",
        },
        "aggregate_xl": agg_xl,
        "aggregate_nonn": agg_nonn,
        "comparisons": comparisons,
        "per_persona_xl": res_xl,
        "per_persona_nonn": res_nonn,
    }

    out_path = "data/academic_beast_nn_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[Ablation Complete] Full paired results written to {out_path}")
    return output_payload


if __name__ == "__main__":
    run_ablation_study(games_per_persona=100, master_seed=20260911)
