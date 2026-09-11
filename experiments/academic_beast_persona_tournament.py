"""
Academic Beast Persona Tournament (Phase 2).
Evaluates AcademicBeastBot (Sigmoidal TD-MoE + BluffNet-XL + Conditioned Thompson + Dewey EV)
across the 20 parameterized synthetic personas (4,000 games total, N=200 per persona, 50/50 seats).
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


def run_tournament(games_per_persona: int = 200, master_seed: int = 20260911) -> dict:
    print("=" * 80)
    print(f"ACADEMIC BEAST PERSONA TOURNAMENT ({len(SYNTHETIC_POPULATION)} Personas x {games_per_persona} Games = {len(SYNTHETIC_POPULATION)*games_per_persona} Games)")
    print("Architecture: BluffNet-XL (1.35M weights) + Sigmoidal S-Curve TD-MoE + Dewey EV + Conditioned Thompson")
    print("=" * 80)

    results = []
    total_w = 0
    total_l = 0
    total_d = 0
    t0 = time.time()

    for idx, persona in enumerate(SYNTHETIC_POPULATION, 1):
        w = 0
        l = 0
        d = 0
        call_successes = 0
        call_attempts = 0
        lock_hand_sizes = []
        bluff_errors = []
        call_errors = []

        np.random.seed(master_seed + idx * 7919)

        for g in range(games_per_persona):
            game_seed = master_seed + idx * 7919 + g
            random.seed(game_seed)
            np.random.seed(game_seed)
            beast_seat = g % 2  # Strict 50/50 seat alternation

            beast = AcademicBeastBot(
                checkpoint_path="nn/checkpoints/bluffnet_xl_league.pt",
                schedule_type="sigmoidal",
                thompson_sampling=True,
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

            # Match outcome
            if winner is None or winner == -1:
                d += 1
            elif (winner == 0 and beast_seat == 0) or (winner == 1 and beast_seat == 1):
                w += 1
            else:
                l += 1

            # Estimation accuracy
            est_bluff = beast.model.overall_bluff.mean()
            est_call = beast.model.estimate_call_frequency()
            bluff_errors.append(abs(est_bluff - persona.true_bluff_rate))
            call_errors.append(abs(est_call - persona.true_call_rate))

        total_w += w
        total_l += l
        total_d += d

        wr = (w / games_per_persona) * 100
        lr = (l / games_per_persona) * 100
        ci_lo, ci_hi = wilson_ci(w, games_per_persona)
        a_call = (call_successes / call_attempts * 100) if call_attempts > 0 else 0.0
        s_lock = float(np.mean(lock_hand_sizes)) if lock_hand_sizes else 0.0
        b_mae = float(np.mean(bluff_errors)) if bluff_errors else 0.0
        c_mae = float(np.mean(call_errors)) if call_errors else 0.0

        persona_res = {
            "persona": persona.name,
            "true_bluff": persona.true_bluff_rate,
            "true_call": persona.true_call_rate,
            "w": w,
            "l": l,
            "d": d,
            "win_rate": wr,
            "loss_rate": lr,
            "w_ci": [ci_lo, ci_hi],
            "a_call": a_call,
            "s_lock": s_lock,
            "bluff_mae": b_mae,
            "call_mae": c_mae,
        }
        results.append(persona_res)
        print(f"[{idx:02d}/20] {persona.name:<24} | {w:>3}W - {l:>2}L - {d:>3}D | WR: {wr:>5.1f}% [{ci_lo:>4.1f}%, {ci_hi:>4.1f}%] | LR: {lr:>4.1f}% | A_call: {a_call:>5.1f}% | Bluff MAE: {b_mae:.3f}")

    dt = time.time() - t0
    grand_total = total_w + total_l + total_d
    global_wr = (total_w / grand_total) * 100
    global_lr = (total_l / grand_total) * 100

    print("=" * 80)
    print(f"OVERALL TOURNAMENT STANDINGS ({grand_total} Games in {dt:.1f}s):")
    print(f"  Record: {total_w}W - {total_l}L - {total_d}D (+{total_w - total_l} Net Score)")
    print(f"  Win Rate: {global_wr:.2f}% | Loss Rate: {global_lr:.2f}% | Non-Loss Rate: {100.0 - global_lr:.2f}%")
    print("=" * 80)

    summary = {
        "metadata": {
            "total_games": grand_total,
            "games_per_persona": games_per_persona,
            "duration_seconds": round(dt, 2),
            "master_seed": master_seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "aggregate": {
            "w": total_w,
            "l": total_l,
            "d": total_d,
            "win_rate": global_wr,
            "loss_rate": global_lr,
            "non_loss_rate": 100.0 - global_lr,
        },
        "personas": results,
    }

    os.makedirs("data", exist_ok=True)
    out_path = "data/academic_beast_persona_tournament.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved tournament results to {out_path}")

    return summary


if __name__ == "__main__":
    run_tournament(games_per_persona=200)
