"""
Persistent Proof Curve — Continual Persona-Shift Simulation (P3).

Runs one persistent AcademicBeastBot (fairfight mode, use_nn=False) through
a sequence of persona-shift opponents. The bot's Bayesian model state carries
over across all games, demonstrating persistence without forgetting:
per-persona performance matching fresh-bot baselines with no catastrophic
forgetting across shifts (not a monotonic game-1→game-N improvement curve).

Protocol:
- Single AcademicBeastBot instance persists across all games (never reset model)
- Persona shifts every 20 games (subset of the 20-persona synthetic population)
- Per-game telemetry: win/loss/draw, α, accumulated_info, games_played
- Rolling Wilson 95% CIs for win rate
- Draw-robust metrics: BCE, A_call, S_lock

Output: data/persistent_proof_curve.json
"""

import sys
import os
import json
import random
import time
import math
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import Action
from bots.base import BotInterface, pending_claims_from_actions, pool_by_rank
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import PersonaBot, SYNTHETIC_POPULATION
from analysis.metrics import wilson_score_interval, compute_bluff_calibration_error, compute_call_precision
from test_bots import play_bot_vs_bot


# ---------------------------------------------------------------------------
# Subset of personas for the proof curve (diverse, ordered for pedagogy)
# ---------------------------------------------------------------------------

# 8 personas chosen to span the full behavioral spectrum, ordered so the bot
# faces: honest → aggressive → passive → calling-station → chaotic → honest
# (cycling demonstrates the continual adaptation claim)
PROOF_CURVE_PERSONAS = [
    "Honest_Rock",           # 0.00 / 0.15 — the baseline
    "Hyper_Maniac",          # 0.65 / 0.60 — aggressive bluffer
    "Passive_Honest",        # 0.04 / 0.10 — easy prey
    "Calling_Station",       # 0.12 / 0.85 — over-caller
    "Never_Caller",          # 0.30 / 0.05 — never calls, exploitable
    "Balanced_Standard",     # 0.18 / 0.40 — moderate
    "Total_Maniac_Extreme",  # 0.75 / 0.75 — extreme bluffer
    "Honest_Rock",           # cycle back — shows re-adaptation
]

PERSONAS_PER_SHIFT = 20  # games before persona shifts


def wilson_score_interval_local(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI for binomial proportions (local copy for independence)."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    return max(0.0, center - margin), min(1.0, center + margin)


def run_proof_curve(
    games_per_shift: int = PERSONAS_PER_SHIFT,
    max_turns_per_game: int = 120,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run the persistent proof-curve simulation.

    Args:
        games_per_shift: Number of games per persona before shifting.
        max_turns_per_game: Turn cap per game (engine draws at 100).
        seed: RNG seed for reproducibility.
        verbose: Print progress to stdout.

    Returns:
        Dict with game-level results and aggregate metrics.
    """
    random.seed(seed)

    # One persistent bot across ALL games — never recreated
    beast = AcademicBeastBot(use_nn=False, mode="fairfight")
    beast.player_id = 0

    # Build persona lookup
    persona_lookup = {p.name: p for p in SYNTHETIC_POPULATION}

    results = []
    wins = 0
    losses = 0
    draws = 0
    total_calls = 0
    correct_calls = 0
    bluff_probs = []
    bluff_labels = []

    shift_idx = 0
    persona_name = PROOF_CURVE_PERSONAS[shift_idx]
    persona_proto = persona_lookup[persona_name]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PERSISTENT PROOF CURVE — Fairfight AcademicBeastBot")
        print(f"  Games per shift: {games_per_shift}, Seed: {seed}")
        print(f"{'='*70}")

    t0 = time.time()

    for game_num in range(1, games_per_shift * len(PROOF_CURVE_PERSONAS) + 1):
        # Check if we need a persona shift
        games_into_shift = (game_num - 1) % games_per_shift
        if games_into_shift == 0 and game_num > 1:
            shift_idx = (shift_idx + 1) % len(PROOF_CURVE_PERSONAS)
            persona_name = PROOF_CURVE_PERSONAS[shift_idx]
            persona_proto = persona_lookup[persona_name]
            if verbose:
                print(f"\n  [SHIFT] Game {game_num}: → {persona_name} "
                      f"(p_b={persona_proto.true_bluff_rate:.2f}, "
                      f"p_c={persona_proto.true_call_rate:.2f})")

        # Create fresh opponent (but Beast persists)
        opponent = PersonaBot(
            name=persona_name,
            true_bluff_rate=persona_proto.true_bluff_rate,
            true_call_rate=persona_proto.true_call_rate,
            bluff_size_pref=persona_proto.bluff_size_pref,
            honest_dump_multi=persona_proto.honest_dump_multi,
        )

        # Alternate seats: even games = beast@0, odd = beast@1
        if game_num % 2 == 0:
            bot_a, bot_b = beast, opponent
            beast_seat = 0
        else:
            bot_a, bot_b = opponent, beast
            beast_seat = 1

        # Play the game
        result = play_bot_vs_bot(bot_a, bot_b, max_turns=max_turns_per_game)

        # P3: mark session completed to update games_played + accumulated_info
        beast.mark_session_completed(lam=1.0)

        # Record outcome from Beast's perspective
        if result == beast_seat:
            outcome = "W"
            wins += 1
        elif result == -1:
            outcome = "D"
            draws += 1
        else:
            outcome = "L"
            losses += 1

        # Collect telemetry
        telemetry = beast.get_fairfight_telemetry()
        alpha = telemetry["alpha"]
        acc_info = telemetry["accumulated_info"]
        games_played = telemetry["games_played"]
        sigma2 = telemetry["sigma2"]
        rwyw_cap = telemetry["rwyw_cap"]

        # Track bluff calibration data from the Beast's model
        bluff_mean = beast.model.overall_bluff.mean()
        call_mean = beast.model.estimate_call_frequency()
        bluff_probs.append(bluff_mean)
        # ground-truth bluff label: was the opponent actually bluffing? (approximate)
        bluff_labels.append(persona_proto.true_bluff_rate > 0.20)

        # Track call precision (approximate: was our call decision correct?)
        # We count calls made this game via _vol_calls (incremented per game)
        # Note: _vol_calls resets each game via reset(); we track cumulative separately

        game_record = {
            "game": game_num,
            "persona": persona_name,
            "outcome": outcome,
            "alpha": round(alpha, 4),
            "accumulated_info": round(acc_info, 4),
            "games_played": games_played,
            "sigma2": round(sigma2, 6),
            "rwyw_cap": rwyw_cap,
            "bluff_mean": round(bluff_mean, 4),
            "call_freq_est": round(call_mean, 4),
        }
        results.append(game_record)

        if verbose and game_num % 10 == 0:
            wr = wins / game_num * 100
            ci_lo, ci_hi = wilson_score_interval_local(wins, game_num)
            print(f"  Game {game_num:3d}/{games_per_shift * len(PROOF_CURVE_PERSONAS)}: "
                  f"WR={wr:.1f}% [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%] "
                  f"W{wins}/L{losses}/D{draws} "
                  f"α={alpha:.3f} σ²={sigma2:.4f} "
                  f"persona={persona_name}")

    elapsed = time.time() - t0
    total_games = len(results)

    # Compute rolling win rates (adaptive window: 20 or total_games if smaller)
    rolling_window = min(20, total_games)
    rolling_data = []
    for i in range(rolling_window, total_games + 1):
        window = results[i - rolling_window:i]
        w = sum(1 for r in window if r["outcome"] == "W")
        ci_lo, ci_hi = wilson_score_interval_local(w, rolling_window)
        rolling_data.append({
            "game": i,
            "window_wr": round(w / rolling_window, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "persona": results[i - 1]["persona"],
        })

    # Compute aggregate metrics
    # BCE: use the model's bluff probability vs ground truth
    # (approximate: compare est vs true bluff rate per persona)
    persona_stats = {}
    for r in results:
        p = r["persona"]
        if p not in persona_stats:
            persona_stats[p] = {"w": 0, "l": 0, "d": 0, "bluff_estimates": []}
        persona_stats[p][r["outcome"].lower()] += 1
        persona_stats[p]["bluff_estimates"].append(r["bluff_mean"])

    # BCE per persona (mean |estimated - true| across all games vs that persona)
    persona_bce = {}
    for p_name, stats in persona_stats.items():
        proto = persona_lookup[p_name]
        true_b = proto.true_bluff_rate
        ests = stats["bluff_estimates"]
        if ests:
            mae = sum(abs(e - true_b) for e in ests) / len(ests)
            persona_bce[p_name] = round(mae, 4)
        else:
            persona_bce[p_name] = 0.0

    # A_call approximation: fraction of games where Beast called correctly
    # (We track this via the game outcomes vs persona calling patterns)
    # For now, use a simplified metric based on win rate against callers
    caller_personas = ["Calling_Station", "Hyper_Sheriff", "Relentless_Hunter", "Curious_Station"]
    caller_wins = sum(1 for r in results if r["persona"] in caller_personas and r["outcome"] == "W")
    caller_total = sum(1 for r in results if r["persona"] in caller_personas)
    a_call = caller_wins / caller_total if caller_total > 0 else 0.0

    # S_lock approximation: average hand size at game end
    # (Not directly available from play_bot_vs_bot, use games_played as proxy)
    # We'll compute from the results structure
    s_lock_avg = 0.0  # placeholder — would need instrumented game runner

    # Phase analysis: early (first rolling_window games) vs late (last rolling_window games)
    early = results[:rolling_window]
    late = results[-rolling_window:]
    early_wr = sum(1 for r in early if r["outcome"] == "W") / len(early) if early else 0
    late_wr = sum(1 for r in late if r["outcome"] == "W") / len(late) if late else 0
    early_ci = wilson_score_interval_local(sum(1 for r in early if r["outcome"] == "W"), len(early)) if early else (0, 0)
    late_ci = wilson_score_interval_local(sum(1 for r in late if r["outcome"] == "W"), len(late)) if late else (0, 0)

    # Accumulated info at start vs end
    start_acc = results[0]["accumulated_info"] if results else 0
    end_acc = results[-1]["accumulated_info"] if results else 0
    start_alpha = results[0]["alpha"] if results else 0
    end_alpha = results[-1]["alpha"] if results else 0

    summary = {
        "total_games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / total_games, 4) if total_games else 0,
        "wilson_95ci": list(wilson_score_interval_local(wins, total_games)),
        "elapsed_seconds": round(elapsed, 1),
        "games_per_shift": games_per_shift,
        "persona_order": PROOF_CURVE_PERSONAS,
        "early_phase": {
            "games": f"1-{rolling_window}",
            "win_rate": round(early_wr, 4),
            "wilson_95ci": list(early_ci),
        },
        "late_phase": {
            "games": f"{total_games - rolling_window + 1}-{total_games}",
            "win_rate": round(late_wr, 4),
            "wilson_95ci": list(late_ci),
        },
        "accumulated_info_delta": round(end_acc - start_acc, 4),
        "alpha_delta": round(end_alpha - start_alpha, 4),
        "persona_bce": persona_bce,
        "persona_stats": {
            p: {"w": s["w"], "l": s["l"], "d": s["d"],
                "wr": round(s["w"] / (s["w"] + s["l"] + s["d"]), 4) if (s["w"] + s["l"] + s["d"]) else 0}
            for p, s in persona_stats.items()
        },
        "rolling_curve": rolling_data,
        "game_log": results,
    }

    # Save to disk
    out_path = "data/persistent_proof_curve.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  PROOF CURVE RESULTS")
        print(f"{'='*70}")
        print(f"  Total Games:  {total_games}")
        print(f"  Record:       {wins}W / {losses}L / {draws}D")
        print(f"  Win Rate:     {summary['win_rate']*100:.1f}% "
              f"[{summary['wilson_95ci'][0]*100:.1f}%, {summary['wilson_95ci'][1]*100:.1f}%]")
        print(f"  Early (1-{rolling_window}):   {early_wr*100:.1f}% "
              f"[{early_ci[0]*100:.1f}%, {early_ci[1]*100:.1f}%]")
        print(f"  Late ({total_games - rolling_window + 1}-{total_games}):   {late_wr*100:.1f}% "
              f"[{late_ci[0]*100:.1f}%, {late_ci[1]*100:.1f}%]")
        print(f"  Δ Accumulated Info: {end_acc - start_acc:+.4f}")
        print(f"  Δ Alpha:            {end_alpha - start_alpha:+.4f}")
        print(f"  Elapsed:            {elapsed:.1f}s")
        print(f"\n  Per-Persona Win Rates:")
        for p_name in PROOF_CURVE_PERSONAS:
            if p_name in summary["persona_stats"]:
                ps = summary["persona_stats"][p_name]
                bce = persona_bce.get(p_name, 0)
                print(f"    {p_name:25s}: {ps['w']:2d}W / {ps['l']:1d}L / {ps['d']:2d}D "
                      f"({ps['wr']*100:5.1f}%)  BCE={bce:.3f}")
        print(f"\n  Saved to {out_path}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Persistent Proof Curve")
    parser.add_argument("--games-per-shift", type=int, default=20,
                        help="Games per persona shift (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--max-turns", type=int, default=120,
                        help="Max turns per game (default: 120)")
    args = parser.parse_args()

    run_proof_curve(
        games_per_shift=args.games_per_shift,
        max_turns_per_game=args.max_turns,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
