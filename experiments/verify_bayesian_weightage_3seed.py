"""
3-Seed Sensitivity Verification Study of Bayesian Weightage & BET Lookahead (Parallelized).
Directly addresses:
1. User mandate on Bayesian weightage and opponent adaptation:
   "did we experiment with the bayesian part having more weightage, logically speaking
    despite however you were trained on an NN, shouldnt how your opponents truly play
    and bluff me the major contributor? are you sure there are no more tweaking around we can do?
    ideally I want the Hybrid NN bot to be an absolute bluff beast able to take on any opponents"
2. Tess's statistical rigor requirement in AGENT_CHAT.md:
   "Wilson 95% CIs on your 4 regimes (N=2000 each)... overlap substantially —
    the monotone ordering is suggestive and directionally supports the user's hypothesis,
    but +1.25% at +/-2.2% margin is not a citable finding...
    Required before main.tex cites the ordering: one confirmation at N>=400/persona or 3 seeds"

Experimental Matrix:
  - 4 Regimes:
      1. Balanced_TDMoE (Sigmoidal S-curve, w_floor=0.15, tau=8.0)
      2. Heavy_Bayesian (Fast exponential, w_floor=0.05, tau=4.0)
      3. BET_Lookahead (Closed-Form Bluff Expected Value Theorem Lookahead)
      4. Pure_Bayesian (w_nn=0.0, w_bayes=1.0)
  - 3 Independent Seeds: 20260911, 20260912, 20260913
  - 20 Synthetic Personas
  - Strict 50/50 Seat Alternation
  - Multiprocessing worker pool utilizing available CPU cores
  - Wilson 95% Confidence Intervals computed on pooled metrics
"""

import os
import sys
import time
import json
import math
import random
import argparse
import concurrent.futures
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from game import GameState
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from analysis.metrics import (
    compute_call_precision,
    wilson_score_interval,
)


def run_single_match(
    beast: AcademicBeastBot,
    opp: PersonaBot,
    seat: int,
    seed: int,
    max_turns: int = 100,
) -> Tuple[int, int, int, int]:
    """
    Executes one match.
    Returns: (winner_rel [-1, 0, 1], turns, calls_made, calls_correct)
    """
    random.seed(seed)
    game = GameState(num_players=2)
    game.deal(14)

    bots = [None, None]
    bots[seat] = beast
    bots[1 - seat] = opp
    beast.player_id = seat
    opp.player_id = 1 - seat
    beast.reset()
    opp.reset()

    turn = 0
    beast_calls = 0
    beast_correct = 0

    while not game.game_over and turn < max_turns:
        curr = game.current_player
        other = 1 - curr
        b_curr = bots[curr]
        b_other = bots[other]
        hand = game.get_hand(curr).cards

        cards, rank = b_curr.decide_play(
            hand,
            build_game_state(
                hand_size=len(hand),
                opponent_hand_size=len(game.get_hand(other).cards),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=game.actions[-1] if game.actions else None,
                hand=hand,
                actions=game.actions,
            ),
        )

        success, _ = game.play_cards(curr, cards, rank)
        if not success:
            turn += 1
            continue

        if game.game_over:
            break

        last_act = game.actions[-1]

        # Response
        if hasattr(b_other, "decide_call"):
            should_call = b_other.decide_call(
                last_act,
                build_game_state(
                    hand_size=len(game.get_hand(other).cards),
                    opponent_hand_size=len(game.get_hand(curr).cards),
                    pile_size=game.get_pile_size(),
                    draw_pile_size=len(game.draw_pile),
                    turn_number=turn,
                    last_action=last_act,
                    hand=game.get_hand(other).cards,
                    actions=game.actions,
                ),
            )
        else:
            should_call = False

        if should_call:
            if other == seat:
                beast_calls += 1
                if last_act.was_bluff:
                    beast_correct += 1
            game.call_bluff(caller=other)
        else:
            game.pass_turn(passer=other)

        turn += 1

    if game.winner is None:
        rel_winner = 0
    elif game.winner == seat:
        rel_winner = 1
    else:
        rel_winner = -1

    return rel_winner, turn, beast_calls, beast_correct


def create_beast_for_regime(regime: str) -> AcademicBeastBot:
    if regime == "Balanced_TDMoE":
        return AcademicBeastBot(
            schedule_type="sigmoidal",
            nn_floor=0.15,
            decay_tau=8.0,
            bet_lookahead=False,
            use_nn=True,
        )
    elif regime == "Heavy_Bayesian":
        return AcademicBeastBot(
            schedule_type="exponential",
            nn_floor=0.05,
            decay_tau=3.0,
            bet_lookahead=False,
            use_nn=True,
        )
    elif regime == "BET_Lookahead":
        return AcademicBeastBot(
            schedule_type="sigmoidal",
            nn_floor=0.15,
            decay_tau=8.0,
            bet_lookahead=True,
            use_nn=True,
        )
    elif regime == "Pure_Bayesian":
        return AcademicBeastBot(
            schedule_type="exponential",
            nn_floor=0.0,
            decay_tau=1.0,
            use_nn=False,
            bet_lookahead=True,
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")


def _worker_task(args) -> Tuple[str, int, str, int, int, int, int, int]:
    regime, seed, p_idx, p_name, true_bluff_rate, true_call_rate, bluff_size_pref, honest_dump_multi, games_count = args
    beast = create_beast_for_regime(regime)
    opp = PersonaBot(p_name, true_bluff_rate, true_call_rate, bluff_size_pref, honest_dump_multi)

    pw, pl, pd = 0, 0, 0
    c_tot, c_corr = 0, 0

    for g in range(games_count):
        seat = g % 2
        g_seed = seed + p_idx * 1000 + g
        outcome, turns, c_made, c_ok = run_single_match(beast, opp, seat, g_seed)
        if outcome == 1:
            pw += 1
        elif outcome == -1:
            pl += 1
        else:
            pd += 1
        c_tot += c_made
        c_corr += c_ok

    return regime, seed, p_name, pw, pl, pd, c_tot, c_corr


def run_3seed_study(games_per_persona: int = 50, output_path: str = "data/bayesian_weightage_3seed_study.json"):
    seeds = [20260911, 20260912, 20260913]
    regimes = ["Balanced_TDMoE", "Heavy_Bayesian", "BET_Lookahead", "Pure_Bayesian"]
    num_workers = min(12, os.cpu_count() or 4)

    print("=" * 85)
    print("  PARALLEL 3-SEED SENSITIVITY VERIFICATION STUDY OF BAYESIAN WEIGHTAGE & BET LOOKAHEAD")
    print(f"  Seeds: {seeds}")
    print(f"  Regimes ({len(regimes)}): {regimes}")
    print(f"  Personas: {len(SYNTHETIC_POPULATION)} | N per cell: {games_per_persona} (50/50 seat balanced)")
    print(f"  Worker Processes: {num_workers}")
    total_games = len(seeds) * len(regimes) * len(SYNTHETIC_POPULATION) * games_per_persona
    print(f"  Total Games across sweep: {total_games:,}")
    print("=" * 85)

    # Prepare tasks
    tasks = []
    for seed in seeds:
        for regime in regimes:
            for p_idx, proto in enumerate(SYNTHETIC_POPULATION):
                tasks.append((
                    regime,
                    seed,
                    p_idx,
                    proto.name,
                    proto.true_bluff_rate,
                    proto.true_call_rate,
                    proto.bluff_size_pref,
                    proto.honest_dump_multi,
                    games_per_persona,
                ))

    results = {
        r: {
            "seeds": {str(s): {"wins": 0, "losses": 0, "draws": 0, "calls": 0, "correct": 0, "personas": {}} for s in seeds},
            "pooled": {"w": 0, "l": 0, "d": 0, "calls": 0, "correct": 0}
        }
        for r in regimes
    }

    t0 = time.time()
    completed = 0
    total_cells = len(tasks)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker_task, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            regime, seed, p_name, pw, pl, pd, c_tot, c_corr = f.result()
            s_key = str(seed)

            r_seed = results[regime]["seeds"][s_key]
            r_seed["wins"] += pw
            r_seed["losses"] += pl
            r_seed["draws"] += pd
            r_seed["calls"] += c_tot
            r_seed["correct"] += c_corr
            r_seed["personas"][p_name] = {"w": pw, "l": pl, "d": pd}

            r_pooled = results[regime]["pooled"]
            r_pooled["w"] += pw
            r_pooled["l"] += pl
            r_pooled["d"] += pd
            r_pooled["calls"] += c_tot
            r_pooled["correct"] += c_corr

            completed += 1
            if completed % 20 == 0 or completed == total_cells:
                rate = (completed * games_per_persona) / max(0.1, time.time() - t0)
                print(f"  [Progress] {completed}/{total_cells} cells completed ({rate:.1f} games/s)...")

    # Format summaries per seed
    for seed in seeds:
        s_key = str(seed)
        print(f"\n--- Results for Master Seed {seed} ---")
        for regime in regimes:
            rs = results[regime]["seeds"][s_key]
            tot = rs["wins"] + rs["losses"] + rs["draws"]
            wr = rs["wins"] / tot if tot > 0 else 0.0
            lr = rs["losses"] / tot if tot > 0 else 0.0
            ci_low, ci_high = wilson_score_interval(rs["wins"], tot)
            prec = (rs["correct"] / rs["calls"]) if rs["calls"] > 0 else 0.0
            rs["total"] = tot
            rs["win_rate"] = wr
            rs["loss_rate"] = lr
            rs["ci_95"] = [ci_low, ci_high]
            rs["net_score"] = rs["wins"] - rs["losses"]
            rs["call_precision"] = prec
            print(f"  [{seed}] {regime:<18}: {rs['wins']:4d}W - {rs['losses']:3d}L - {rs['draws']:4d}D ({wr*100:5.2f}% WR [{ci_low*100:4.1f}%, {ci_high*100:4.1f}%], Net {rs['net_score']:+5d})")

    print("\n" + "=" * 95)
    print("  POOLED 3-SEED CONSOLIDATED RESULTS & WILSON 95% CONFIDENCE INTERVALS")
    print("=" * 95)
    print(f"{'Regime':<18} | {'Total':>6} | {'Wins':>6} | {'Losses':>6} | {'Draws':>6} | {'Win Rate':>8} | {'Wilson 95% CI':>16} | {'Loss Rate':>9} | {'Net Score':>9}")
    print("-" * 95)

    for regime in regimes:
        pw = results[regime]["pooled"]["w"]
        pl = results[regime]["pooled"]["l"]
        pd = results[regime]["pooled"]["d"]
        p_total = pw + pl + pd
        p_wr = pw / p_total
        p_lr = pl / p_total
        p_ci_low, p_ci_high = wilson_score_interval(pw, p_total)
        p_calls = results[regime]["pooled"]["calls"]
        p_correct = results[regime]["pooled"]["correct"]
        p_prec = (p_correct / p_calls) if p_calls > 0 else 0.0

        results[regime]["pooled"]["win_rate"] = p_wr
        results[regime]["pooled"]["loss_rate"] = p_lr
        results[regime]["pooled"]["ci_95"] = [p_ci_low, p_ci_high]
        results[regime]["pooled"]["net_score"] = pw - pl
        results[regime]["pooled"]["call_precision"] = p_prec

        ci_str = f"[{p_ci_low*100:.2f}%, {p_ci_high*100:.2f}%]"
        print(f"{regime:<18} | {p_total:6d} | {pw:6d} | {pl:6d} | {pd:6d} | {p_wr*100:7.2f}% | {ci_str:>16} | {p_lr*100:8.2f}% | {pw-pl:+9d}")

    total_time = time.time() - t0
    print("-" * 95)
    print(f"Study completed in {total_time:.1f}s ({total_games/total_time:.1f} games/s).")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Complete multi-seed dataset written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-persona", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/bayesian_weightage_3seed_study.json")
    args = parser.parse_args()

    run_3seed_study(games_per_persona=args.games_per_persona, output_path=args.output)
