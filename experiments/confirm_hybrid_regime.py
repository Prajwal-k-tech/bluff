"""
Confirmation Run for HybridBot Winning Regime — CORRECTED to Gate-3/4 spec.
(Owner: Tess 2026-09-11 — spec correction assigned on AGENT_CHAT after the
original draft mismatched: N=100 not 400, 2 opponents not 4, AcademicBeastBot
instead of the promoted HybridBot parameters, A_call tracked in seat-0 games
only, no sensitivity pass.)

Spec (docs/claims-drawlock-calibration.md §Action items, item 2):
- MAIN RUN: N=400 games per matchup × 4 opponents (Honest, CardCount,
  Bayesian, Random), strict 50/50 seat alternation, fresh seed 20260911.
- CONDITIONS: PROMOTED regime — HybridBot(w_model_cap=0.75, exploit_mult=2.5,
  call_mult=1.8, variance_scaled=True) — vs CONSERVATIVE baseline —
  HybridBot(0.40, 1.5, 1.2, variance_scaled=False).
- Primary endpoints: S_lock (hand sizes at draw-pile exhaustion), A_call
  (measured bot's voluntary-call accuracy, caller perspective, BOTH seats),
  W/L/D with Wilson 95% CIs.
- SENSITIVITY PASS: 3 extra seeds {1, 2, 3} × N=100/matchup (seed-dependence
  check of the W/L/D ordering; Gate finding C4).

Run:  python3 experiments/confirm_hybrid_regime.py [--smoke]
      --smoke reduces N to 2/matchup (pipeline validation only — not citable)
"""

import sys
import os
import math
import random
import json
import argparse
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import BotInterface, build_game_state
from bots.hybrid_bot import HybridBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.random_bot import RandomBot

PROMOTED = dict(w_model_cap=0.75, exploit_mult=2.5, call_mult=1.8, variance_scaled=True)
BASELINE = dict(w_model_cap=0.40, exploit_mult=1.5, call_mult=1.2, variance_scaled=False)

OPPONENTS = {
    "Honest": HonestBot,
    "CardCount": CardCountBot,
    "Bayesian": BayesianBot,
    "Random": RandomBot,
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_measured_game(make_measured, make_opponent, measured_seat: int,
                       max_turns: int = 100) -> dict:
    """One game; metrics attributed to the MEASURED bot regardless of seat."""
    game = GameState(num_players=2)
    game.deal(14)

    bots = [make_measured(), make_opponent()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()

    turn = 0
    s_lock = None
    calls = 0          # measured bot's voluntary calls
    correct = 0        # of those, calls that were right

    while not game.game_over and turn < max_turns:
        current = game.current_player
        other = 1 - current

        if len(game.draw_pile) == 0 and s_lock is None:
            s_lock = (game.get_hand(0).size(), game.get_hand(1).size())

        hand = game.get_hand(current)
        cards, rank = bots[current].decide_play(
            hand=hand.cards,
            game_state=build_game_state(
                hand_size=hand.size(),
                opponent_hand_size=game.get_hand(other).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=game.actions[-1] if game.actions else None,
                cards_played=game.get_cards_played(),
                actions=game.actions,
            ),
        )
        if not cards:
            turn += 1
            continue
        success, _ = game.play_cards(current, cards, rank)
        if not success:
            turn += 1
            continue

        last_action = game.actions[-1]
        should_call = bots[other].decide_call(
            last_action=last_action,
            game_state=build_game_state(
                hand_size=game.get_hand(other).size(),
                opponent_hand_size=game.get_hand(current).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=last_action,
                cards_played=game.get_cards_played(),
                hand=game.get_hand(other).cards,
                actions=game.actions,
            ),
        )

        if should_call:
            if other == measured_seat:
                calls += 1
                if last_action.was_bluff:
                    correct += 1
            success, _, action = game.call_bluff(other)
            if success and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                success, _ = game.pass_turn(passer=other)
                if success:
                    pass_act = Action(
                        player=other, cards_played=[], claimed_rank=Rank.TWO,
                        was_bluff=False, bluff_called=False,
                        caller_was_right=False,
                        pile_size_before=game.get_pile_size(),
                    )
                    bots[current].observe_action(pass_act, game.get_hand(other).size())
                    bots[other].observe_action(pass_act, game.get_hand(current).size())
            # empty draw pile + declined call cannot happen: driver forces the call
            turn += 1
            continue

        turn += 1

    if s_lock is None:
        s_lock = (game.get_hand(0).size(), game.get_hand(1).size())
    return {
        "winner": game.winner,
        "s_lock_measured": s_lock[measured_seat],
        "calls": calls,
        "correct": correct,
    }


def evaluate_condition(name: str, make_measured,
                       games_per_matchup: int, seed: int) -> dict:
    random.seed(seed)
    res = {"name": name, "wins": 0, "losses": 0, "draws": 0,
           "total_calls": 0, "correct_calls": 0,
           "s_lock_sum": 0, "games": 0, "per_opp": {}}

    for opp_name, opp_cls in OPPONENTS.items():
        w = l = d = 0
        calls = correct = 0
        s_sum = 0
        for g in range(games_per_matchup):
            measured_seat = g % 2  # strict 50/50 seat alternation
            if measured_seat == 0:
                r = play_measured_game(make_measured, opp_cls, 0)
            else:
                r = play_measured_game(opp_cls, make_measured, 1)
            if r["winner"] is None:
                d += 1
            elif r["winner"] == measured_seat:
                w += 1
            else:
                l += 1
            calls += r["calls"]; correct += r["correct"]
            s_sum += r["s_lock_measured"]
        res["per_opp"][opp_name] = {"w": w, "l": l, "d": d,
                                    "a_call": (100.0 * correct / calls) if calls else 0.0,
                                    "s_lock": s_sum / games_per_matchup}
        res["wins"] += w; res["losses"] += l; res["draws"] += d
        res["total_calls"] += calls; res["correct_calls"] += correct
        res["s_lock_sum"] += s_sum; res["games"] += games_per_matchup

    total = res["games"]
    res["win_rate"] = 100.0 * res["wins"] / total
    res["loss_rate"] = 100.0 * res["losses"] / total
    res["draw_rate"] = 100.0 * res["draws"] / total
    res["w_ci"] = wilson_ci(res["wins"], total)
    res["call_accuracy"] = (100.0 * res["correct_calls"] / res["total_calls"]) if res["total_calls"] else 0.0
    res["mean_s_lock"] = res["s_lock_sum"] / total
    return res


def run_confirmation(smoke: bool = False):
    n_main = 2 if smoke else 400
    n_sens = 2 if smoke else 100
    main_seed = 20260911
    sens_seeds = [1, 2, 3]

    conditions = {
        "baseline_promoted": (lambda: HybridBot(**PROMOTED), main_seed),
        "conservative_baseline": (lambda: HybridBot(**BASELINE), main_seed),
    }

    out = {"main": {}, "sensitivity": {}}

    for name, (maker, seed) in conditions.items():
        print(f"\n=== MAIN: {name} — N={n_main}/matchup × 4 opponents, seed {seed} ===", flush=True)
        r = evaluate_condition(name, maker, n_main, seed)
        out["main"][name] = r
        print(f"  {r['wins']}W-{r['losses']}L-{r['draws']}D | win {r['win_rate']:.1f}% "
              f"CI[{r['w_ci'][0]*100:.1f},{r['w_ci'][1]*100:.1f}] | "
              f"A_call={r['call_accuracy']:.1f}% | S_lock={r['mean_s_lock']:.2f}", flush=True)
        for opp, s in r["per_opp"].items():
            print(f"    vs {opp}: {s['w']}-{s['l']}-{s['d']} | A_call {s['a_call']:.1f}% | S_lock {s['s_lock']:.2f}", flush=True)

    print(f"\n=== SENSITIVITY: 3 seeds × N={n_sens}/matchup ===", flush=True)
    for seed in sens_seeds:
        out["sensitivity"][seed] = {}
        for name, (maker, _) in conditions.items():
            r = evaluate_condition(name, maker, n_sens, seed)
            out["sensitivity"][seed][name] = {k: r[k] for k in ("wins", "losses", "draws", "win_rate")}
            print(f"  seed {seed} | {name}: {r['wins']}W-{r['losses']}L-{r['draws']}D ({r['win_rate']:.1f}%)", flush=True)

    # Verdict: promoted beats baseline on win rate AND does not lose on S_lock/A_call
    m = out["main"]
    pw, bw = m["baseline_promoted"]["win_rate"], m["conservative_baseline"]["win_rate"]
    out["verdict"] = {
        "promoted_win_rate": pw, "baseline_win_rate": bw,
        "promoted_wins": pw > bw,
        "s_lock_not_worse": m["baseline_promoted"]["mean_s_lock"] <= m["conservative_baseline"]["mean_s_lock"] + 0.5,
        "a_call_not_worse": m["baseline_promoted"]["call_accuracy"] >= m["conservative_baseline"]["call_accuracy"] - 2.0,
        "seed_stable": all(
            out["sensitivity"][s]["baseline_promoted"]["win_rate"] >= out["sensitivity"][s]["conservative_baseline"]["win_rate"]
            for s in sens_seeds),
    }
    out_path = "data/confirmation_hybrid_regime.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nVERDICT: promoted>{'Yes' if out['verdict']['promoted_wins'] else 'No'} | "
          f"S_lock ok={out['verdict']['s_lock_not_worse']} | "
          f"A_call ok={out['verdict']['a_call_not_worse']} | "
          f"seed-stable={out['verdict']['seed_stable']}")
    print(f"[OK] Results saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="N=2/matchup pipeline validation (not citable)")
    args = ap.parse_args()
    run_confirmation(smoke=args.smoke)
