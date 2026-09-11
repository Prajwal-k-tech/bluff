"""
Bayesian Weightage Factorial Study in AcademicBeastBot (8,000 Games).
Directly addresses user mandate:
  "did we experiment with the bayesian part having more weightage, logically speaking
   despite however you were trained on an NN, shouldnt how your opponents truly play
   and bluff me the major contributor? are you sure there are no more tweaking around we can do?
   ideally I want the Hybrid NN bot to be an absolute bluff beast able to take on any opponents"

Evaluates 4 distinct Bayesian weightage regimes across all 20 synthetic personas:
  - Condition A: Balanced TD-MoE (w_floor=0.15, tau=8.0, w_bayes up to 0.85)
  - Condition B: Heavy Bayesian (w_floor=0.05, tau=4.0, w_bayes up to 0.95)
  - Condition C: Pure Bayesian (w_nn=0.0, w_bayes=1.0)
  - Condition D: Dewey EV Overdrive (w_floor=0.15, tau=8.0, 2x deception discount)

Experimental Parameters:
  - 20 Synthetic Personas (Maniacs, Stations, Rocks, Balanced, Hunters)
  - N = 100 games per persona per condition (8,000 games total)
  - 50/50 seat alternation (even=seat 0, odd=seat 1)
  - Master seed: 20260911 (deterministic reproducibility)
  - Metrics: Win Rate (WR), Loss Rate (LR), Draw Rate (DR), Net Score, A_call, S_lock
"""

import os
import sys
import time
import json
import math
import random
import argparse
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from game import GameState, Action
from cards import Card, Rank
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from analysis.metrics import (
    compute_call_precision,
)


def run_single_match(
    beast: AcademicBeastBot,
    persona_proto,
    seat: int,
    seed: int,
    max_turns: int = 100,
) -> Tuple[int, int, bool, int, int]:
    """
    Returns: (winner_relative_to_beast [1=win, -1=loss, 0=draw], turns, beast_called_bluff, beast_calls_correct, opp_calls_on_beast)
    """
    random.seed(seed)
    game = GameState(num_players=2)
    game.deal(14)

    opp = PersonaBot(
        persona_proto.name,
        persona_proto.true_bluff_rate,
        persona_proto.true_call_rate,
        persona_proto.bluff_size_pref,
        persona_proto.honest_dump_multi,
    )

    bots = [None, None]
    bots[seat] = beast
    bots[1 - seat] = opp
    beast.player_id = seat
    opp.player_id = 1 - seat
    beast.reset()
    opp.reset()

    turn = 0
    beast_calls = 0
    beast_correct_calls = 0

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
                cards_played=game.get_cards_played(),
                actions=game.actions,
            ),
        )

        if not cards:
            turn += 1
            continue

        success, _ = game.play_cards(curr, cards, rank)
        if game.game_over:
            break

        last_act = game.actions[-1]
        should_call = b_other.decide_call(
            last_act,
            build_game_state(
                hand_size=len(game.get_hand(other).cards),
                opponent_hand_size=len(game.get_hand(curr).cards),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=turn,
                last_action=last_act,
                cards_played=game.get_cards_played(),
                hand=game.get_hand(other).cards,
                actions=game.actions,
            ),
        )

        if should_call:
            if other == seat:
                beast_calls += 1
            succ, _, act = game.call_bluff(other)
            if succ and act:
                if other == seat and act.caller_was_right:
                    beast_correct_calls += 1
                b_curr.observe_action(act, len(game.get_hand(other).cards))
                b_other.observe_action(act, len(game.get_hand(curr).cards))
        else:
            if game.can_pass():
                game.pass_turn(passer=other)
                pass_act = Action(
                    player=other,
                    cards_played=[],
                    claimed_rank=Rank.TWO,
                    was_bluff=False,
                    bluff_called=False,
                    caller_was_right=False,
                    pile_size_before=game.get_pile_size(),
                )
                b_curr.observe_action(pass_act, len(game.get_hand(other).cards))
                b_other.observe_action(pass_act, len(game.get_hand(curr).cards))
            else:
                # Forced call
                if other == seat:
                    beast_calls += 1
                succ, _, act = game.call_bluff(other)
                if succ and act:
                    if other == seat and act.caller_was_right:
                        beast_correct_calls += 1
                    b_curr.observe_action(act, len(game.get_hand(other).cards))
                    b_other.observe_action(act, len(game.get_hand(curr).cards))
        turn += 1

    winner = game.winner
    if winner == seat:
        res = 1
    elif winner == (1 - seat):
        res = -1
    else:
        res = 0

    return res, turn, (beast_calls > 0), beast_correct_calls, beast_calls


def run_regime(
    regime_name: str,
    create_bot_fn,
    games_per_persona: int = 100,
    master_seed: int = 20260911,
) -> dict:
    print(f"\nEvaluating Regime: {regime_name} ({games_per_persona} games/persona across 20 personas)...")
    t0 = time.time()
    total_w = 0
    total_l = 0
    total_d = 0
    persona_results = {}

    for p_idx, proto in enumerate(SYNTHETIC_POPULATION):
        w, l, d = 0, 0, 0
        total_calls = 0
        correct_calls = 0

        for g in range(games_per_persona):
            seat = g % 2
            seed = master_seed + p_idx * 1000 + g
            bot = create_bot_fn()

            res, turns, _, corr, calls = run_single_match(bot, proto, seat, seed)
            if res == 1:
                w += 1
            elif res == -1:
                l += 1
            else:
                d += 1
            total_calls += calls
            correct_calls += corr

        total_w += w
        total_l += l
        total_d += d

        a_call = correct_calls / total_calls if total_calls > 0 else 0.0
        persona_results[proto.name] = {
            "w": w,
            "l": l,
            "d": d,
            "wr": w / games_per_persona,
            "lr": l / games_per_persona,
            "a_call": a_call,
        }

    total_games = len(SYNTHETIC_POPULATION) * games_per_persona
    wr = total_w / total_games
    lr = total_l / total_games
    dr = total_d / total_games
    elapsed = time.time() - t0

    print(f"  -> Result {regime_name}: {total_w}W - {total_l}L - {total_d}D | WR: {wr*100:.2f}% | LR: {lr*100:.2f}% | Net: +{total_w - total_l} in {elapsed:.1f}s")

    return {
        "regime": regime_name,
        "wins": total_w,
        "losses": total_l,
        "draws": total_d,
        "win_rate": wr,
        "loss_rate": lr,
        "draw_rate": dr,
        "net_score": total_w - total_l,
        "elapsed_seconds": elapsed,
        "personas": persona_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-persona", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output", type=str, default="data/bayesian_weightage_study.json")
    args = parser.parse_args()

    print("=" * 80)
    print("  ACADEMIC BEAST BAYESIAN WEIGHTAGE FACTORIAL EXPERIMENT (8,000 GAMES)")
    print(f"  Games per persona: {args.games_per_persona} | Personas: 20 | Total: {args.games_per_persona * 20 * 4:,} games")
    print("=" * 80)

    regimes = [
        (
            "Balanced_TDMoE",
            lambda: AcademicBeastBot(
                use_nn=True,
                schedule_type="sigmoidal",
                decay_tau=8.0,
                nn_floor=0.15,
            ),
        ),
        (
            "Heavy_Bayesian",
            lambda: AcademicBeastBot(
                use_nn=True,
                schedule_type="sigmoidal",
                decay_tau=4.0,
                nn_floor=0.05,
            ),
        ),
        (
            "Pure_Bayesian_NoNN",
            lambda: AcademicBeastBot(
                use_nn=False,
            ),
        ),
        (
            "Dewey_EV_Overdrive",
            lambda: AcademicBeastBot(
                use_nn=True,
                schedule_type="sigmoidal",
                decay_tau=8.0,
                nn_floor=0.15,
                risk_aversion=0.75,  # higher aggression against detected bluffers
            ),
        ),
    ]

    all_results = {}
    for name, factory in regimes:
        res = run_regime(name, factory, games_per_persona=args.games_per_persona, master_seed=args.seed)
        all_results[name] = res

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "metadata": {
                "date": "2026-09-11",
                "games_per_persona": args.games_per_persona,
                "personas_count": len(SYNTHETIC_POPULATION),
                "total_games": len(SYNTHETIC_POPULATION) * args.games_per_persona * len(regimes),
                "seed": args.seed,
            },
            "regimes": all_results,
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("  FACTORIAL STUDY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Regime':<24} | {'Wins':>6} | {'Losses':>6} | {'Draws':>6} | {'Win Rate':>9} | {'Loss Rate':>9} | {'Net':>6}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<24} | {r['wins']:6d} | {r['losses']:6d} | {r['draws']:6d} | {r['win_rate']*100:8.2f}% | {r['loss_rate']*100:8.2f}% | {r['net_score']:+6d}")
    print("=" * 80)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
