"""
Adaptive Persona Bayesian Weight & Thompson Sampling Grid Search.

Evaluates 6 architectural regimes across 20 synthetic opponent personas:
- w_model_cap in {0.40, 0.60, 0.75}
- thompson_sampling in {True, False}

Metrics per condition:
- Decisive Win Rate, Loss Rate, Draw Rate (with Wilson 95% CIs)
- Voluntary Call Accuracy (A_call) from caller perspective in both seats
- Bluff Estimation MAE (|p_hat_bluff - p_true_bluff|)
- Call Estimation MAE (|p_hat_call - p_true_call|)
- Hand Size at Draw Exhaustion (S_lock)
- Strict 50/50 seat alternation
"""

import sys
import os
import math
import random
import json
import argparse
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game import GameState, Action
from cards import Rank
from bots.base import BotInterface, build_game_state
from bots.hybrid_bot import HybridBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_measured_game(bot_maker, persona: PersonaBot, measured_seat: int,
                       max_turns: int = 100) -> dict:
    game = GameState(num_players=2)
    game.deal(14)

    measured_bot = bot_maker()
    measured_bot.player_id = measured_seat
    measured_bot.reset()

    persona_bot = PersonaBot(
        name=persona.name,
        true_bluff_rate=persona.true_bluff_rate,
        true_call_rate=persona.true_call_rate,
        bluff_size_pref=persona.bluff_size_pref,
        honest_dump_multi=persona.honest_dump_multi,
    )
    persona_bot.player_id = 1 - measured_seat
    persona_bot.reset()

    bots = [measured_bot, persona_bot] if measured_seat == 0 else [persona_bot, measured_bot]

    turn = 0
    s_lock = None
    calls = 0
    correct = 0

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
            turn += 1
            continue

        turn += 1

    if s_lock is None:
        s_lock = (game.get_hand(0).size(), game.get_hand(1).size())

    # Extract final estimated beliefs
    est_bluff = measured_bot.model.overall_bluff.mean()
    est_call = measured_bot.model.call_frequency.mean()

    return {
        "winner": game.winner,
        "s_lock_measured": s_lock[measured_seat],
        "calls": calls,
        "correct": correct,
        "est_bluff": est_bluff,
        "est_call": est_call,
        "true_bluff": persona.true_bluff_rate,
        "true_call": persona.true_call_rate,
    }


def evaluate_condition(name: str, w_model_cap: float, thompson: bool,
                       games_per_persona: int, seed: int) -> dict:
    random.seed(seed)
    bot_maker = lambda: HybridBot(
        w_model_cap=w_model_cap,
        thompson_sampling=thompson,
        exploit_mult=2.5,
        call_mult=1.8,
        variance_scaled=True,
    )

    res = {
        "condition": name,
        "w_model_cap": w_model_cap,
        "thompson_sampling": thompson,
        "wins": 0, "losses": 0, "draws": 0,
        "total_calls": 0, "correct_calls": 0,
        "s_lock_sum": 0, "games": 0,
        "bluff_mae_sum": 0.0, "call_mae_sum": 0.0,
        "personas": {},
    }

    for persona in SYNTHETIC_POPULATION:
        p_name = persona.name
        w = l = d = 0
        calls = correct = 0
        s_sum = 0
        bluff_mae_p = 0.0
        call_mae_p = 0.0

        for g in range(games_per_persona):
            seat = g % 2  # Strict 50/50 seat alternation
            r = play_measured_game(bot_maker, persona, seat)

            if r["winner"] is None:
                d += 1
            elif r["winner"] == seat:
                w += 1
            else:
                l += 1

            calls += r["calls"]
            correct += r["correct"]
            s_sum += r["s_lock_measured"]
            bluff_mae_p += abs(r["est_bluff"] - r["true_bluff"])
            call_mae_p += abs(r["est_call"] - r["true_call"])

        res["personas"][p_name] = {
            "w": w, "l": l, "d": d,
            "a_call": (100.0 * correct / calls) if calls else 0.0,
            "s_lock": s_sum / games_per_persona,
            "bluff_mae": bluff_mae_p / games_per_persona,
            "call_mae": call_mae_p / games_per_persona,
        }
        res["wins"] += w
        res["losses"] += l
        res["draws"] += d
        res["total_calls"] += calls
        res["correct_calls"] += correct
        res["s_lock_sum"] += s_sum
        res["bluff_mae_sum"] += bluff_mae_p
        res["call_mae_sum"] += call_mae_p
        res["games"] += games_per_persona

    total = res["games"]
    res["win_rate"] = 100.0 * res["wins"] / total
    res["loss_rate"] = 100.0 * res["losses"] / total
    res["draw_rate"] = 100.0 * res["draws"] / total
    res["w_ci"] = wilson_ci(res["wins"], total)
    res["call_accuracy"] = (100.0 * res["correct_calls"] / res["total_calls"]) if res["total_calls"] else 0.0
    res["mean_s_lock"] = res["s_lock_sum"] / total
    res["mean_bluff_mae"] = res["bluff_mae_sum"] / total
    res["mean_call_mae"] = res["call_mae_sum"] / total

    return res


def run_grid(games_per_persona: int = 100, seed: int = 20260911):
    print("=" * 75)
    print(f"  ADAPTIVE PERSONA GRID SWEEP (20 Personas × N={games_per_persona}, Seed={seed})")
    print("=" * 75)

    grid = [
        ("w0.40_thompson_off", 0.40, False),
        ("w0.40_thompson_on",  0.40, True),
        ("w0.60_thompson_off", 0.60, False),
        ("w0.60_thompson_on",  0.60, True),
        ("w0.75_thompson_off", 0.75, False),
        ("w0.75_thompson_on",  0.75, True),
    ]

    results = {}
    for name, w_cap, thomp in grid:
        print(f"\n>>> Running Condition: {name} (w_cap={w_cap}, thompson={thomp}) ...", flush=True)
        r = evaluate_condition(name, w_cap, thomp, games_per_persona, seed)
        results[name] = r
        print(f"  Record: {r['wins']}W - {r['losses']}L - {r['draws']}D | "
              f"Win Rate: {r['win_rate']:.1f}% [{r['w_ci'][0]*100:.1f}%, {r['w_ci'][1]*100:.1f}%] | "
              f"Loss Rate: {r['loss_rate']:.2f}% | "
              f"A_call: {r['call_accuracy']:.1f}% | S_lock: {r['mean_s_lock']:.2f} | "
              f"Bluff MAE: {r['mean_bluff_mae']:.4f} | Call MAE: {r['mean_call_mae']:.4f}", flush=True)

    out_file = "data/adaptive_persona_grid_sweep.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Complete grid results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="Games per persona")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 games/persona)")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    g = 2 if args.smoke else args.games
    run_grid(games_per_persona=g, seed=args.seed)
