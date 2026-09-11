"""
Benchmark & Half-Life Sweep for Temporal Decaying Mixture of Experts (TD-MoE).

Compares:
1. Fast Decay (tau = 4.0) — rapid hand-off to Bayesian exploitation by turn 4-8.
2. Balanced Decay (tau = 8.0) — smooth transition with equilibrium around turn 8-12.
3. Slow Decay (tau = 16.0) — persistent NN prior guiding well into the late game.
4. Static Control (no temporal decay, fixed w_nn=0.50, w_bayes=0.50).

Evaluates against:
- HonestBot
- CardCountBot
- BayesianBot
- RandomBot

Metrics:
- Decisive Win Rate, Loss Rate, Draw Rate (with Wilson 95% CIs)
- S_lock (Hand size differential at draw exhaustion)
- A_call (Voluntary call accuracy from caller's perspective)
- Strict 50/50 seat alternation
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
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.random_bot import RandomBot

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
    game = GameState(num_players=2)
    game.deal(14)

    bots = [make_measured(), make_opponent()]
    for i, b in enumerate(bots):
        if hasattr(b, "player_id"):
            b.player_id = i
        b.reset()

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
    return {
        "winner": game.winner,
        "s_lock_measured": s_lock[measured_seat],
        "calls": calls,
        "correct": correct,
    }


def evaluate_td_moe_condition(name: str, bot_maker, games_per_opp: int, seed: int) -> dict:
    random.seed(seed)
    res = {"name": name, "wins": 0, "losses": 0, "draws": 0,
           "total_calls": 0, "correct_calls": 0,
           "s_lock_sum": 0, "games": 0, "per_opp": {}}

    for opp_name, opp_cls in OPPONENTS.items():
        w = l = d = 0
        calls = correct = 0
        s_sum = 0
        for g in range(games_per_opp):
            seat = g % 2  # 50/50 seat alternation
            if seat == 0:
                r = play_measured_game(bot_maker, opp_cls, 0)
            else:
                r = play_measured_game(opp_cls, bot_maker, 1)

            if r["winner"] is None:
                d += 1
            elif r["winner"] == seat:
                w += 1
            else:
                l += 1
            calls += r["calls"]
            correct += r["correct"]
            s_sum += r["s_lock_measured"]

        res["per_opp"][opp_name] = {
            "w": w, "l": l, "d": d,
            "a_call": (100.0 * correct / calls) if calls else 0.0,
            "s_lock": s_sum / games_per_opp,
        }
        res["wins"] += w
        res["losses"] += l
        res["draws"] += d
        res["total_calls"] += calls
        res["correct_calls"] += correct
        res["s_lock_sum"] += s_sum
        res["games"] += games_per_opp

    total = res["games"]
    res["win_rate"] = 100.0 * res["wins"] / total
    res["loss_rate"] = 100.0 * res["losses"] / total
    res["draw_rate"] = 100.0 * res["draws"] / total
    res["w_ci"] = wilson_ci(res["wins"], total)
    res["call_accuracy"] = (100.0 * res["correct_calls"] / res["total_calls"]) if res["total_calls"] else 0.0
    res["mean_s_lock"] = res["s_lock_sum"] / total
    return res


def run_sweep(games_per_opp: int = 100, seed: int = 20260911):
    print("=" * 70)
    print(f"  TD-MoE DECAY HALF-LIFE SWEEP — N={games_per_opp}/matchup, Seed={seed}")
    print("=" * 70)

    class StaticBeast(AcademicBeastBot):
        """Static Neutral Control: fixed 50/50 NN and Bayesian weighting."""
        def get_td_moe_weights(self, tau=None, w_floor=None):
            return 0.50, 0.50

    conditions = {
        "TD_MoE_Fast_tau4": lambda: AcademicBeastBot(decay_tau=4.0),
        "TD_MoE_Balanced_tau8": lambda: AcademicBeastBot(decay_tau=8.0),
        "TD_MoE_Slow_tau16": lambda: AcademicBeastBot(decay_tau=16.0),
        "Static_Control_50_50": lambda: StaticBeast(),
    }

    results = {}
    for name, maker in conditions.items():
        print(f"\n--- Evaluating {name} ---", flush=True)
        r = evaluate_td_moe_condition(name, maker, games_per_opp, seed)
        results[name] = r
        print(f"  Result: {r['wins']}W - {r['losses']}L - {r['draws']}D | "
              f"Win Rate: {r['win_rate']:.1f}% (CI [{r['w_ci'][0]*100:.1f}%, {r['w_ci'][1]*100:.1f}%]) | "
              f"Loss Rate: {r['loss_rate']:.1f}% | "
              f"A_call: {r['call_accuracy']:.1f}% | S_lock: {r['mean_s_lock']:.2f}")
        for opp, stats in r["per_opp"].items():
            print(f"    vs {opp:10s}: {stats['w']:3d}W - {stats['l']:3d}L - {stats['d']:3d}D | "
                  f"A_call: {stats['a_call']:.1f}% | S_lock: {stats['s_lock']:.2f}")

    out_file = "data/benchmark_td_moe.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Benchmark sweep successfully exported to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 games/opp)")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    g = 2 if args.smoke else args.games
    run_sweep(games_per_opp=g, seed=args.seed)
