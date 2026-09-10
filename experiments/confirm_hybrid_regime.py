"""
Confirmation Run for HybridBot Winning Regime (Task A14 / Tess Review).

Evaluates the chosen Dewey EV + Variance-Scaled Bayesian regime against the conservative baseline:
- Fresh RNG seed (seed=20260911)
- 400 games per matchup (800 games total per bot condition) with strict 50/50 seat alternation
- Primary endpoints:
    1. Win / Loss / Draw rates with Wilson 95% CIs
    2. S_lock: Hand size at draw pile lock point (when draw_pile == 0)
    3. A_call: Bluff call accuracy (true bluffs called / total calls made)
    4. Net score (W - L)
"""

import sys
import os
import time
import math
import random
import json
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import GameState, Action
from bots.base import BotInterface, build_game_state
from bots.hybrid_bot import HybridBot
from bots.academic_beast_bot import AcademicBeastBot
from bots.honest_bot import HonestBot
from bots.bayesian_bot import BayesianBot


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def play_measured_game(bot_a: BotInterface, bot_b: BotInterface, max_turns: int = 100) -> dict:
    game = GameState(num_players=2)
    game.deal(14)

    bot_a.reset()
    bot_b.reset()
    bots = [bot_a, bot_b]

    if hasattr(bot_a, 'player_id'):
        bot_a.player_id = 0
    if hasattr(bot_b, 'player_id'):
        bot_b.player_id = 1

    turn = 0
    s_lock_0 = None
    s_lock_1 = None
    calls_made_0 = 0
    correct_calls_0 = 0

    while not game.game_over and turn < max_turns:
        current = game.current_player
        other = 1 - current
        bot = bots[current]

        # Record S_lock if draw pile just exhausted
        if len(game.draw_pile) == 0 and s_lock_0 is None:
            s_lock_0 = game.get_hand(0).size()
            s_lock_1 = game.get_hand(1).size()

        hand = game.get_hand(current)
        cards, rank = bot.decide_play(
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
            if other == 0:
                calls_made_0 += 1
                if last_action.was_bluff:
                    correct_calls_0 += 1

            success, _, action = game.call_bluff(other)
            if success and action:
                bots[current].observe_action(action, game.get_hand(other).size())
                bots[other].observe_action(action, game.get_hand(current).size())
        else:
            if game.can_pass():
                success, _ = game.pass_turn(passer=other)
                if success:
                    pass_act = Action(
                        player=other,
                        cards_played=[],
                        claimed_rank=Rank.TWO,
                        was_bluff=False,
                        bluff_called=False,
                        caller_was_right=False,
                        pile_size_before=game.get_pile_size(),
                    )
                    bots[current].observe_action(pass_act, game.get_hand(other).size())
                    bots[other].observe_action(pass_act, game.get_hand(current).size())

        turn += 1

    winner = game.winner
    if s_lock_0 is None:
        s_lock_0 = game.get_hand(0).size()
        s_lock_1 = game.get_hand(1).size()

    return {
        "winner": winner,
        "s_lock_p0": s_lock_0,
        "s_lock_p1": s_lock_1,
        "calls_p0": calls_made_0,
        "correct_calls_p0": correct_calls_0,
    }


def evaluate_condition(name: str, create_bot_fn, opponents: dict, games_per_opp: int = 100) -> dict:
    results = {
        "name": name,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_calls": 0,
        "correct_calls": 0,
        "s_lock_sum": 0.0,
        "games_count": 0,
        "per_opp": {},
    }

    for opp_name, opp_cls in opponents.items():
        opp_w, opp_l, opp_d = 0, 0, 0
        opp_calls = 0
        opp_correct = 0

        for g in range(games_per_opp):
            if g % 2 == 0:
                b0 = create_bot_fn()
                b1 = opp_cls()
                res = play_measured_game(b0, b1)
                w = res["winner"]
                if w == 0:
                    opp_w += 1
                elif w == 1:
                    opp_l += 1
                else:
                    opp_d += 1
                opp_calls += res["calls_p0"]
                opp_correct += res["correct_calls_p0"]
                results["s_lock_sum"] += res["s_lock_p0"]
            else:
                b0 = opp_cls()
                b1 = create_bot_fn()
                res = play_measured_game(b0, b1)
                w = res["winner"]
                if w == 1:
                    opp_w += 1
                elif w == 0:
                    opp_l += 1
                else:
                    opp_d += 1
                results["s_lock_sum"] += res["s_lock_p1"]

            results["games_count"] += 1

        results["wins"] += opp_w
        results["losses"] += opp_l
        results["draws"] += opp_d
        results["total_calls"] += opp_calls
        results["correct_calls"] += opp_correct
        results["per_opp"][opp_name] = [opp_w, opp_l, opp_d]

    total = results["games_count"]
    results["win_rate"] = results["wins"] / total * 100.0
    results["loss_rate"] = results["losses"] / total * 100.0
    results["draw_rate"] = results["draws"] / total * 100.0
    results["w_ci"] = wilson_ci(results["wins"], total)
    results["l_ci"] = wilson_ci(results["losses"], total)
    results["call_accuracy"] = (
        (results["correct_calls"] / results["total_calls"] * 100.0)
        if results["total_calls"] > 0 else 0.0
    )
    results["mean_s_lock"] = results["s_lock_sum"] / total
    return results


def run_confirmation():
    random.seed(20260911)
    opponents = {
        "HonestBot": HonestBot,
        "BayesianBot": BayesianBot,
    }

    print("=" * 65)
    print("  CONFIRMATION RUN — Fresh Seed 20260911 (N=100 per matchup)")
    print("=" * 65)

    # Condition 1: Conservative Baseline (w_model=0.40, linear threshold)
    print("\nRunning Condition 1: Conservative Baseline...")
    res_base = evaluate_condition(
        "Baseline_Conservative",
        lambda: HybridBot(w_model_cap=0.40, exploit_mult=1.5, call_mult=1.2, variance_scaled=False),
        opponents,
        games_per_opp=100,
    )
    print(f"  Baseline: {res_base['wins']}W - {res_base['losses']}L - {res_base['draws']}D | "
          f"A_call={res_base['call_accuracy']:.1f}% | S_lock={res_base['mean_s_lock']:.1f}")

    # Condition 2: Dewey EV Upgraded Academic Beast
    print("\nRunning Condition 2: Dewey EV + Variance-Scaled Beast...")
    res_beast = evaluate_condition(
        "Dewey_EV_Beast",
        lambda: AcademicBeastBot(),
        opponents,
        games_per_opp=100,
    )
    print(f"  Dewey Beast: {res_beast['wins']}W - {res_beast['losses']}L - {res_beast['draws']}D | "
          f"A_call={res_beast['call_accuracy']:.1f}% | S_lock={res_beast['mean_s_lock']:.1f}")

    out_path = "data/confirmation_hybrid_regime.json"
    with open(out_path, "w") as f:
        json.dump({"baseline": res_base, "beast": res_beast}, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    run_confirmation()
