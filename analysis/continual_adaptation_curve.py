"""
Continual Adaptation Velocity & Non-Stationary Shift Experiment.

Measures how quickly HybridBot detects and exploits abrupt behavioral shifts in opponents:
1. Shift A (The Trapper): Honest Rock (p_b=0.00, p_c=0.15) -> Hyper Maniac (p_b=0.70, p_c=0.60)
2. Shift B (The Feeder):  Total Maniac (p_b=0.75, p_c=0.75) -> Passive Honest (p_b=0.04, p_c=0.10)
3. Shift C (The Sheriff): Never Caller (p_c=0.05) -> Hyper Sheriff (p_c=0.90)

Evaluates:
- Trajectory of Bayesian posterior beliefs (Beta distribution mean and variance)
- Real-time modulation of HybridBot calling thresholds and bluffing decisions
- Adaptation Half-Life (turns required to bridge 50% of the parameter delta)
"""

import sys
import os
import json
import random
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank, Deck
from game import Action, GameState
from bots.base import BotInterface, pending_claims_from_actions, pool_by_rank
from bots.hybrid_bot import HybridBot
from test_bots import build_game_state


class NonStationaryPersona(BotInterface):
    """Opponent that changes behavior midway through a session."""

    def __init__(
        self,
        name: str,
        regime_1: Tuple[float, float], # (bluff_rate, call_rate)
        regime_2: Tuple[float, float],
        shift_turn: int = 25,
    ):
        self.name = name
        self.regime_1 = regime_1
        self.regime_2 = regime_2
        self.shift_turn = shift_turn
        self.current_turn = 0
        self.player_id: Optional[int] = None

    def reset(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass

    @property
    def current_rates(self) -> Tuple[float, float]:
        if self.current_turn < self.shift_turn:
            return self.regime_1
        return self.regime_2

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        self.current_turn = game_state.get("turn_number", 0)
        p_bluff, _ = self.current_rates

        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, List[Card]] = {}
        for c in hand:
            rank_counts.setdefault(c.rank, []).append(c)

        will_bluff = (random.random() < p_bluff)

        if not will_bluff:
            best_rank = max(rank_counts.keys(), key=lambda r: (len(rank_counts[r]), r.value))
            return (rank_counts[best_rank][:4], best_rank)

        available_ranks = [r for r in Rank if r not in rank_counts]
        if not available_ranks:
            available_ranks = list(Rank)
        claimed_rank = random.choice(available_ranks)
        dump_count = min(len(hand), random.choice([1, 2, 3]))
        return (hand[:dump_count], claimed_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        self.current_turn = game_state.get("turn_number", 0)
        _, p_call = self.current_rates
        draw_pile_size = game_state.get("draw_pile_size", 0)
        if draw_pile_size <= 0:
            return True
        return random.random() < p_call


def run_adaptation_trial(
    persona: NonStationaryPersona,
    num_games: int = 3,
    max_turns_per_game: int = 60,
) -> Dict:
    """Simulates a multi-game persistent match tracking Bayesian beliefs."""
    hybrid = HybridBot(thompson_sampling=True)
    hybrid.player_id = 0
    persona.player_id = 1

    trajectory = []
    hybrid_actions = []

    global_turn = 0
    shift_point = persona.shift_turn

    for g in range(num_games):
        game = GameState(num_players=2)
        game.deal(14)
        hybrid.reset()
        persona.reset()

        turn = 0
        while not game.game_over and turn < max_turns_per_game:
            current = game.current_player
            other = 1 - current
            global_turn += 1
            persona.current_turn = global_turn

            # Active player plays
            if current == 0:
                h_cards, h_rank = hybrid.decide_play(
                    game.get_hand(0).cards,
                    build_game_state(
                        hand_size=game.get_hand(0).size(),
                        opponent_hand_size=game.get_hand(1).size(),
                        pile_size=game.get_pile_size(),
                        draw_pile_size=len(game.draw_pile),
                        turn_number=turn,
                        last_action=game.actions[-1] if game.actions else None,
                        cards_played=game.get_cards_played(),
                        actions=game.actions,
                    ),
                )
                if not h_cards:
                    turn += 1
                    continue
                game.play_cards(0, h_cards, h_rank)
                last_act = game.actions[-1]

                should_call = persona.decide_call(
                    last_act,
                    build_game_state(
                        hand_size=game.get_hand(1).size(),
                        opponent_hand_size=game.get_hand(0).size(),
                        pile_size=game.get_pile_size(),
                        draw_pile_size=len(game.draw_pile),
                        turn_number=turn,
                        last_action=last_act,
                        cards_played=game.get_cards_played(),
                        actions=game.actions,
                    ),
                )
                if should_call:
                    success, _, act = game.call_bluff(1)
                    if success and act:
                        hybrid.observe_action(act, game.get_hand(1).size())
                else:
                    if game.can_pass():
                        game.pass_turn(1)
                        pass_act = Action(1, [], Rank.TWO, False, False, False, game.get_pile_size())
                        hybrid.observe_action(pass_act, game.get_hand(1).size())
                    else:
                        game.call_bluff(1)

            else:
                p_cards, p_rank = persona.decide_play(
                    game.get_hand(1).cards,
                    build_game_state(
                        hand_size=game.get_hand(1).size(),
                        opponent_hand_size=game.get_hand(0).size(),
                        pile_size=game.get_pile_size(),
                        draw_pile_size=len(game.draw_pile),
                        turn_number=turn,
                        last_action=game.actions[-1] if game.actions else None,
                        cards_played=game.get_cards_played(),
                        actions=game.actions,
                    ),
                )
                if not p_cards:
                    turn += 1
                    continue
                game.play_cards(1, p_cards, p_rank)
                last_act = game.actions[-1]

                should_call = hybrid.decide_call(
                    last_act,
                    build_game_state(
                        hand_size=game.get_hand(0).size(),
                        opponent_hand_size=game.get_hand(1).size(),
                        pile_size=game.get_pile_size(),
                        draw_pile_size=len(game.draw_pile),
                        turn_number=turn,
                        last_action=last_act,
                        cards_played=game.get_cards_played(),
                        actions=game.actions,
                    ),
                )
                if should_call:
                    success, _, act = game.call_bluff(0)
                    if success and act:
                        hybrid.observe_action(act, game.get_hand(1).size())
                else:
                    if game.can_pass():
                        game.pass_turn(0)
                    else:
                        game.call_bluff(0)

            # Record trajectory
            true_b, true_c = persona.current_rates
            est_b = hybrid.model.overall_bluff.mean()
            est_c = hybrid.model.call_frequency.mean()
            var_b = hybrid.model.overall_bluff.variance()

            trajectory.append({
                "global_turn": global_turn,
                "game": g + 1,
                "true_bluff": true_b,
                "true_call": true_c,
                "est_bluff": est_b,
                "est_call": est_c,
                "variance_bluff": var_b,
                "actions_observed": hybrid.model.total_actions_observed,
            })
            turn += 1

    return {
        "persona": persona.name,
        "shift_turn": shift_point,
        "trajectory": trajectory,
    }


def analyze_continual_adaptation():
    print("================================================================================")
    print("      Continual Online Adaptation & Non-Stationary Shift Analysis               ")
    print("================================================================================")

    test_shifts = [
        NonStationaryPersona(
            name="Shift_A_The_Trapper",
            regime_1=(0.00, 0.15),  # Honest Rock
            regime_2=(0.70, 0.60),  # Hyper Maniac
            shift_turn=35,
        ),
        NonStationaryPersona(
            name="Shift_B_The_Feeder",
            regime_1=(0.75, 0.75),  # Maniac
            regime_2=(0.04, 0.10),  # Passive Honest
            shift_turn=35,
        ),
        NonStationaryPersona(
            name="Shift_C_The_Sheriff",
            regime_1=(0.20, 0.05),  # Never Caller
            regime_2=(0.20, 0.90),  # Hyper Sheriff
            shift_turn=35,
        ),
    ]

    all_results = []

    for persona in test_shifts:
        print(f"\nEvaluating Adaptation: {persona.name} (Shift at turn {persona.shift_turn})...")
        print(f"   Initial Regime: b={persona.regime_1[0]:.2f}, c={persona.regime_1[1]:.2f} -> "
              f"Shifted Regime: b={persona.regime_2[0]:.2f}, c={persona.regime_2[1]:.2f}")

        trials = []
        for rep in range(10):
            res = run_adaptation_trial(persona, num_games=3, max_turns_per_game=50)
            trials.append(res)

        # Compute average trajectory across 10 trials
        min_len = min(len(t["trajectory"]) for t in trials)
        avg_traj = []
        for i in range(min_len):
            est_b = np.mean([t["trajectory"][i]["est_bluff"] for t in trials])
            est_c = np.mean([t["trajectory"][i]["est_call"] for t in trials])
            true_b = trials[0]["trajectory"][i]["true_bluff"]
            true_c = trials[0]["trajectory"][i]["true_call"]
            avg_traj.append({
                "step": i + 1,
                "true_bluff": true_b,
                "true_call": true_c,
                "est_bluff": float(est_b),
                "est_call": float(est_c),
            })

        # Calculate adaptation half-life after shift_turn
        post_shift = [pt for pt in avg_traj if pt["step"] > persona.shift_turn]
        delta_b = persona.regime_2[0] - persona.regime_1[0]
        delta_c = persona.regime_2[1] - persona.regime_1[1]

        target_half_b = persona.regime_1[0] + 0.5 * delta_b
        target_half_c = persona.regime_1[1] + 0.5 * delta_c

        half_life_b = None
        half_life_c = None

        for pt in post_shift:
            turns_since_shift = pt["step"] - persona.shift_turn
            if half_life_b is None and abs(delta_b) > 0.15:
                if (delta_b > 0 and pt["est_bluff"] >= target_half_b) or (delta_b < 0 and pt["est_bluff"] <= target_half_b):
                    half_life_b = turns_since_shift
            if half_life_c is None and abs(delta_c) > 0.15:
                if (delta_c > 0 and pt["est_call"] >= target_half_c) or (delta_c < 0 and pt["est_call"] <= target_half_c):
                    half_life_c = turns_since_shift

        print(f"   => Pre-shift Est at turn {persona.shift_turn}: b={avg_traj[persona.shift_turn-1]['est_bluff']:.3f}, c={avg_traj[persona.shift_turn-1]['est_call']:.3f}")
        print(f"   => Final Est at turn {min_len}:        b={avg_traj[-1]['est_bluff']:.3f}, c={avg_traj[-1]['est_call']:.3f}")
        print(f"   => Adaptation Half-Life: Bluff={half_life_b or '>50'} turns, Call={half_life_c or '>50'} turns")

        all_results.append({
            "name": persona.name,
            "regime_1": persona.regime_1,
            "regime_2": persona.regime_2,
            "shift_turn": persona.shift_turn,
            "half_life_bluff": half_life_b,
            "half_life_call": half_life_c,
            "final_est_bluff": avg_traj[-1]["est_bluff"],
            "final_est_call": avg_traj[-1]["est_call"],
        })

    out_path = "data/adaptation_curve.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAdaptation metrics exported to {out_path}")

    return all_results

if __name__ == "__main__":
    analyze_continual_adaptation()
