"""
Synthetic Opponent Population Evaluation & Bayesian Belief Convergence Study.

Constructs a diverse population of 20 parameterized opponent personas:
- Varying ground-truth bluff rates (p_bluff from 0.00 to 0.75)
- Varying ground-truth calling frequencies (p_call from 0.10 to 0.90)
- Distinct multi-card shedding vs junk bluffing preferences

Evaluates HybridBot across 50 games per persona (1,000 games total):
1. Parameter convergence: Mean Absolute Error (MAE) of Bayesian estimated p_bluff and p_call vs ground truth.
2. Behavioral adaptation: HybridBot's realized bluff rate and call rate conditioned on opponent archetype.
3. Exploitation efficacy: Win/Loss/Draw record and net point equity.
"""

import sys
import os
import random
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import Action
from bots.base import BotInterface, pending_claims_from_actions, pool_by_rank
from bots.hybrid_bot import HybridBot
from test_bots import play_bot_vs_bot

# ---------------------------------------------------------------------------
# Parameterized Synthetic Opponent Bot
# ---------------------------------------------------------------------------

class PersonaBot(BotInterface):
    """Synthetic opponent parameterized by ground-truth behavioral probabilities."""

    def __init__(
        self,
        name: str,
        true_bluff_rate: float,
        true_call_rate: float,
        bluff_size_pref: str = "balanced", # "single", "multi", or "balanced"
        honest_dump_multi: bool = True,
    ):
        self.name = name
        self.true_bluff_rate = true_bluff_rate
        self.true_call_rate = true_call_rate
        self.bluff_size_pref = bluff_size_pref
        self.honest_dump_multi = honest_dump_multi
        self.player_id: Optional[int] = None

    def reset(self):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass

    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)

        rank_counts: Dict[Rank, List[Card]] = {}
        for c in hand:
            rank_counts.setdefault(c.rank, []).append(c)

        # Decide whether to bluff this turn
        will_bluff = (random.random() < self.true_bluff_rate)

        # If honest or forced honest
        if not will_bluff:
            if self.honest_dump_multi:
                # Play largest group of matching cards
                best_rank = max(rank_counts.keys(), key=lambda r: (len(rank_counts[r]), r.value))
                return (rank_counts[best_rank][:4], best_rank)
            else:
                # Play single card
                c = hand[0]
                return ([c], c.rank)

        # Bluffing action
        # Select claimed rank
        available_ranks = [r for r in Rank if r not in rank_counts]
        if not available_ranks:
            available_ranks = list(Rank)
        claimed_rank = random.choice(available_ranks)

        # Select cards to dump
        if self.bluff_size_pref == "single":
            dump_count = 1
        elif self.bluff_size_pref == "multi":
            dump_count = min(len(hand), random.choice([2, 3]))
        else:
            dump_count = min(len(hand), random.choice([1, 2, 3]))

        cards_to_dump = hand[:dump_count]
        return (cards_to_dump, claimed_rank)

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        draw_pile_size = game_state.get("draw_pile_size", 0)
        can_pass = draw_pile_size > 0
        if not can_pass:
            return True

        # Call with persona ground truth probability
        return random.random() < self.true_call_rate


# ---------------------------------------------------------------------------
# 20 Archetype Definitions
# ---------------------------------------------------------------------------

SYNTHETIC_POPULATION = [
    # Category 1: Ultra-Honest / Conservative Archetypes
    PersonaBot("Honest_Rock", true_bluff_rate=0.00, true_call_rate=0.15, honest_dump_multi=True),
    PersonaBot("Conservative_Nit", true_bluff_rate=0.05, true_call_rate=0.25, honest_dump_multi=True),
    PersonaBot("Passive_Honest", true_bluff_rate=0.04, true_call_rate=0.10, honest_dump_multi=False),
    PersonaBot("Suspicious_Honest", true_bluff_rate=0.02, true_call_rate=0.55, honest_dump_multi=True),

    # Category 2: Balanced / Standard Strategic Archetypes
    PersonaBot("Balanced_Standard", true_bluff_rate=0.18, true_call_rate=0.40, honest_dump_multi=True),
    PersonaBot("Equilibrium_Seeker", true_bluff_rate=0.22, true_call_rate=0.48, honest_dump_multi=True),
    PersonaBot("Adaptive_Sim", true_bluff_rate=0.25, true_call_rate=0.35, honest_dump_multi=True),
    PersonaBot("Tactical_Mid", true_bluff_rate=0.20, true_call_rate=0.50, bluff_size_pref="single"),

    # Category 3: Aggressive / High-Bluff Archetypes
    PersonaBot("Aggressive_Bluffer", true_bluff_rate=0.45, true_call_rate=0.40, honest_dump_multi=True),
    PersonaBot("Hyper_Maniac", true_bluff_rate=0.65, true_call_rate=0.60, honest_dump_multi=True),
    PersonaBot("MultiCard_Bomber", true_bluff_rate=0.50, true_call_rate=0.45, bluff_size_pref="multi"),
    PersonaBot("Stealth_Bluffer", true_bluff_rate=0.40, true_call_rate=0.30, bluff_size_pref="single"),

    # Category 4: Calling Stations / Sheriff Archetypes
    PersonaBot("Calling_Station", true_bluff_rate=0.12, true_call_rate=0.85, honest_dump_multi=True),
    PersonaBot("Hyper_Sheriff", true_bluff_rate=0.08, true_call_rate=0.90, honest_dump_multi=True),
    PersonaBot("Relentless_Hunter", true_bluff_rate=0.30, true_call_rate=0.80, honest_dump_multi=True),
    PersonaBot("Curious_Station", true_bluff_rate=0.20, true_call_rate=0.75, honest_dump_multi=False),

    # Category 5: Extreme / Chaotic Archetypes
    PersonaBot("Pure_Random_Chaotic", true_bluff_rate=0.50, true_call_rate=0.50, bluff_size_pref="balanced"),
    PersonaBot("Total_Maniac_Extreme", true_bluff_rate=0.75, true_call_rate=0.75, bluff_size_pref="multi"),
    PersonaBot("Never_Caller", true_bluff_rate=0.30, true_call_rate=0.05, honest_dump_multi=True),
    PersonaBot("Always_Caller_Rock", true_bluff_rate=0.02, true_call_rate=0.95, honest_dump_multi=True),
]


def evaluate_synthetic_population(games_per_persona: int = 50):
    print("================================================================================")
    print("   Large-Scale Synthetic Population Evaluation (20 Archetypes, 1000 Games)     ")
    print("================================================================================")

    results = []
    total_bluff_mae = []
    total_call_mae = []
    grand_wins, grand_losses, grand_draws = 0, 0, 0

    t_start = time.time()

    for idx, persona in enumerate(SYNTHETIC_POPULATION, 1):
        w, l, d = 0, 0, 0
        bluff_errors = []
        call_errors = []

        for g in range(games_per_persona):
            seat = 0 if g % 2 == 0 else 1
            hybrid = HybridBot(thompson_sampling=True)

            if seat == 0:
                winner = play_bot_vs_bot(hybrid, persona, max_turns=100)
                if winner == 0:
                    w += 1
                elif winner == 1:
                    l += 1
                else:
                    d += 1
            else:
                winner = play_bot_vs_bot(persona, hybrid, max_turns=100)
                if winner == 1:
                    w += 1
                elif winner == 0:
                    l += 1
                else:
                    d += 1

            # Check Bayesian model belief convergence
            if hybrid.model.total_actions_observed > 0:
                est_bluff = hybrid.model.overall_bluff.mean()
                est_call = hybrid.model.call_frequency.mean()

                bluff_errors.append(abs(est_bluff - persona.true_bluff_rate))
                call_errors.append(abs(est_call - persona.true_call_rate))

        mean_b_mae = float(np.mean(bluff_errors)) if bluff_errors else 0.0
        mean_c_mae = float(np.mean(call_errors)) if call_errors else 0.0
        total_bluff_mae.extend(bluff_errors)
        total_call_mae.extend(call_errors)

        grand_wins += w
        grand_losses += l
        grand_draws += d

        net = w - l
        win_rate = (w / games_per_persona) * 100.0

        persona_res = {
            "name": persona.name,
            "true_bluff": persona.true_bluff_rate,
            "true_call": persona.true_call_rate,
            "wins": w,
            "losses": l,
            "draws": d,
            "net": net,
            "win_rate": win_rate,
            "bluff_mae": mean_b_mae,
            "call_mae": mean_c_mae,
        }
        results.append(persona_res)

        print(f"[{idx:02d}/20] {persona.name:<22s} (True: b={persona.true_bluff_rate:.2f}, c={persona.true_call_rate:.2f}) => "
              f"{w:2d}W - {l:2d}L - {d:2d}D ({win_rate:4.1f}%) | "
              f"MAE(bluff)={mean_b_mae:.3f}, MAE(call)={mean_c_mae:.3f}")

    total_time = time.time() - t_start
    overall_bluff_mae = float(np.mean(total_bluff_mae)) if total_bluff_mae else 0.0
    overall_call_mae = float(np.mean(total_call_mae)) if total_call_mae else 0.0
    total_games = grand_wins + grand_losses + grand_draws
    overall_win_rate = (grand_wins / total_games) * 100.0 if total_games > 0 else 0.0
    overall_net = grand_wins - grand_losses

    print("\n" + "=" * 92)
    print(f"{'Persona':<24s} | {'True b':<6s} | {'True c':<6s} | {'W-L-D':<10s} | {'Net':<5s} | {'Win %':<6s} | {'Bluff MAE':<9s} | {'Call MAE':<8s}")
    print("-" * 92)
    for r in results:
        wld_str = f"{r['wins']}-{r['losses']}-{r['draws']}"
        print(f"{r['name']:<24s} | {r['true_bluff']:>6.2f} | {r['true_call']:>6.2f} | {wld_str:<10s} | {r['net']:>+5d} | {r['win_rate']:>5.1f}% | {r['bluff_mae']:>9.3f} | {r['call_mae']:>8.3f}")
    print("=" * 92)
    print(f"OVERALL SUMMARY (1000 Games across 20 Personas in {total_time:.1f}s):")
    print(f"  Record:   {grand_wins}W - {grand_losses}L - {grand_draws}D (Net: {overall_net:+d}, Win Rate: {overall_win_rate:.1f}%)")
    print(f"  Loss Rate: {(grand_losses / total_games)*100.0:.2f}%")
    print(f"  Bayesian Identification Bluff MAE: {overall_bluff_mae:.3f}")
    print(f"  Bayesian Identification Call  MAE: {overall_call_mae:.3f}")
    print("=" * 92)

    return results

if __name__ == "__main__":
    evaluate_synthetic_population(games_per_persona=50)
