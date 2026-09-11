"""
SOTA Arbitration Benchmark: Does PureBayes > Balanced replicate?
Design arbitrating the A23 Bayesian weightage claim at definitive scale.

Conditions (locked):
  1. Balanced_TDMoE  — sigmoidal S-curve, τ=8.0, w_floor=0.15
  2. Heavy_Bayesian  — sigmoidal S-curve, τ=4.0, w_floor=0.05
  3. Pure_Bayesian   — w_nn=0.0, 100% Bayesian-combinatorial
  4. Dewey_EV_Overdrive — Balanced + risk_aversion=0.75

Protocol:
  - 20 synthetic personas, N=250 games/matchup, strict 50/50 seat split
  - Master seed 20260911 (A23 direct comparability)
  - Second seed 20260912 on PureBayes vs Balanced finalists
  - Cheap calibration: rule-vs-rule round-robin N=200
  - Cheap NN eval: final.pt, v61_best.pt vs baselines N=200
  - human_adapted.pt: included (same BluffNet loader)
"""

import os
import sys
import time
import json
import math
import random
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.abspath("."))

from game import GameState, Action
from cards import Card, Rank
from bots.base import build_game_state
from bots.academic_beast_bot import AcademicBeastBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot


# ── Wilson 95% CI ──────────────────────────────────────────────────────────
def wilson_ci(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ── Game loop (reused from tune_bayesian_weightage_beast.py, verified) ─────
def run_single_match(
    bot_a,
    bot_b,
    seat_a: int,
    seed: int,
    max_turns: int = 100,
    bot_a_is_beast: bool = False,
    bot_b_is_beast: bool = False,
) -> Tuple[int, int]:
    """
    Play one game. seat_a is the seat of bot_a (0 or 1).
    Returns (winner, turns). winner: 0=seat_a wins, 1=seat_b wins, -1=draw.
    """
    random.seed(seed)
    game = GameState(num_players=2)
    game.deal(14)

    bots = [None, None]
    bots[seat_a] = bot_a
    bots[1 - seat_a] = bot_b
    bot_a.player_id = seat_a
    bot_b.player_id = 1 - seat_a
    bot_a.reset()
    bot_b.reset()

    turn = 0
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
            succ, _, act = game.call_bluff(other)
            if succ and act:
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
                succ, _, act = game.call_bluff(other)
                if succ and act:
                    b_curr.observe_action(act, len(game.get_hand(other).cards))
                    b_other.observe_action(act, len(game.get_hand(curr).cards))
        turn += 1

    winner = game.winner
    if winner == seat_a:
        return 0, turn
    elif winner == (1 - seat_a):
        return 1, turn
    else:
        return -1, turn


# ── A23 Regime Definitions (exact mapping from tune_bayesian_weightage_beast.py) ──
# Mapping assumption (documented): The A23 regimes map to AcademicBeastBot constructor
# args exactly as in Section 15.2 of benchmarks.md + lines 250-285 of the prior script.
# Condition A = Balanced_TDMoE, B = Heavy_Bayesian, C = Pure_Bayesian, D = Dewey_EV_Overdrive.
REGIMES = {
    "Balanced_TDMoE": lambda: AcademicBeastBot(
        use_nn=True, schedule_type="sigmoidal", decay_tau=8.0, nn_floor=0.15,
    ),
    "Heavy_Bayesian": lambda: AcademicBeastBot(
        use_nn=True, schedule_type="sigmoidal", decay_tau=4.0, nn_floor=0.05,
    ),
    "Pure_Bayesian": lambda: AcademicBeastBot(
        use_nn=False,
    ),
    "Dewey_EV_Overdrive": lambda: AcademicBeastBot(
        use_nn=True, schedule_type="sigmoidal", decay_tau=8.0, nn_floor=0.15,
        risk_aversion=0.75,
    ),
}

RULE_BOTS = {
    "Random": lambda: RandomBot(),
    "Honest": lambda: HonestBot(),
    "CardCount": lambda: CardCountBot(),
    "Bayesian": lambda: BayesianBot(),
}


# ── Per-game JSONL writer ──────────────────────────────────────────────────
class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "w")

    def write(self, record: dict):
        self.f.write(json.dumps(record) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# ── Run block against 20-persona pool ──────────────────────────────────────
def run_persona_pool(
    condition_name: str,
    create_bot_fn,
    games_per_persona: int,
    master_seed: int,
    writer: JsonlWriter,
    bot_is_beast: bool = True,
) -> dict:
    """Run one condition against all 20 personas, log per-game, return summary."""
    total_w, total_l, total_d = 0, 0, 0
    persona_results = {}

    for p_idx, proto in enumerate(SYNTHETIC_POPULATION):
        w, l, d = 0, 0, 0
        for g in range(games_per_persona):
            seat = g % 2  # STRICT 50/50 alternation
            seed = master_seed + p_idx * 10000 + g  # unique seed per game
            bot = create_bot_fn()
            opp = PersonaBot(
                proto.name, proto.true_bluff_rate, proto.true_call_rate,
                proto.bluff_size_pref, proto.honest_dump_multi,
            )

            if bot_is_beast:
                winner_abs, turns = run_single_match(bot, opp, seat, seed)
                # winner_abs: 0=beast wins, 1=opp wins, -1=draw
                if winner_abs == 0:
                    w += 1
                    winner = "beast"
                elif winner_abs == 1:
                    l += 1
                    winner = "opp"
                else:
                    d += 1
                    winner = "draw"
            else:
                # Rule-vs-rule: both are BotInterface
                winner_abs, turns = run_single_match(bot, opp, seat, seed, bot_a_is_beast=False, bot_b_is_beast=False)
                if winner_abs == 0:
                    w += 1
                    winner = "A"
                elif winner_abs == 1:
                    l += 1
                    winner = "B"
                else:
                    d += 1
                    winner = "draw"

            writer.write({
                "condition": condition_name,
                "persona": proto.name,
                "seed": seed,
                "seat": seat,
                "winner": winner,
                "turns": turns,
            })

        total_w += w
        total_l += l
        total_d += d
        persona_results[proto.name] = {"w": w, "l": l, "d": d}

    total_games = len(SYNTHETIC_POPULATION) * games_per_persona
    return {
        "condition": condition_name,
        "wins": total_w,
        "losses": total_l,
        "draws": total_d,
        "total": total_games,
        "win_rate": total_w / total_games,
        "loss_rate": total_l / total_games,
        "personas": persona_results,
    }


# ── NN eval via nn/benchmark infrastructure ────────────────────────────────
def run_nn_eval(checkpoint_path: str, checkpoint_label: str,
                games_per_matchup: int, master_seed: int, writer: JsonlWriter) -> dict:
    """Evaluate a PureNN checkpoint against the 4 rule baselines."""
    try:
        from nn.training import NNPlayer, RulePlayer, play_training_game, load_checkpoint
        from nn.state_encoder import StateEncoder
    except ImportError:
        return {"condition": checkpoint_label, "error": "nn.training import failed"}

    net = load_checkpoint(checkpoint_path)
    enc = StateEncoder()

    results = {}
    for baseline_name, BotCls in [
        ("Random", RandomBot), ("Honest", HonestBot),
        ("CardCount", CardCountBot), ("Bayesian", BayesianBot),
    ]:
        w, l, d = 0, 0, 0
        for g in range(games_per_matchup):
            seat = g % 2
            random.seed(master_seed + g)
            nn_player = NNPlayer(net, enc)
            rule_player = RulePlayer(BotCls(), baseline_name)
            # play_training_game always puts NN at seat=0; alternate agent_seat
            _, winner, _ = play_training_game(
                nn_player, rule_player, agent_seat=seat, deterministic=True,
            )
            if winner is None or winner == -1:
                d += 1
                outcome = "draw"
            elif winner == seat:
                w += 1
                outcome = "nn_win"
            else:
                l += 1
                outcome = "nn_loss"
            writer.write({
                "condition": checkpoint_label,
                "baseline": baseline_name,
                "seed": master_seed + g,
                "seat": seat,
                "winner": outcome,
                "turns": -1,
            })
        matchup = f"NN({checkpoint_label})_vs_{baseline_name}"
        results[matchup] = {"w": w, "l": l, "d": d, "n": games_per_matchup}

    return {"condition": checkpoint_label, "matchups": results}


# ── Rule-vs-rule calibration round-robin ────────────────────────────────────
def run_calibration(games_per_matchup: int, master_seed: int, writer: JsonlWriter) -> dict:
    """Cheap rule-vs-rule round-robin: Random/Honest/CardCount/Bayesian."""
    bot_names = ["Random", "Honest", "CardCount", "Bayesian"]
    results = {}
    for i in range(len(bot_names)):
        for j in range(i + 1, len(bot_names)):
            name_a, name_b = bot_names[i], bot_names[j]
            w_a, w_b, d = 0, 0, 0
            for g in range(games_per_matchup):
                seat = g % 2
                seed = master_seed + i * 1000 + j * 100 + g
                bot_a = RULE_BOTS[name_a]()
                bot_b = RULE_BOTS[name_b]()
                winner_abs, turns = run_single_match(bot_a, bot_b, seat, seed)
                if winner_abs == 0:
                    w_a += 1
                    outcome = "A"
                elif winner_abs == 1:
                    w_b += 1
                    outcome = "B"
                else:
                    d += 1
                    outcome = "draw"
                writer.write({
                    "condition": "calibration",
                    "matchup": f"{name_a}_vs_{name_b}",
                    "seed": seed,
                    "seat": seat,
                    "winner": outcome,
                    "turns": turns,
                })
            results[f"{name_a}_vs_{name_b}"] = {"w_a": w_a, "w_b": w_b, "d": d, "n": games_per_matchup}
    return {"condition": "calibration", "matchups": results}


# ── Table formatting ────────────────────────────────────────────────────────
def fmt_row(label, w, l, d, n, ci=None):
    wr = w / n if n else 0
    ci_str = ""
    if ci:
        ci_str = f" [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
    return f"| {label:<30} | {w:>5} | {l:>5} | {d:>5} | {wr*100:>6.2f}%{ci_str} |"


def print_section(title):
    print(f"\n### {title}")
    print(f"| {'Condition':<30} | {'Wins':>5} | {'Losses':>5} | {'Draws':>5} | {'Win Rate [95% CI]':>16} |")
    print(f"|{'-'*32}|{'-'*7}|{'-'*7}|{'-'*7}|{'-'*18}|")


def main():
    parser = argparse.ArgumentParser(description="SOTA Arbitration Benchmark")
    parser.add_argument("--games", type=int, default=250,
                        help="Games per matchup (default 250)")
    parser.add_argument("--calib-games", type=int, default=200,
                        help="Games per calibration matchup (default 200)")
    parser.add_argument("--nn-games", type=int, default=200,
                        help="Games per NN eval matchup (default 200)")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--sensitivity-seed", type=int, default=20260912)
    parser.add_argument("--output", type=str, default="data/sota_arbitration.jsonl")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 20-game pilot only")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    writer = JsonlWriter(args.output)
    t_start = time.time()

    n_games = 20 if args.dry_run else args.games
    print(f"SOTA Arbitration Benchmark — N={n_games}/persona, seed={args.seed}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # ── Part 1: A23 4-regime factorial ───────────────────────────────────
    print("\n[PART 1] A23 4-Regime Factorial (20 personas × N games)")
    regime_results = {}
    for regime_name, factory in REGIMES.items():
        t0 = time.time()
        res = run_persona_pool(regime_name, factory, n_games, args.seed, writer)
        elapsed = time.time() - t0
        regime_results[regime_name] = res
        print(f"  {regime_name}: {res['wins']}W-{res['losses']}L-{res['draws']}D "
              f"(WR={res['win_rate']*100:.2f}%) [{elapsed:.1f}s]")

    # ── Part 2: Second-seed sensitivity (PureBayes vs Balanced only) ────
    print("\n[PART 2] Sensitivity Check (seed 20260912, PureBayes vs Balanced)")
    sensitivity_results = {}
    for regime_name in ["Pure_Bayesian", "Balanced_TDMoE"]:
        factory = REGIMES[regime_name]
        t0 = time.time()
        res = run_persona_pool(
            f"{regime_name}_s20260912", factory, n_games,
            args.sensitivity_seed, writer,
        )
        elapsed = time.time() - t0
        sensitivity_results[regime_name] = res
        print(f"  {regime_name} (seed2): {res['wins']}W-{res['losses']}L-{res['draws']}D "
              f"(WR={res['win_rate']*100:.2f}%) [{elapsed:.1f}s]")

    # ── Part 3: Cheap calibration (rule-vs-rule round-robin) ────────────
    print("\n[PART 3] Calibration Round-Robin (rule bots only, N=200)")
    calib = run_calibration(args.calib_games, args.seed, writer)
    for matchup, r in calib["matchups"].items():
        print(f"  {matchup}: {r['w_a']}W-{r['w_b']}W-{r['d']}D (N={r['n']})")

    # ── Part 4: Cheap NN eval (final.pt, v61_best.pt, human_adapted.pt) ─
    print("\n[PART 4] NN Checkpoint Eval (N=200/matchup)")
    nn_checkpoints = [
        ("nn/checkpoints/final.pt", "final_pt"),
        ("nn/checkpoints/v61_best.pt", "v61_best_pt"),
        ("nn/checkpoints/human_adapted.pt", "human_adapted_pt"),
    ]
    nn_results = {}
    for ckpt_path, ckpt_label in nn_checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path} not found")
            nn_results[ckpt_label] = {"error": "checkpoint not found"}
            continue
        t0 = time.time()
        res = run_nn_eval(ckpt_path, ckpt_label, args.nn_games, args.seed, writer)
        elapsed = time.time() - t0
        nn_results[ckpt_label] = res
        print(f"  {ckpt_label}: [{elapsed:.1f}s]")
        for matchup, r in res.get("matchups", {}).items():
            print(f"    {matchup}: {r['w']}W-{r['l']}L-{r['d']}D")

    writer.close()
    total_elapsed = time.time() - t_start

    # ── Verification: count records ──────────────────────────────────────
    record_count = 0
    with open(args.output) as f:
        for line in f:
            line = line.strip()
            if line:
                record_count += 1
    print(f"\n[VERIFY] Total JSONL records written: {record_count}")
    print(f"[VERIFY] Expected (approx): {n_games*20*4 + n_games*20*2 + args.calib_games*6 + args.nn_games*4*3}")
    print(f"[VERIFY] Total runtime: {total_elapsed:.1f}s")

    # ── Summary tables ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("## PART 1: A23 Regime Comparison (N={}, seed={})".format(n_games, args.seed))
    print_section("A23 Regime Factorial")
    for name, r in regime_results.items():
        ci = wilson_ci(r["wins"], r["total"])
        print(fmt_row(name, r["wins"], r["losses"], r["draws"], r["total"], ci))

    print("\n## PART 2: Sensitivity (seed={})".format(args.sensitivity_seed))
    print_section("Sensitivity Check")
    for name, r in sensitivity_results.items():
        ci = wilson_ci(r["wins"], r["total"])
        label = f"{name} (seed {args.sensitivity_seed})"
        print(fmt_row(label, r["wins"], r["losses"], r["draws"], r["total"], ci))

    print("\n## PART 3: Calibration Round-Robin")
    print("| Matchup | A Wins | B Wins | Draws | N |")
    print("|---|---|---|---|---|")
    for matchup, r in calib["matchups"].items():
        print(f"| {matchup} | {r['w_a']} | {r['w_b']} | {r['d']} | {r['n']} |")

    print("\n## PART 4: NN Checkpoint Eval")
    print("| Checkpoint | Matchup | W/L/D | Win Rate |")
    print("|---|---|---|---|")
    for ckpt_label, res in nn_results.items():
        for matchup, r in res.get("matchups", {}).items():
            wr = r["w"] / r["n"] * 100
            print(f"| {ckpt_label} | {matchup} | {r['w']}-{r['l']}-{r['d']} | {wr:.1f}% |")

    # ── Hand-check verification ──────────────────────────────────────────
    print("\n## VERIFICATION: Hand-Check One Cell")
    verify_cell(args.output, "Pure_Bayesian", "Honest_Rock")

    # ── Per-persona top deltas for PureBayes vs Balanced ─────────────────
    if "Pure_Bayesian" in regime_results and "Balanced_TDMoE" in regime_results:
        pb = regime_results["Pure_Bayesian"]["personas"]
        bal = regime_results["Balanced_TDMoE"]["personas"]
        print("\n## Per-Persona Delta (PureBayes vs Balanced, WR delta)")
        print("| Persona | PureBayes WR | Balanced WR | Delta |")
        print("|---|---|---|---|")
        deltas = []
        for pname in SYNTHETIC_POPULATION[0].__class__.__name__ and [p.name for p in SYNTHETIC_POPULATION]:
            if pname in pb and pname in bal:
                wr_pb = pb[pname]["w"] / n_games * 100
                wr_bal = bal[pname]["w"] / n_games * 100
                delta = wr_pb - wr_bal
                deltas.append((pname, wr_pb, wr_bal, delta))
        deltas.sort(key=lambda x: x[3], reverse=True)
        for pname, wr_pb, wr_bal, delta in deltas:
            print(f"| {pname:<24} | {wr_pb:>5.1f}% | {wr_bal:>5.1f}% | {delta:>+5.1f}% |")


def verify_cell(output_path: str, condition: str, persona: str):
    """Hand-check: recompute W/L/D for one condition×persona cell from JSONL."""
    w, l, d = 0, 0, 0
    seats = []
    with open(output_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("condition") == condition and rec.get("persona") == persona:
                w += 1 if rec["winner"] in ("beast", "A") else 0
                l += 1 if rec["winner"] in ("opp", "B") else 0
                d += 1 if rec["winner"] == "draw" else 0
                seats.append(rec.get("seat", -1))
    n = w + l + d
    seat_0 = sum(1 for s in seats if s == 0)
    seat_1 = sum(1 for s in seats if s == 1)
    print(f"Cell: {condition} × {persona}")
    print(f"  W={w}, L={l}, D={d}, N={n}")
    print(f"  W+L+D == N: {w+l+d == n}")
    print(f"  Seat balance: seat_0={seat_0}, seat_1={seat_1}, balanced={seat_0 == seat_1}")
    ci = wilson_ci(w, n)
    print(f"  Win rate: {w/n*100:.2f}% [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")


if __name__ == "__main__":
    main()
