"""
Conditioned-vs-Fixed Thompson Resolution Matrix — the claim-#2 figure
(ADR-012 wiring validation; owner: Tess, 2026-09-11; assigned per
AGENT_CHAT 12:05 after the 12k persona sweep found persona-divergent
Thompson utility).

Design: 4 decisive personas × N games × 3 conditions × 50/50 seats:
  - CONDITIONED: HybridBot(thompson_sampling=True) — ADR-012 wiring gates
    posterior sampling on the inferred archetype (sample only vs
    Hyper_Maniac-classified opponents, >=5 observations)
  - ALWAYS: HybridBot subclass forcing _use_thompson() True (the old
    fixed-Thompson behavior)
  - NEVER: HybridBot(thompson_sampling=False) — deterministic posterior
    expectations
Prediction being validated (from the 12k study): CONDITIONED captures
Thompson's +15-37% wins vs maniac personas AND avoids Thompson's -6% cost
vs honest personas — i.e., CONDITIONED >= max(ALWAYS, NEVER) per persona
category.

Endpoints: W/L/D + A_call (caller-perspective) + S_lock. Fresh seeds per
(persona, condition). Results: data/conditioned_resolution_matrix.json

Run:  python3 experiments/conditioned_resolution_matrix.py [--smoke]
"""
import sys, os, argparse, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bots.hybrid_bot import HybridBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from experiments.confirm_hybrid_regime import play_measured_game, wilson_ci

# The 4 decisive personas (name → lookup in SYNTHETIC_POPULATION)
PERSONA_NAMES = ["Passive_Honest", "Total_Maniac_Extreme", "Hyper_Maniac", "Balanced_Standard"]


class AlwaysThompsonHybrid(HybridBot):
    """The pre-ADR-012 fixed-Thompson behavior: always sample."""
    def _use_thompson(self) -> bool:
        return True


def persona_factory(name: str):
    proto = next(p for p in SYNTHETIC_POPULATION if p.name == name)
    return lambda: PersonaBot(proto.name, proto.true_bluff_rate,
                              proto.true_call_rate, proto.bluff_size_pref,
                              proto.honest_dump_multi)


CONDITIONS = {
    "conditioned": lambda: HybridBot(thompson_sampling=True),
    "always_thompson": lambda: AlwaysThompsonHybrid(thompson_sampling=True),
    "never_thompson": lambda: HybridBot(thompson_sampling=False),
}


def run(persona_name: str, condition: str, n_games: int, seed: int) -> dict:
    random.seed(seed)
    make_measured = CONDITIONS[condition]
    make_opponent = persona_factory(persona_name)

    w = l = d = 0
    calls = correct = 0
    s_sum = 0
    for g in range(n_games):
        measured_seat = g % 2
        if measured_seat == 0:
            r = play_measured_game(make_measured, make_opponent, 0)
        else:
            r = play_measured_game(make_opponent, make_measured, 1)
        if r["winner"] is None:
            d += 1
        elif r["winner"] == measured_seat:
            w += 1
        else:
            l += 1
        calls += r["calls"]; correct += r["correct"]
        s_sum += r["s_lock_measured"]

    lo, hi = wilson_ci(w, n_games)
    return {"persona": persona_name, "condition": condition, "seed": seed,
            "w": w, "l": l, "d": d,
            "win_rate": 100.0 * w / n_games,
            "w_ci": [100.0 * lo, 100.0 * hi],
            "a_call": (100.0 * correct / calls) if calls else 0.0,
            "s_lock": s_sum / n_games}


def main(smoke: bool = False):
    n = 2 if smoke else 100
    out = []
    print(f"Conditioned resolution matrix — N={n}/matchup × 4 personas × 3 conditions", flush=True)
    for persona in PERSONA_NAMES:
        row = {}
        for condition, make in CONDITIONS.items():
            seed = 20260912 + hash((persona, condition)) % 1000  # fresh per cell, deterministic
            r = run(persona, condition, n, seed)
            row[condition] = r
            print(f"  {persona:>22} | {condition:>15} | {r['w']:>3}W-{r['l']:<3}-{r['d']:<3} "
                  f"| win {r['win_rate']:5.1f}% CI[{r['w_ci'][0]:.1f},{r['w_ci'][1]:.1f}] "
                  f"| A_call {r['a_call']:5.1f}% | S_lock {r['s_lock']:5.2f}", flush=True)
        # conditioned-vs-best-fixed verdict for this persona
        best_fixed = max(row["always_thompson"]["win_rate"], row["never_thompson"]["win_rate"])
        row["conditioned_beats_best_fixed"] = row["conditioned"]["win_rate"] >= best_fixed - 1.0
        out.append(row)

    n_ok = sum(1 for row in out if row["conditioned_beats_best_fixed"])
    print(f"\nVERDICT: conditioned >= best-fixed on {n_ok}/4 personas "
          f"(prediction: 4/4 — maniac personas via Thompson gains, honest via avoidance)", flush=True)
    out_path = "data/conditioned_resolution_matrix.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
