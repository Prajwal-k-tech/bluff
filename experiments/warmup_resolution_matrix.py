"""
Warmup-vs-gate resolution — the final ADR-012 experiment (Tess).

Tests the last unmeasured policy variant: WarmupHybrid (posterior-sampling
warmup for the first 8 opponent responses, then the mean-gate) against the
two fixed policies on the 4 decisive personas, same-seed.

Prediction: warmup ≈ always on maniacs (the warmup IS always there, and the
formed gate keeps sampling), ≈ never on balanced (the gate OFF post-warmup
matches never's edge) → strictly dominant if both hold.

Run:  python3 experiments/warmup_resolution_matrix.py [--smoke]
"""
import sys, os, argparse, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bots.hybrid_bot import HybridBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from experiments.confirm_hybrid_regime import play_measured_game, wilson_ci

PERSONA_NAMES = ["Passive_Honest", "Total_Maniac_Extreme", "Hyper_Maniac", "Balanced_Standard"]


class WarmupHybrid(HybridBot):
    """Posterior-sampling warmup (first 8 opponent responses) → mean-gate."""
    def _use_thompson(self) -> bool:
        if not self.thompson_sampling:
            return False
        responses = self.opp_calls + self.opp_passes
        if responses < 8:
            return True  # warmup: posterior-weighted exploration
        ob = getattr(self.model, "overall_bluff", None)
        if ob is None:
            return False
        observed = (ob.alpha + ob.beta) - 10  # minus the Beta(3,7) prior
        if observed < 3:
            return False
        return ob.mean() >= 0.35


def persona_factory(name: str):
    proto = next(p for p in SYNTHETIC_POPULATION if p.name == name)
    return lambda: PersonaBot(proto.name, proto.true_bluff_rate,
                              proto.true_call_rate, proto.bluff_size_pref,
                              proto.honest_dump_multi)


CONDITIONS = {
    "warmup_gate": lambda: WarmupHybrid(thompson_sampling=True),
    "always_thompson": lambda: HybridBot(thompson_sampling=True),
    "never_thompson": lambda: HybridBot(thompson_sampling=False),
}


def run(persona: str, condition: str, n_games: int, seed: int) -> dict:
    random.seed(seed)
    make_measured = CONDITIONS[condition]
    make_opponent = persona_factory(persona)
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
    return {"persona": persona, "condition": condition, "seed": seed,
            "w": w, "l": l, "d": d, "win_rate": 100.0 * w / n_games,
            "w_ci": [100.0 * lo, 100.0 * hi],
            "a_call": (100.0 * correct / calls) if calls else 0.0,
            "s_lock": s_sum / n_games}


def main(smoke: bool = False):
    n = 2 if smoke else 100
    seed = 20260913  # same seed across conditions — isolates policy
    out = []
    print(f"Warmup-vs-gate resolution — N={n}/matchup × 4 personas × 3 conditions, "
          f"same seed {seed}", flush=True)
    for persona in PERSONA_NAMES:
        row = {"persona": persona}
        for condition, make in CONDITIONS.items():
            r = run(persona, condition, n, seed) if False else None
            # inline for per-cell seed control
            random.seed(seed)
            make_a = make
            make_b = persona_factory(persona)
            w = l = d = 0; calls = correct = 0; s_sum = 0
            for g in range(n):
                seat = g % 2
                if seat == 0:
                    r = play_measured_game(make_a, make_b, 0)
                else:
                    r = play_measured_game(make_b, make_a, 1)
                if r["winner"] is None: d += 1
                elif r["winner"] == seat: w += 1
                else: l += 1
                calls += r["calls"]; correct += r["correct"]; s_sum += r["s_lock_measured"]
            lo, hi = wilson_ci(w, n)
            row[condition] = {"w": w, "l": l, "d": d, "win_rate": 100.0*w/n,
                              "w_ci": [100*lo, 100*hi],
                              "a_call": (100.0*correct/calls) if calls else 0.0,
                              "s_lock": s_sum/n}
            print(f"  {persona:>22} | {condition:>15} | {w:>3}W-{l:<3}-{d:<3} "
                  f"| win {row[condition]['win_rate']:5.1f}% "
                  f"CI[{row[condition]['w_ci'][0]:.1f},{row[condition]['w_ci'][1]:.1f}]", flush=True)
        out.append(row)

    print("\nVERDICT (warmup vs best-fixed per persona):", flush=True)
    n_ok = 0
    for row in out:
        best_fixed = max(row["always_thompson"]["win_rate"], row["never_thompson"]["win_rate"])
        ok = row["warmup_gate"]["win_rate"] >= best_fixed - 1.0
        n_ok += ok
        print(f"  {row['persona']:>22}: warmup {row['warmup_gate']['win_rate']:.1f}% "
              f"vs best-fixed {best_fixed:.1f}% -> {'OK' if ok else 'MISS'}", flush=True)
    out_path = "data/warmup_resolution_matrix.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] {n_ok}/4 personas — saved to {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    main(smoke=args.smoke)
