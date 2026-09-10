# Benchmarks — Results Tables for the Paper

> **Created:** 2026-09-10 by Muse, joint work with Buffy (OpenCode × Codebuff).
> **Status:** living document — v5 rows land when training completes (Buffy fills §2).
> **Scope:** numbers only. Claims and experiment design live in
> `docs/paper-outline.md`; methods detail in `docs/bot-modes.md`,
> `docs/neural-network.md`, `docs/data-pipeline.md`.

---

## Methodology (read before citing any table)

- Engine: fixed 100-turn draw rule (`game.py`, LOCKED `game-rules.md` §4).
  Pre-cap tables (e.g. old `handoff.md` round-robins) are NOT comparable.
  (`test_bots.py` retains a `max_turns=300` harness cap as dead safety net —
  the engine cap always fires first.)
- Rule-bot tournaments: `python test_bots.py --games N --seed S [--log F]`
  (`--seed` seeds Python RNG: deck shuffles, RandomBot, Bayesian draws).
- NN benchmarks: `python -m nn.benchmark --checkpoint CKPT --games N`
  (deterministic argmax policy, seats alternated).
- Logged action data: `--log` JSONL → `python -m analysis.report --input F`.
  Two known semantics quirks (Buffy's logger/report, flagged — see AGENT_CHAT
  13:52): (1) "call" records attribute to the play OWNER, so CallAcc reads as
  "callers correct against this bot"; (2) called bluffs land in the "call"
  bucket, so BluffOK is trivially 100%. Primary metrics (win/bluff rates,
  matchups) are unaffected.
- Statistical hygiene (per paper-outline): ≥100 games per matchup for cited
  numbers; Wilson intervals once E4 lands.

---

## 1. Rule-bot baselines (fixed engine)

`test_bots.py --games 20 --seed 0` (PureNN rows omitted — no `final.pt`, random
fallback; logged to `data/terminal/tournament_baseline_s0.jsonl`, gitignored):

| Matchup | A wins | B wins | Draws |
|---|---|---|---|
| Random vs Honest | 0 | 20 | 0 |
| Random vs CardCount | 0 | 20 | 0 |
| Random vs Bayesian | 0 | 20 | 0 |
| Honest vs CardCount | 0 | 0 | 20 |
| Honest vs Bayesian | 1 | 1 | 18 |
| CardCount vs Bayesian | 0 | 4 | 16 |

Reading: honest-heavy matchups are draw-heavy on the capped engine (over-call
ping-pong — see AGENT_CHAT 13:01, Oracle Gate 1). Decisive games come from
bluff pressure (Random's 98% bluffs get caught; Bayesian's 12% bluffs leak
through rare callers). Baselines are valid but low-resolution — E4's 100-game
rows will tighten them.

---

## 2. PureNN checkpoints vs baselines (deterministic, 100 games each)

### v4 (`nn/checkpoints/v4.pt`, 38-dim, unconditioned baseline)

`nn/benchmark.py --checkpoint nn/checkpoints/v4.pt --games 100`:

| Matchup | NN win rate | NN bluff rate |
|---|---|---|
| PureNN vs Random | 100% | 48% |
| PureNN vs Honest | 0% | 12% |
| PureNN vs CardCount | 0% | 5% |
| PureNN vs Bayesian | 0% | 3% |

Verdict: **FAILS** the v4 criterion (vs Honest >50%). Cause: deterministic
policy calls ~90% vs never-bluffing Honest (independent traces agree).
Keep as the unconditioned baseline for claim #2.

### v5 (`nn/checkpoints/v5.pt`, 39-dim, opponent-conditioned) — PENDING

> Buffy fills when training lands (~2.5–3h from 13:01). Decides claim #2:
> v4 (unconditioned) vs v5 (conditioned) on the same eval.
> Known blocker: v4.pt is unloadable at HEAD (38-dim net vs 39-dim encoder) —
> greeted by a warned fallback (HOTFIX, `bots/pure_nn_bot.py`); v4-vs-v5
> same-eval needs the 38-dim legacy loader path (Buffy's call).

| Matchup | NN win rate | NN bluff rate |
|---|---|---|
| PureNN vs Random | pending | pending |
| PureNN vs Honest | pending | pending |
| PureNN vs CardCount | pending | pending |
| PureNN vs Bayesian | pending | pending |

---

## 3. Full round-robin incl. trained NN (E4) — PENDING

100 games/matchup, `benchmark.py` + `test_bots.py --log`, Wilson intervals.
Blocked on v5 verdict.

## 4. Bluff-calibration figures (E6) — PENDING

Bluff-rate-by-hand-size (desperation) curves + Yeung-equilibrium comparison,
from E4 logs via `analysis/report.py`. Reported semantics quirks (§Methodology)
must be labeled on the figures.

---

*Update this file when v5/E4 land. Numbers here are the paper's raw material —
keep provenance (command, seed, checkpoint) on every row added.*
