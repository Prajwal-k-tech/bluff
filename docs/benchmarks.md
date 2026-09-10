# Benchmarks — Results Tables for the Paper

> **Created:** 2026-09-10 by Muse, joint work with Buffy (OpenCode × Codebuff).
> **Status:** living document — v5 rows land when training completes (Buffy fills §2).
> **Scope:** numbers only. Claims and experiment design live in
> `docs/paper-outline.md`; methods detail in `docs/bot-modes.md`,
> `docs/neural-network.md`, `docs/data-pipeline.md`.

---

## Methodology (read before citing any table)

> **Comparability break 2026-09-10 15:00:** `bluff_probability()` v2 shrinkage
> (decisions.md ADR) changed Honest/CardCount calling distributions. Every
> table below is labeled pre- or post-fix; the two eras are NOT comparable.

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

### Pre-fix era (v1 likelihood — HISTORICAL, incomparable with later)

`test_bots.py --games 20 --seed 0` (PureNN rows omitted — random fallback then):

| Matchup | A wins | B wins | Draws |
|---|---|---|---|
| Random vs Honest | 0 | 20 | 0 |
| Random vs CardCount | 0 | 20 | 0 |
| Random vs Bayesian | 0 | 20 | 0 |
| Honest vs CardCount | 0 | 0 | 20 |
| Honest vs Bayesian | 1 | 1 | 18 |
| CardCount vs Bayesian | 0 | 4 | 16 |

Reading (historical): honest-heavy matchups drew via over-call ping-pong
(call rates ~100%; AGENT_CHAT 13:01, Oracle Gate 1). Superseded below.

### Post-fix era (v2 shrinkage, current)

`test_bots.py --games 20 --seed 0`, post `bluff_probability` v2 (PureNN rows =
early-v6 net, mid-training — illustrative only, NOT citable):

| Matchup | A wins | B wins | Draws |
|---|---|---|---|
| Random vs Honest | 0 | 20 | 0 |
| Random vs CardCount | 0 | 20 | 0 |
| Random vs Bayesian | 0 | 20 | 0 |
| Honest vs CardCount | 0 | 0 | 20 |
| Honest vs Bayesian | 20 | 0 | 0 |
| CardCount vs Bayesian | 13 | 0 | 7 |

Reading: call-war lock broken — Honest 0.0% calls, CardCount 4.3% vs Honest
(measured probe). Honest–CardCount still draw-heavy (0-0-20 with near-zero
calls — symmetric honest shedding hits the cap; 20 games can't separate close
matchups), not ping-pong. Honest–Bayesian flipped 1-1-18 →
20-0-0 because BayesianBot's private counter still over-calls (61.5%) —
same v1 fix pending there (held: trains in v6's league, post-v6 decision).
E4's 100-game rows will tighten all cells.

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

### v5 (`nn/checkpoints/v5.pt`, 39-dim, opponent-conditioned)

> v5 completed 2026-09-10 13:41 (PID 56693, 60k eps, ~39.6 min, 25.3 eps/sec).
> Eval by Fixer 2026-09-10 ~16:05 during solo pickup.

`python -m nn.benchmark --checkpoint nn/checkpoints/v5.pt --games 100 --seed 42`:

| Matchup | NN win rate | NN bluff rate |
|---|---|---|
| PureNN vs Random | 100% | 42% |
| PureNN vs Honest | 0% | 12% |
| PureNN vs CardCount | 0% | 11% |
| PureNN vs Bayesian | 0% | 4% |

Verdict: **FAILS** the vs-Honest criterion (same signature as v4). Conditioning
fixes bluff calibration (play head: bluff 42% vs Random → 12% vs Honest —
opponent-sensitive) but the respond head remains deaf (call 91-99% vs everyone
per Buffy's `conditioning_curve.py`). v6.1 (immediate respond-credit fix +
league roster) targets this.

> v4 re-bench blocked: 38-dim net vs 39-dim encoder — crashes `benchmark.py`.
> Same-eval comparison needs the 38-dim legacy loader (Buffy's call).
>
> **Provenance:** checkpoint `nn/checkpoints/v5.pt` (mtime 2026-09-10 13:41,
> 39-dim/54-act); seed 42; deterministic argmax policy; seats alternated;
> benchmark.py at HEAD (39-dim encoder); 100 games per matchup (400 total,
> ~10.7s CPU).

### v6 (`nn/checkpoints/v6.pt`, 39-dim, partial — credit-fix training run killed)

> v6 first run (PID 92730) logged 5,842 game records (15:14:59–15:19:38, 4.7
> min) before being killed. NOT the 60k target. v6.1 relaunch (PID 95806) failed
> (nohup died with shell). See §v6 anomaly notes below.

`python -m nn.benchmark --checkpoint nn/checkpoints/v6.pt --games 20 --seed 42`:

| Matchup | NN win rate | NN bluff rate |
|---|---|---|
| PureNN vs Random | 100% | 89% |
| PureNN vs Honest | 0% | 3% |
| PureNN vs CardCount | 0% | 3% |
| PureNN vs Bayesian | 0% | 16% |

> **Staleness warning:** v6.pt (15:16:09) is from only ~5.8k/60k episodes —
> partial training, NOT a finished checkpoint. Same 0%-vs-competent-bots
> pattern. Bluff rate 89% vs Random (over-bluffing from incomplete training).
> Not citable; kept for forensics only.

### v6.1 relaunch — PENDING

> v6.1 credit fix + league roster. Exact command prepared, smoke-tested (20
> eps), awaiting scheduler launch.

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
