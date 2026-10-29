# Predecision Bluff-forecast calibration audit

Status: frozen prospective corrected-rule synthetic diagnostic. This measures
the live Flagship's `RoundAdaptiveBot` forecast before decisions, without
changing its actions or training on hidden labels. It does not test a policy
candidate or authorize deployment.

## Question and fixed comparators

Historical human-game observations have been omitted from this public snapshot. The original frozen plan remains local.

Compare three fixed forecasts at the same predecision legal observations:

1. Current `RoundAdaptiveBot._posterior_bluff` output.
2. Its nominal fixed bluff prior, 0.30, without looking at card counts.
3. A certificate-only control: 1.0 when the same viewer can prove false with
   `_can_prove_false`, otherwise 0.30. This is a control, not a posterior.

No model is fitted, no threshold is swept, and no evaluator target label or
uncalled truth enters the player's forecast. Evaluator-only true packet labels
are joined after each game. Forced final reviews are excluded from the primary
voluntary-response forecast support and counted separately.

## Frozen study and smoke

Use `analysis.round_forecast_audit.traced_policy` with
`analysis.round_arena.play_game` on the actual corrected engine; preserve the
policy object across deals. The runner does not override initiative for an
already constructed fixture, so set the fixture's `current_player` to the
predeclared alternating starter before passing it to `play_game`.
Target streams: Math,
Honest, Random, and one declared non-template `ScriptedRoundOpponent` with
call probability 0.35 and bluff probability 0.60 in every one of its nine
contexts. Both focal seats. One persistent bot profile per target/seat stream,
four sequential games each, total 32 games. Use disjoint stream seeds
`2026122301 + stream_index * 100`, deal seeds `base+10..13`, recycle seeds
`base+50..53`, policy seed `base`, opponent seed `base+1`, starting player
alternates by game. No seed in the 2026122001/2026122101 value study.
Operational smoke: one Math/one Honest game using 2026122401 and 2026122402;
do not include smoke in study scores. Cap each game at 256 decisions and
the serial study at 60 seconds; stop at complete-game boundary. Keep
truncation explicit; do not relabel as a draw/loss. One frozen full run, no
same-seed retry for poor results.

## Integrity and output

For each pending packet, require a single predecision row and a joined
evaluator truth label after the game; fail on duplicate or missing action
indices. Validate that predictions are finite and in [0,1], that selection
is a legal `call_bluff`/`play`/`pass`, and that the bot sees redacted opponent
cards. Keep rows, hands, rank identities, sampled details and individual game
trajectories in memory only. Emit aggregate-only exclusive-create mode-0600
JSON with source hashes/config, completed/truncated game counts, support by
target/seat and selected challenge versus nonchallenge, predecision Brier/NLL,
mean forecast versus observed bluff frequency, and per-game metric means.
Preserve caller RNG. No database read or production write.

Interpretation: macro-average by game/target, report support and selected
versus unselected separately. If the current posterior is worse than the
fixed prior by >=0.02 Brier against at least two target strata with >=4
scored games/target, mark a calibration problem worth a *fresh* candidate
test. Otherwise report mixed/insufficient evidence. This gate is diagnostic
only; no policy promotion even if it flags a problem. Future candidate must
pass fresh predecision calibration plus paired both-seat complete-game tests
and cross-session adaptation checks before replacing the live bot.
