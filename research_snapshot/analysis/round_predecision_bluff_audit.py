"""Aggregate-only corrected-rule predecision bluff forecast audit.

Oracle packet truth is joined only after each game. This diagnostic does not
change policy decisions, fit a model, or consume hidden labels online.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import stat
import time

from analysis.round_arena import play_game
from analysis.round_opponents import BehaviorPattern, ScriptedRoundOpponent
from bots.round_policy import (RoundAdaptiveBot, RoundHonestBot, RoundMathBot,
                               RoundRandomBot, _can_prove_false, _context,
                               _field, _packet_size)
from game_v2 import GameState, RULESET_ID

TARGETS = ("math", "honest", "random", "scripted")
FORECASTS = ("posterior", "prior", "certificate")
SCRIPTED_PATTERN = BehaviorPattern((0.35,) * 9, (0.60,) * 9)


def traced_policy(policy_class=RoundAdaptiveBot):
    """Capture only forecasts computed from the player's normal redacted view."""
    class Traced(policy_class):
        def __init__(self, seed=None):
            super().__init__(seed)
            self.forecast_rows = []
            self._audit_prediction = None

        def _posterior_bluff(self, hand, state, action):
            result = super()._posterior_bluff(hand, state, action)
            self._audit_prediction = result[0]
            return result

        def choose_action(self, hand, state):
            self._audit_prediction = None
            choice = super().choose_action(hand, state)
            pending = state.get("last_action")
            if state.get("can_call_bluff") and pending is not None:
                # Fail closed if hidden identities ever cross the policy boundary.
                if "opponent_cards" in state or "opponent_hand" in state:
                    raise ValueError("redacted view contains opponent identities")
                if _field(pending, "was_bluff") is not None:
                    raise ValueError("redacted pending packet exposes truth")
                cards = _field(pending, "cards_played")
                if cards is None or any(card is not None for card in cards):
                    raise ValueError("redacted pending packet exposes card identities")
                posterior = self._audit_prediction
                qty = _packet_size(pending)
                rank = _field(pending, "claimed_rank")
                if type(rank) is not int and hasattr(rank, "value"):
                    rank = rank.value
                if posterior is None:
                    # A guaranteed truthful finish returns before the planner
                    # evaluates its challenge option. The packet is still a
                    # legal predecision forecast opportunity: compute from
                    # the same unchanged, redacted observation before the
                    # selected action reaches the engine.
                    posterior = self._posterior_bluff(hand, state, pending)[0]
                self.forecast_rows.append({
                    "action_index": _field(pending, "action_index"),
                    "predictions": {"posterior": posterior, "prior": 0.30,
                                    "certificate": 1.0 if _can_prove_false(
                                        hand, state, pending) else 0.30},
                    "selected": choice.get("action", choice.get("kind")),
                    "quantity": qty, "pile_size": state["pile_size"],
                    "context": _context(qty, state["pile_size"]),
                })
            return choice
    return Traced


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("analysis/round_predecision_bluff_audit.py",
             "analysis/round_arena.py", "analysis/round_opponents.py",
             "docs/PREDECISION_BLUFF_FORECAST_AUDIT_PLAN_2026-10-13.md",
             "bots/round_policy.py", "bots/round_observation.py",
             "bots/base.py", "game_v2.py", "game.py", "cards.py")
    return {name: sha256((root / name).read_bytes()).hexdigest() for name in names}


def _game_trace(policy, opponent, *, seat, deal_seed, recycle_seed,
                starting_player, max_decisions=256, decision_hook=None):
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("seat must be 0 or 1")
    if type(starting_player) is not int or starting_player not in (0, 1):
        raise ValueError("starting_player must be 0 or 1")
    before = random.getstate()
    try:
        random.seed(deal_seed)
        game = GameState()
        game.deal(14, random_start=False)
        # play_game does not override initiative when given a fixture game.
        game.current_player = starting_player
    finally:
        random.setstate(before)
    policy_start = len(policy.forecast_rows)
    result = play_game([policy, opponent] if seat == 0 else [opponent, policy],
                       game=game, seed=recycle_seed,
                       max_decisions=max_decisions,
                       starting_player=starting_player, decision_hook=decision_hook)
    oracle = {}
    for action in game.actions:
        index = getattr(action, "action_index", None)
        if type(index) is not int or index in oracle:
            raise ValueError("duplicate or invalid engine action index")
        oracle[index] = action
    public = policy.forecast_rows[policy_start:]
    seen = set()
    scored = []
    for row in public:
        index = row["action_index"]
        if type(index) is not int or index in seen:
            raise ValueError("duplicate or invalid forecast action index")
        seen.add(index)
        action = oracle.get(index)
        if action is None or action.player == seat or type(action.was_bluff) is not bool:
            raise ValueError("forecast/oracle action mismatch")
        # Forced terminal challenge is excluded, even if encountered at a cap edge.
        if action.response_was_forced is True:
            raise ValueError("forced response was incorrectly forecast")
        if action.response_kind not in ("play", "pass", "challenge"):
            raise ValueError("pending packet has no resolved legal response")
        if action.response_kind == "challenge" and action.response_was_forced is not False:
            raise ValueError("challenge voluntariness metadata invalid")
        if row["selected"] not in ("call_bluff", "play", "pass"):
            raise ValueError("forecast selection is not a legal response")
        for value in row["predictions"].values():
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("invalid forecast probability")
        scored_row = {"truth": action.was_bluff,
                      "selected": row["selected"] == "call_bluff",
                      "predictions": row["predictions"]}
        if 'reset_selected' in row:
            scored_row['reset_selected'] = row['reset_selected']
        scored.append(scored_row)
    # Every resolved voluntary response by the focal player must have one forecast.
    expected = {a.action_index for a in game.actions
                if a.player != seat and a.response_player == seat
                and a.response_was_forced is False
                and a.response_kind in ("play", "pass", "challenge")}
    if seen != expected:
        raise ValueError("missing or extra predecision forecast rows")
    return {"termination": result.termination, "winner": result.winner,
            "focal_outcome": (None if result.winner is None else
                              "win" if result.winner == seat else "loss"),
            "decisions": result.decisions,
            "rows": scored}


def _metric(rows, forecast):
    n = len(rows)
    if not n:
        return {"support": 0, "mean_predicted": None, "observed_bluff_rate": None,
                "brier": None, "nll": None}
    ps = [r["predictions"][forecast] for r in rows]
    ys = [float(r["truth"]) for r in rows]
    return {"support": n, "mean_predicted": sum(ps) / n,
            "observed_bluff_rate": sum(ys) / n,
            "brier": sum((p-y)**2 for p, y in zip(ps, ys)) / n,
            "nll": -sum(math.log(max(1e-12, p if y else 1-p))
                        for p, y in zip(ps, ys)) / n}


def _aggregate(games):
    # Internal rows are reduced immediately; none are serialized.
    buckets = defaultdict(list)
    game_metrics = defaultdict(list)
    target_game_briers = defaultdict(list)
    game_counts = defaultdict(lambda: {"completed": 0, "truncated": 0,
                                       "voluntary_support": 0})
    for game in games:
        key = (game["target"], game["seat"])
        status = game["termination"]
        game_counts[key]["truncated" if status == "truncated" else "completed"] += 1
        game_counts[key]["voluntary_support"] += len(game["rows"])
        for model in ("posterior", "prior"):
            value = _metric(game["rows"], model)["brier"]
            if value is not None:
                target_game_briers[(game["target"], model)].append(value)
        for selection, keep in (("selected", True), ("nonselected", False)):
            subset = [r for r in game["rows"] if r["selected"] is keep]
            for model in FORECASTS:
                m = _metric(subset, model)
                game_metrics[(key, selection, model)].append(m["brier"])
                buckets[(key, selection, model)].extend(subset)
    out = {}
    for key in sorted({k[0] for k in buckets} | set(game_counts)):
        label = f"{key[0]}/seat_{key[1]}"
        groups = {}
        for selection in ("selected", "nonselected"):
            models = {}
            for model in FORECASTS:
                rows = buckets[(key, selection, model)]
                metric = _metric(rows, model)
                vals = [v for v in game_metrics[(key, selection, model)] if v is not None]
                metric["per_game_brier_mean"] = sum(vals) / len(vals) if vals else None
                metric["games_scored"] = len(vals)
                models[model] = metric
            groups[selection] = models
        out[label] = {**game_counts[key], "selection": groups}
    qualifying = []
    target_names = sorted({name for name, _ in target_game_briers})
    for name in target_names:
        posterior = target_game_briers[(name, "posterior")]
        prior = target_game_briers[(name, "prior")]
        if len(posterior) >= 4 and len(prior) >= 4:
            if (sum(posterior) / len(posterior)
                    - sum(prior) / len(prior)) >= 0.02:
                qualifying.append(name)
    out["calibration_gate"] = {
        "status": ("problem_worth_fresh_candidate_test" if len(qualifying) >= 2
                   else "mixed_or_insufficient_evidence"),
        "qualifying_target_count": len(qualifying),
        "qualifying_targets": qualifying,
        "criterion": "posterior per-game Brier exceeds prior by >=0.02 in at least two targets, each with >=4 scored games",
        "promotion_authorized": False,
    }
    return out


def run_audit(output, *, seed=2026122301, max_decisions=256, wall_seconds=60,
              smoke=False, progress=None):
    target_path = Path(output)
    if target_path.exists():
        raise FileExistsError(target_path)
    if type(seed) is not int or type(max_decisions) is not int or max_decisions <= 0:
        raise ValueError("seed and positive max_decisions required")
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError("wall_seconds must be finite and positive")
    if smoke:
        streams = [(name, seat, seed + i) for i, (name, seat) in
                   enumerate((("math", 0), ("honest", 1)))]
        deals_each = 1
    else:
        streams = [(name, seat, seed + i * 100) for i, (name, seat) in
                   enumerate(( (t, s) for t in TARGETS for s in (0, 1)))]
        deals_each = 4
    source_start = _source_hashes()
    start = time.monotonic()
    games = []
    stopped = False
    for stream_index, (name, seat, base) in enumerate(streams):
        policy = traced_policy()(seed=base)
        factory = {"math": RoundMathBot, "honest": RoundHonestBot,
                   "random": RoundRandomBot}.get(name)
        opponent = (ScriptedRoundOpponent(SCRIPTED_PATTERN, seed=base + 1)
                    if factory is None else factory(seed=base + 1))
        for deal in range(deals_each):
            if time.monotonic() - start >= wall_seconds:
                stopped = True
                break
            game_seed = (seed + stream_index if smoke else base + 10 + deal)
            recycle_seed = (seed + 100 + stream_index if smoke
                            else base + 50 + deal)
            try:
                row = _game_trace(policy, opponent, seat=seat,
                    deal_seed=game_seed, recycle_seed=recycle_seed,
                    starting_player=deal % 2, max_decisions=max_decisions)
            except Exception:
                # Fail closed: no partial artifact on integrity failure.
                raise
            games.append({"target": name, "seat": seat, **row})
            if progress:
                progress(len(games))
        if stopped:
            break
    elapsed = time.monotonic() - start
    source_end = _source_hashes()
    status = ("invalid_source_drift" if source_start != source_end else
              "budget_stop" if stopped else "completed")
    payload = {"format": "bluff-predecision-bluff-audit-v1",
        "ruleset_id": RULESET_ID, "synthetic_only": True, "smoke": smoke,
        "status": status,
        "inputs": {"seed": seed, "targets": [g[0] for g in streams],
                   "seats": [g[1] for g in streams], "deals_per_stream": deals_each,
                   "max_decisions_per_game": max_decisions,
                   "wall_seconds": wall_seconds, "starting_player": "deal index mod 2",
                   "scripted_call_probability": 0.35,
                   "scripted_bluff_probability": 0.60,
                   "forecasts": list(FORECASTS)},
        "source_sha256_start": source_start, "source_sha256_end": source_end,
        "elapsed_seconds": elapsed, "game_count": len(games),
        "completed_games": sum(g["termination"] == "win" for g in games),
        "truncated_games": sum(g["termination"] == "truncated" for g in games),
        "summary": _aggregate(games),
        "limitations": ["dependent synthetic decisions; descriptive metrics only",
            "no human validation or policy promotion", "forced terminal reviews excluded",
            "target/seat groups are small and selected responses are endogenous"]}
    # Exclusive creation with explicit restrictive mode; avoid exposing rows.
    fd = os.open(target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if stat.S_IMODE(target_path.stat().st_mode) != 0o600:
        raise PermissionError("audit artifact mode is not 0600")
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026122301)
    parser.add_argument("--max-decisions", type=int, default=256)
    parser.add_argument("--wall-seconds", type=float, default=60)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.output, seed=args.seed, max_decisions=args.max_decisions,
                       wall_seconds=args.wall_seconds, smoke=args.smoke)
    print(json.dumps({k: result[k] for k in ("status", "game_count", "elapsed_seconds")},
                     sort_keys=True))
