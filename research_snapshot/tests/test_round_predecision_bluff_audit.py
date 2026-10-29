import json
import random
import stat

import pytest

from analysis.round_predecision_bluff_audit import (
    _aggregate, _game_trace, _metric, run_audit, traced_policy,
)
from bots.round_policy import RoundHonestBot, RoundMathBot
from cards import Card, Rank, Suit
from game import Action


def test_truthful_finish_still_records_predecision_forecast():
    last = Action(1, [None], Rank.TWO, None, False, False, 0)
    last.action_index = 0
    view = {
        "viewer": 0, "player_id": 0, "current_player": 0,
        "active_rank": Rank.TWO, "opponent_hand_size": 4,
        "pile_size": 1, "draw_pile_size": 20,
        "actions": [last], "last_action": last, "own_pile_cards": [],
        "can_play": True, "can_call_bluff": True, "can_pass": True,
    }
    policy = traced_policy()(seed=6)
    choice = policy.choose_action([Card(Rank.TWO, Suit.HEARTS)], view)
    assert choice["action"] == "play"
    assert len(policy.forecast_rows) == 1
    assert 0 <= policy.forecast_rows[0]["predictions"]["posterior"] <= 1


def test_predecision_trace_joins_truth_afterward_and_preserves_rng():
    policy = traced_policy()(seed=6)
    opponent = RoundHonestBot(7)
    before = random.getstate()
    row = _game_trace(policy, opponent, seat=0, deal_seed=2026122401,
                      recycle_seed=2026122501, starting_player=0,
                      max_decisions=256)
    assert random.getstate() == before
    assert row["termination"] in ("win", "truncated")
    assert (row['winner'] in (0, 1) and row['focal_outcome'] ==
            ('win' if row['winner'] == 0 else 'loss')) if row['termination'] == 'win' else (
                row['winner'] is None and row['focal_outcome'] is None)
    assert all(set(item["predictions"]) == {"posterior", "prior", "certificate"}
               for item in row["rows"])
    assert all(type(item["truth"]) is bool for item in row["rows"])
    assert all(0 <= value <= 1 for item in row["rows"]
               for value in item["predictions"].values())
    assert all(not hasattr(item, "action_index") for item in row["rows"])


def test_truncated_trace_has_no_winner_or_outcome():
    row = _game_trace(traced_policy()(6), RoundHonestBot(7), seat=0,
                      deal_seed=2026122401, recycle_seed=2026122501,
                      starting_player=0, max_decisions=1)
    assert row['termination'] == 'truncated'
    assert row['winner'] is None and row['focal_outcome'] is None


def test_aggregate_emits_selected_and_nonselected_only():
    games = [{"target": "math", "seat": 0, "termination": "win", "decisions": 7,
        "rows": [{"truth": True, "selected": True,
                  "predictions": {"posterior": .4, "prior": .3, "certificate": 1.}},
                 {"truth": False, "selected": False,
                  "predictions": {"posterior": .2, "prior": .3, "certificate": .3}}]}]
    result = _aggregate(games)
    assert set(result) == {"math/seat_0", "calibration_gate"}
    assert result["math/seat_0"]["selection"]["selected"]["posterior"]["support"] == 1
    assert result["math/seat_0"]["selection"]["nonselected"]["prior"]["support"] == 1
    assert "rows" not in json.dumps(result)
    assert _metric([], "posterior")["nll"] is None


def test_smoke_artifact_is_aggregate_only_exclusive_and_private(tmp_path):
    output = tmp_path / "audit.json"
    payload = run_audit(output, seed=2026122401, smoke=True,
                        max_decisions=256, wall_seconds=60)
    artifact = json.loads(output.read_text())
    assert payload["status"] == "completed"
    assert artifact["game_count"] == 2
    assert {key.split("/")[0] for key in artifact["summary"]
            if "/seat_" in key} == {"math", "honest"}
    raw = output.read_text()
    assert '"rows"' not in raw and "truth_bluff" not in raw
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        run_audit(output, smoke=True)


def test_trace_rejects_bad_seat():
    with pytest.raises(ValueError):
        _game_trace(traced_policy()(1), RoundMathBot(2), seat=2,
                    deal_seed=1, recycle_seed=2, starting_player=0)
    with pytest.raises(ValueError, match="starting_player"):
        _game_trace(traced_policy()(1), RoundMathBot(2), seat=0,
                    deal_seed=1, recycle_seed=2, starting_player=2)


def test_given_fixture_honors_starter(monkeypatch):
    from analysis import round_predecision_bluff_audit as audit
    real = audit.play_game
    seen = []

    def checked_play_game(*args, **kwargs):
        seen.append(kwargs["game"].current_player)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit, "play_game", checked_play_game)
    for starter in (0, 1):
        audit._game_trace(traced_policy()(seed=6), RoundHonestBot(7), seat=0,
                          deal_seed=2026122401, recycle_seed=2026122501,
                          starting_player=starter, max_decisions=256)
    assert seen == [0, 1]
