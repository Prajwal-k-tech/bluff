import json
import pytest
from analysis.report import BotStats, Report


def test_bot_stats_metrics():
    stats = BotStats()
    stats.observe({"action_type": "play", "was_bluff": False, "bluff_called": False})
    stats.observe({"action_type": "play", "was_bluff": True, "bluff_called": False})  # uncalled bluff
    stats.observe({"action_type": "play", "was_bluff": True, "bluff_called": True})   # called bluff
    stats.observe({"action_type": "pass"})
    stats.observe({"action_type": "call", "caller_was_right": True})
    stats.observe({"action_type": "call", "caller_was_right": False})

    assert stats.plays == 3
    assert stats.bluffs == 2
    assert stats.bluffs_uncalled == 1
    assert stats.passes == 1
    assert stats.calls == 2
    assert stats.calls_correct == 1
    assert abs(stats.bluff_rate - (2 / 3)) < 1e-4
    assert abs(stats.bluff_success - 0.5) < 1e-4
    assert abs(stats.call_accuracy - 0.5) < 1e-4
    assert abs(stats.a_call - 0.5) < 1e-4


def test_report_s_lock(tmp_path):
    log_file = tmp_path / "test_game.jsonl"
    records = [
        {"record": "action", "bot_mode": "Hybrid", "action_type": "play", "was_bluff": False, "hand_size": 12, "turn_number": 24, "game_id": "g1"},
        {"record": "action", "bot_mode": "Honest", "action_type": "play", "was_bluff": False, "hand_size": 18, "turn_number": 24, "game_id": "g1"},
        {"record": "action", "bot_mode": "Hybrid", "action_type": "pass", "hand_size": 11, "turn_number": 25, "game_id": "g1"},
        {"record": "action", "bot_mode": "Hybrid", "action_type": "play", "was_bluff": False, "hand_size": 10, "turn_number": 24, "game_id": "g2"},
        {"record": "game_end", "bot_mode_a": "Hybrid", "bot_mode_b": "Honest", "winner_player": 0, "game_winner": "Hybrid"}
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    report = Report()
    n = report.feed(str(log_file))
    assert n == 5
    assert report.bots["Hybrid"].s_lock == (12 + 10) / 2  # 11.0
    assert report.bots["Honest"].s_lock == 18.0
    assert "Hybrid" in report.ratings
    assert "Honest" in report.ratings
    assert report.ratings["Hybrid"] > 1500.0
    assert report.ratings["Honest"] < 1500.0
