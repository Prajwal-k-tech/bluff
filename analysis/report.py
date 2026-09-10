"""Bluff analytics report — paper-figure data from JSONL game logs.

Reads JSONL files produced by analysis.logger (test_bots.py --log, PPO
training --log) and computes the statistics the paper needs:

Per-bot metrics (aggregated over all games in the file):
    - plays, bluffs, bluff rate (bluffs / plays)          [Dewey 2025 / Ahle
    - bluff success rate (uncalled bluffs / bluffs)        2022 track these
    - calls, call accuracy (correct calls / calls)         over training]
    - passes

Bluff rate by hand size (the desperation curve):
    - buckets 1-2, 3-4, 5-7, 8-10, 11-14 cards
    - paper claim to test: humans/bots bluff more when few cards left
      (data-pipeline.md: humans bluff 35% at 1 card vs 15% at 10+)

Game outcomes per matchup (bot_mode_a vs bot_mode_b win/draw counts).

Usage:
    python -m analysis.report --input data/terminal/tournament.jsonl
    python -m analysis.report --input data/terminal/training_v2.jsonl --markdown
    python -m analysis.report --input a.jsonl b.jsonl   # merged view

Design notes:
- Stream-parses (constant memory) — training files reach millions of lines.
- Only 'action' records with was_bluff not-None count toward bluff stats;
  'pass' records count toward passes only.
- game_end records drive matchup outcome stats.
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

HAND_BUCKETS = [(1, 2), (3, 4), (5, 7), (8, 10), (11, 14), (15, 40)]


def _bucket(hand_size: int) -> str:
    for lo, hi in HAND_BUCKETS:
        if lo <= hand_size <= hi:
            return f"{lo}-{hi}"
    return "?"


class BotStats:
    """Running per-bot statistics."""

    def __init__(self):
        self.plays = 0
        self.bluffs = 0
        self.bluffs_uncalled = 0
        self.calls = 0
        self.calls_correct = 0
        self.passes = 0

    def observe(self, rec: dict) -> None:
        action_type = rec.get("action_type")
        if action_type == "pass":
            self.passes += 1
            return
        if action_type == "call":
            self.calls += 1
            if rec.get("caller_was_right"):
                self.calls_correct += 1
            return
        # play record
        self.plays += 1
        if rec.get("was_bluff"):
            self.bluffs += 1
            if not rec.get("bluff_called"):
                self.bluffs_uncalled += 1

    # -- derived ---------------------------------------------------------------

    @property
    def bluff_rate(self) -> float:
        return self.bluffs / self.plays if self.plays else 0.0

    @property
    def bluff_success(self) -> float:
        return self.bluffs_uncalled / self.bluffs if self.bluffs else 0.0

    @property
    def call_accuracy(self) -> float:
        return self.calls_correct / self.calls if self.calls else 0.0

    def as_row(self, name: str) -> List[str]:
        return [
            name,
            str(self.plays),
            f"{self.bluff_rate:.1%} ({self.bluffs}/{self.plays})",
            f"{self.bluff_success:.1%}",
            f"{self.call_accuracy:.1%} ({self.calls_correct}/{self.calls})",
            str(self.passes),
        ]


class Report:
    def __init__(self):
        self.bots: Dict[str, BotStats] = defaultdict(BotStats)
        self.bluff_by_hand: Dict[str, List[int]] = defaultdict(
            lambda: [0, 0])  # bucket -> [bluffs, plays]
        self.matchups: Dict[tuple, Dict[str, int]] = defaultdict(
            lambda: {"a": 0, "b": 0, "draw": 0})

    def feed(self, path: str) -> int:
        """Consume one JSONL file. Returns number of records read."""
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate truncated tail lines
                n += 1
                self._consume(rec)
        return n

    def _consume(self, rec: dict) -> None:
        kind = rec.get("record")
        if kind == "action":
            mode = rec.get("bot_mode") or "?"
            self.bots[mode].observe(rec)
            if rec.get("action_type") == "play" and rec.get("was_bluff") is not None:
                bucket = _bucket(int(rec.get("hand_size") or 0))
                self.bluff_by_hand[bucket][1] += 1
                if rec["was_bluff"]:
                    self.bluff_by_hand[bucket][0] += 1
        elif kind == "game_end":
            a = rec.get("bot_mode_a") or "?"
            b = rec.get("bot_mode_b") or "?"
            key = tuple(sorted((a, b)))
            winner = rec.get("winner_player")
            if winner is None or winner == -1:
                self.matchups[key]["draw"] += 1
            else:
                winner_name = rec.get("game_winner") or (
                    a if winner == 0 else b)
                if winner_name == a:
                    self.matchups[key]["a"] += 1
                else:
                    self.matchups[key]["b"] += 1

    # -- output ---------------------------------------------------------------

    def print_text(self, markdown: bool = False) -> None:
        out = sys.stdout

        if markdown:
            out.write("\n### Per-bot behavior\n\n")
            out.write("| Bot | Plays | Bluff rate | Bluff success | "
                      "Call accuracy | Passes |\n")
            out.write("|---|---|---|---|---|---|\n")
            for name in sorted(self.bots):
                s = self.bots[name]
                if s.plays or s.calls or s.passes:
                    cells = s.as_row(name)
                    out.write("| " + " | ".join(cells) + " |\n")
        else:
            out.write("\nPER-BOT BEHAVIOR\n")
            out.write(f"  {'Bot':<14} {'Plays':>7} {'BluffRate':>16} "
                      f"{'BluffOK':>8} {'CallAcc':>16} {'Passes':>7}\n")
            out.write("  " + "-" * 74 + "\n")
            for name in sorted(self.bots):
                s = self.bots[name]
                if s.plays or s.calls or s.passes:
                    r = s.as_row(name)
                    out.write(f"  {r[0]:<14} {r[1]:>7} {r[2]:>16} "
                              f"{r[3]:>8} {r[4]:>16} {r[5]:>7}\n")

        if markdown:
            out.write("\n### Bluff rate by hand size\n\n")
            out.write("| Hand size | Plays | Bluffs | Bluff rate |\n"
                      "|---|---|---|---|\n")
            for (lo, _hi) in HAND_BUCKETS:
                bucket = f"{lo}-{_hi}"
                bluffs, plays = self.bluff_by_hand.get(bucket, [0, 0])
                if plays:
                    rate = bluffs / plays
                    out.write(f"| {bucket} | {plays} | {bluffs} "
                              f"| {rate:.1%} |\n")
        else:
            out.write("\nBLUFF RATE BY HAND SIZE\n")
            out.write(f"  {'Hand':>8} {'Plays':>8} {'Bluffs':>8} {'Rate':>7}\n")
            out.write("  " + "-" * 35 + "\n")
            for (lo, hi) in HAND_BUCKETS:
                bucket = f"{lo}-{hi}"
                bluffs, plays = self.bluff_by_hand.get(bucket, [0, 0])
                if plays:
                    out.write(f"  {bucket:>8} {plays:>8} {bluffs:>8} "
                              f"{bluffs / plays:>6.1%}\n")

        if self.matchups:
            if markdown:
                out.write("\n### Matchup outcomes\n\n")
                out.write("| Matchup | A wins | B wins | Draws |\n"
                          "|---|---|---|---|\n")
                for (a, b) in sorted(self.matchups):
                    m = self.matchups[(a, b)]
                    out.write(f"| {a} vs {b} | {m['a']} | {m['b']} "
                              f"| {m['draw']} |\n")
            else:
                out.write("\nMATCHUP OUTCOMES\n")
                for (a, b) in sorted(self.matchups):
                    m = self.matchups[(a, b)]
                    out.write(f"  {a} vs {b}: {m['a']}-{m['b']}"
                              f"-{m['draw']}\n")
        out.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bluff analytics report from JSONL game logs")
    parser.add_argument("--input", nargs="+", required=True,
                        help="JSONL file(s) to analyze")
    parser.add_argument("--markdown", action="store_true",
                        help="Emit markdown tables (for docs)")
    args = parser.parse_args()

    report = Report()
    for path in args.input:
        n = report.feed(path)
        print(f"read {n:>8} records from {path}", file=sys.stderr)

    report.print_text(markdown=args.markdown)


if __name__ == "__main__":
    main()
