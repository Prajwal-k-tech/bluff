"""Verification battery for the Bluff engine and pool-model bots.

Committed per Oracle Gate-1 finding F2 (deepwork supergoal-push, 2026-09-10):
the original probe lived in /tmp and was not replicable. This directory is the
committed, citable version of the verification battery that produced the
findings posted to AGENT_CHAT.md 15:35 (2026-09-10):

  1. Engine soundness: 52-card conservation invariant after every mutation.
  2. "0% vs Honest" confirmed at 200 games/matchup (seat-swapped, seed 42).
  3. L-trace: Honest's bluff_probability() inputs instrumented — meanL 0.267,
     P(L>0.6)=0% vs dump-style play (pre-v2-fix behavior).
  4. Structural draw-lock: mutual passing exhausts the 24-card draw pile
     ~turn 24 -> forced-call pile recycling -> frozen ~25/25 hands -> draws.
     Honest never loses to honest play (0 losses / 600+ games).
  5. Naive bluff-dumps self-destruct via the impossible-claim corner.

Usage:
    python -m verification.probe     # matchup battery + conservation checks
    python -m verification.ltrace    # L-value instrumentation

NOTE: numbers in the 15:35 post were produced pre-bluff_probability-v2-fix
(15:05). Re-running now measures the POST-fix bots — expect different call
rates (Honest 0% voluntary calls) but the same draw-lock structure.
"""
