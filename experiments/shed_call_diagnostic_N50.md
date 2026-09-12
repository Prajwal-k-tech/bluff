# Shed/Call Diagnostic Results (N=50/matchup, seats swapped, seed 7)

Diagnostic — not citable at N=50. Method: `experiments/shed_call_diagnostic.py`
(drives games through `game.py`; per-bot shed/call/wrong-call/pile-take tracking).

`shed/turn` = cards shed per own play turn. `call%` = calls / responses.
`wrong%` = calls where the caller took the pile. `took/game` = avg cards
absorbed via pile takes per game.

## Hybrid (A) vs Honest (B) — A 6-0-44, avg turns 91
- A: shed/turn 3.42 | call 73% (1658) | wrong 100% | took/game 148.1
- B: shed/turn 3.35 | call 75% (1690) | wrong 99% | took/game 141.2

## Hybrid (A) vs Bayesian (B) — A 3-0-47, avg turns 99
- A: shed/turn 3.43 | call 77% (1917) | wrong 88% | took/game 156.2
- B: shed/turn 3.39 | call 75% (1856) | wrong 99% | took/game 166.7

## PureNN-final (A) vs Honest (B) — A 0-4-46, avg turns 95
- A: shed/turn 3.61 | call 92% (2194) | wrong 100% | took/game 198.0
- B: shed/turn 2.88 | call 58% (1363) | wrong 99% | took/game 108.1

## PureNN-final (A) vs Bayesian (B) — A 0-7-43, avg turns 91
- A: shed/turn 2.61 | call 82% (1885) | wrong 87% | took/game 128.6
- B: shed/turn 2.53 | call 67% (1537) | wrong 98% | took/game 91.9

## PureNN-v61 (A) vs Honest (B) — A 0-8-42, avg turns 90
- A: shed/turn 2.36 | call 100% (2244) | wrong 100% | took/game 138.5
- B: shed/turn 2.19 | call 49% (1092) | wrong 100% | took/game 65.5

## PureNN-v61 (A) vs Bayesian (B) — A 0-7-43, avg turns 94
- A: shed/turn 2.00 | call 100% (2350) | wrong 87% | took/game 118.9
- B: shed/turn 2.09 | call 50% (1179) | wrong 93% | took/game 53.2

## Bayesian (A) vs Honest (B) — A 0-40-10, avg turns 67
- A: shed/turn 3.70 | call 68% (1175) | wrong 100% | took/game 134.7
- B: shed/turn 2.75 | call 59% (991) | wrong 81% | took/game 68.6

## CardCount (A) vs Honest (B) — A 0-0-50, avg turns 100
- A: shed/turn 3.73 | call 78% (1953) | wrong 100% | took/game 195.1
- B: shed/turn 3.36 | call 74% (1847) | wrong 100% | took/game 159.0

## CardCount (A) vs Bayesian (B) — A 24-0-26, avg turns 83
- A: shed/turn 3.40 | call 70% (1472) | wrong 86% | took/game 122.7
- B: shed/turn 3.48 | call 71% (1487) | wrong 100% | took/game 150.6

## Reading
- Universal over-calling: every bot wrong-calls at 81–100%. The v61 recipe
  did NOT fix the respond head (100% calls — worse than final.pt's 82–92%).
- BayesianBot loses 0–40 to Honest: its posterior fails to suppress calling
  vs a 0%-bluffer within a game (T6a diagnosis target).
- Hybrid wins only by out-shedding (3.42–3.43/turn, fastest) while absorbing
  piles at the same rate as everyone else.
- Only healthy beating: CardCount 24-0 over Bayesian.
