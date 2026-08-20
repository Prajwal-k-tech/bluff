# Game Rules (Locked)

> **Last updated:** August 10, 2026
> **Status:** FINAL — Do not change without updating this document and all dependent code.

---

## Table of Contents

1. [Rules Summary](#1-rules-summary)
2. [Passing Mechanics](#2-passing-mechanics)
3. [Challenge Mechanics](#3-challenge-mechanics)
4. [Turn Flow](#4-turn-flow)
5. [Information Model](#5-information-model)
6. [Experimentable Parameters](#6-experimentable-parameters)

---

## 1. Rules Summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| Players | 2 (human vs bot) | Asymmetric information |
| Decks | 1 standard 52-card deck | 4 suits × 13 ranks |
| Cards per player | 14 | Remaining 24 form the draw pile |
| First player | Random (50/50 coin flip) | Determined at game start |
| First rank | First player claims Aces | Subsequent turns are free |
| Rank progression | **FREE** — claim any rank each turn | Not sequential, not ±1 |
| Cards per turn | 1–4 cards, all claimed to be the same rank | Cannot mix ranks in a single play |
| Passing | Draw 1 card from the draw pile | Cannot pass if draw pile is empty |
| Challenge | Call "Bluff!" on the **most recent play only** | Cannot challenge earlier plays |
| Bluff caught | Bluffer takes the **entire pile** | All accumulated cards go to bluffer's hand |
| Wrong call | Caller takes the **entire pile** | Caller penalized for incorrect challenge |
| Win condition | First player to empty their hand wins | Game ends immediately |
| Round limit | 100 turns (game is a draw if no winner) | Prevents infinite games |

### Rank Values

| Rank | Value | Display |
|------|-------|---------|
| Two | 2 | 2 |
| Three | 3 | 3 |
| Four | 4 | 4 |
| Five | 5 | 5 |
| Six | 6 | 6 |
| Seven | 7 | 7 |
| Eight | 8 | 8 |
| Nine | 9 | 9 |
| Ten | 10 | 10 |
| Jack | 11 | J |
| Queen | 12 | Q |
| King | 13 | K |
| Ace | 14 | A |

---

## 2. Passing Mechanics

After an opponent plays cards, the non-playing player has two options:

1. **Call Bluff** — Challenge the most recent play
2. **Pass** — Draw 1 card from the draw pile and continue

### Rules

- **Passing** means drawing exactly 1 card from the top of the draw pile
- The drawn card is added to the player's hand (sorted)
- The game then proceeds to the next player's turn
- The pile remains in the center (cards are not discarded on pass)

### Draw Pile Constraints

| Draw Pile State | Can Pass? | What Happens |
|-----------------|-----------|--------------|
| Has cards | ✅ Yes | Draw 1 card, continue |
| Empty | ❌ No | **Cannot pass** — must play cards or call bluff |

When the draw pile is empty, players are **forced** to act:
- On their turn: must play 1–4 cards
- After opponent plays: must call bluff (cannot pass)

This creates endgame tension — once the draw pile is exhausted, every decision carries weight.

---

## 3. Challenge Mechanics

A bluff challenge can only be made on the **most recent play** (the immediately preceding action).

### Challenge Flow

1. **Opponent plays cards** face-down, claiming they are rank X
2. **You decide**: Call Bluff or Pass
3. **If you call bluff**:
   - Cards are revealed
   - If opponent **was bluffing** (cards ≠ claimed rank): Opponent takes entire pile
   - If opponent **was honest** (cards = claimed rank): You take entire pile
4. **If you pass**: Play continues to next turn

### Challenge Rules

| Rule | Detail |
|------|--------|
| Timing | Can only challenge the most recent play |
| Frequency | Each play can only be challenged once |
| Cannot challenge yourself | You cannot call bluff on your own play |
| Cannot challenge nothing | If no actions exist, cannot call bluff |

### After a Challenge

Regardless of outcome, the **pile is cleared** (cards go to the loser's hand) and the game proceeds to the next turn. The player who took the pile is now at a card disadvantage.

---

## 4. Turn Flow

```
Player's turn:
├── Option A: Play 1-4 cards (claim rank)
│   ├── Select cards from hand
│   ├── Claim a rank (must match claimed cards)
│   ├── Cards go face-down to pile
│   └── Opponent can call bluff or pass
│       ├── Call bluff → reveal cards, resolve
│       └── Pass → next player's turn
├── Option B: Pass (draw 1 card)
│   ├── Draw pile must have cards
│   └── Draw 1 card → next player's turn
└── Forced: If draw pile empty, must play or call
    ├── On your turn: must play cards
    └── After opponent plays: must call bluff
```

### Detailed Turn Sequence

```
START OF TURN
│
├─ Is it your turn to PLAY?
│   ├─ YES → Choose cards + claim rank → Play → Opponent decides
│   │         └─ Opponent: Call Bluff / Pass
│   │
│   └─ NO → (Opponent just played)
│       ├─ Can you call bluff? → Yes (if not already called)
│       ├─ Can you pass? → Yes (if draw pile has cards)
│       └─ Forced? → If draw pile empty, MUST call bluff
│
├─ RESOLVE ACTION
│   ├─ Play: Cards added to pile, turn passes
│   ├─ Pass: Draw 1 card, turn passes
│   └─ Call: Reveal cards, loser takes pile, turn passes
│
├─ CHECK WIN
│   ├─ Player's hand empty? → GAME OVER, player wins
│   └─ Turn count ≥ 100? → GAME OVER, draw
│
└─ NEXT TURN
    └─ Switch to other player
```

---

## 5. Information Model

### What the Bot Knows

| Information | Available | Notes |
|-------------|-----------|-------|
| Its own hand | ✅ Full knowledge | Exact cards held |
| Opponent's hand size | ✅ Observable | Number of cards only |
| Pile size | ✅ Observable | Total cards in center |
| Draw pile size | ✅ Observable | Remaining undrawn cards |
| Full game history | ✅ All actions | Every play, claim, bluff, call |
| Cards revealed on challenge | ✅ After bluff call | Full information for that action |

### What the Bot Does NOT Know

| Information | Available | Notes |
|-------------|-----------|-------|
| Opponent's actual hand | ❌ Hidden | Cards are never revealed unless bluff is called |
| Order of cards in draw pile | ❌ Unknown | Only that cards are drawn from top |

### Partial Observability

- When the opponent plays cards face-down, the bot sees **the claim** but not the **actual cards**
- If a bluff is **called**, the cards are revealed (full information for that action)
- If a bluff is **not called**, the cards remain hidden and only the claim is recorded in history
- The bot can reconstruct what cards remain in the game using card counting (tracking all played cards)

### Card Counting Feasibility

In 2-player Bluff, the bot knows:
- Its own hand (exact)
- All cards it has played (exact)
- All cards revealed via bluff calls (exact)

This means the bot can compute exact probabilities for what the opponent might hold, using hypergeometric distributions.

---

## 6. Experimentable Parameters

These parameters can be varied for research experiments. Changes require updating this document and all dependent code.

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `cards_per_player` | 14 | 10, 14, 20, 26 | Cards dealt to each player at game start |
| `num_decks` | 1 | 1, 2 | Number of standard 52-card decks used |
| `pass_cost` | 1 | 1, 2, 3 | Number of cards drawn when passing |
| `max_cards_per_turn` | 4 | 1, 2, 3, 4 | Maximum cards playable per turn |
| `rank_mode` | free | free, sequential, plus_minus_1 | How rank claims are constrained |
| `round_limit` | 100 | 50, 100, 200 | Maximum turns before draw is declared |

### Parameter Details

#### `cards_per_player`
Controls initial hand size. Lower values create faster, more aggressive games. Higher values create longer strategic games.

| Value | Draw Pile | Game Character |
|-------|-----------|----------------|
| 10 | 32 cards | Fast, high-stakes bluffs |
| 14 | 24 cards | Balanced (default) |
| 20 | 12 cards | Slower, more card counting |
| 26 | 0 cards | No draw pile — pure hand management |

#### `num_decks`
More decks reduce card counting effectiveness by increasing the pool of unknown cards.

| Value | Total Cards | Card Counting Accuracy |
|-------|-------------|------------------------|
| 1 | 52 | High — exact probabilities |
| 2 | 104 | Lower — more uncertainty |

#### `pass_cost`
Higher pass costs discourage passing and force more confrontations.

| Value | Effect |
|-------|--------|
| 1 | Standard — passing is cheap |
| 2 | Moderate — passing adds meaningful risk |
| 3 | Aggressive — passing is costly, forces plays |

#### `max_cards_per_turn`
Limits how many cards can be played at once. Lower values make bluffs harder to hide.

| Value | Effect |
|-------|--------|
| 1 | Only 1 card per play — maximum transparency |
| 2 | Up to 2 cards — moderate flexibility |
| 3 | Up to 3 cards — good flexibility |
| 4 | Up to 4 cards — full flexibility (default) |

#### `rank_mode`
Controls how players can choose which rank to claim.

| Value | Rule |
|-------|------|
| free | Claim any rank, any turn (default) |
| sequential | Must claim ranks in order: A→2→3→...→K→A |
| plus_minus_1 | Can only claim rank ±1 from previous claim |

#### `round_limit`
Prevents infinite games. Higher limits allow more comeback potential.

| Value | Effect |
|-------|--------|
| 50 | Fast games, draws more common |
| 100 | Standard — balanced (default) |
| 200 | Long games, fewer draws |

---

*This document is LOCKED. Any rule changes must be approved, documented here, and reflected in all dependent code (`game.py`, `bot.py`, tests).*
