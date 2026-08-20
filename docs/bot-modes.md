# Bot Modes

> **Last updated:** August 10, 2026
> **Scope:** All bot implementations, interfaces, decision logic, and benchmarking.

---

## Table of Contents

1. [BotInterface](#1-botinterface)
2. [StateEncoder](#2-stateencoder)
3. [Bot Implementations](#3-bot-implementations)
   - [RandomBot](#31-randombot)
   - [HonestBot](#32-honestbot)
   - [CardCountBot](#33-cardcountbot)
   - [AdaptiveBot](#34-adaptivebot)
   - [PureNNBot](#35-purennbot)
   - [HybridBot](#36-hybridbot)
4. [Win Rate Table](#4-win-rate-table)
5. [Bot Selection UX](#5-bot-selection-ux)
6. [Benchmarking Protocol](#6-benchmarking-protocol)

---

All bots implement `BotInterface` (see [architecture.md](architecture.md)). Human player chooses which bot to play against in the lobby.

---

## 1. BotInterface

Every bot must implement this abstract base class:

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from cards import Card, Rank
from game import Action, GameState

class BotInterface(ABC):
    @abstractmethod
    def decide_play(self, hand: List[Card], game_state: dict) -> Tuple[List[Card], Rank]:
        """Return (cards_to_play, claimed_rank). Must play 1-4 cards of same rank."""
        pass

    @abstractmethod
    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        """Return True to call bluff, False to pass."""
        pass

    @abstractmethod
    def observe_action(self, action: Action, opponent_hand_size: int):
        """Observe any action (own or opponent's) for learning."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Persist learned state (opponent model, etc.)."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load learned state."""
        pass
```

### `game_state` Dictionary

```python
game_state = {
    "opponent_hand_size": int,      # Cards in opponent's hand
    "pile_size": int,               # Cards in center pile
    "draw_pile_size": int,          # Cards left to draw
    "turn_number": int,             # Current turn (0-indexed)
    "current_rank": Rank,           # Rank claimed this round (or None)
    "last_action": Action,          # Most recent action (for bluff calling)
    "cards_remaining": dict,        # {Rank: count} of unseen cards
    "hand": List[Card],             # Bot's own hand
    "history": List[Action],        # Full action history this game
}
```

---

## 2. StateEncoder

Converts game state to a flat tensor for neural network bots:

```python
import torch
from typing import List, Dict
from cards import Card, Rank
from game import Action

class StateEncoder:
    """Encodes game state into a fixed-size feature vector for NN bots."""

    def encode(self, hand: List[Card], game_state: dict) -> torch.Tensor:
        """
        Returns a ~50-dimensional feature vector:
        - Hand frequency vector (13 values)
        - Game context (5 values)
        - Cards remaining per rank (13 values)
        - History features (8 values)
        """
        features = []

        # Hand frequency vector (13 values)
        # Count of each rank in bot's hand
        hand_counts = {rank: 0 for rank in Rank}
        for card in hand:
            hand_counts[card.rank] += 1
        for rank in Rank:
            features.append(hand_counts[rank] / 14.0)  # Normalize by max hand size

        # Game context (5 values)
        features.append(game_state["opponent_hand_size"] / 26.0)
        features.append(game_state["pile_size"] / 52.0)
        features.append(game_state["draw_pile_size"] / 24.0)
        features.append(game_state["turn_number"] / 100.0)
        features.append(1.0 if game_state.get("current_rank") else 0.0)

        # Cards remaining per rank (13 values)
        # Normalized by 4 (max per rank in deck)
        for rank in Rank:
            remaining = game_state["cards_remaining"].get(rank, 4)
            features.append(remaining / 4.0)

        # History features (8 values)
        history = game_state.get("history", [])
        if history:
            recent = history[-10:]  # Last 10 actions
            bluffs = sum(1 for a in recent if a.was_bluff)
            calls = sum(1 for a in recent if a.bluff_called)
            correct_calls = sum(1 for a in recent if a.caller_was_right)

            features.append(bluffs / max(1, len(recent)))
            features.append(calls / max(1, len(recent)))
            features.append(correct_calls / max(1, max(1, calls)))
            features.append(len(recent) / 10.0)

            # Average claim size in recent history
            avg_claim = sum(len(a.cards_played) for a in recent) / max(1, len(recent))
            features.append(avg_claim / 4.0)

            # Bluff rate by hand size (if available)
            features.append(0.5)  # Placeholder for model weight
            features.append(0.0)  # Placeholder for desperation
            features.append(0.0)  # Placeholder for aggression
        else:
            features.extend([0.0] * 8)

        return torch.tensor(features, dtype=torch.float32)
```

---

## 3. Bot Implementations

### 3.1 RandomBot

| Property | Value |
|----------|-------|
| **Difficulty** | Trivial |
| **File** | `bots/random_bot.py` |
| **Intelligence** | None — pure random |

**What it does:**

Uniform random legal moves. Plays random cards, claims random ranks, calls bluff with 50% probability. Zero strategy, pure baseline for measuring other bots against.

**Decision Logic:**

```
decide_play(hand, game_state):
    claim_size = random(1, min(4, hand.size()))
    cards = random_sample(hand, claim_size)
    rank = random(Rank)
    return (cards, rank)

decide_call(last_action, game_state):
    return random(0, 1) > 0.5  # 50% chance

observe_action(action, opponent_hand_size):
    pass  # Does not learn

save(path):
    pass  # Nothing to save

load(path):
    pass  # Nothing to load
```

**Strengths:**
- Unpredictable — no patterns to exploit
- Fast — no computation needed

**Weaknesses:**
- No strategy — plays cards it doesn't have, wastes good hands
- Never adapts — same behavior regardless of game state
- Calls bluff randomly — misses obvious bluffs, calls honest plays

**When to use it:**
- Baseline for measuring other bot performance
- Testing game engine correctness
- Absolute beginners who want to win easily

---

### 3.2 HonestBot

| Property | Value |
|----------|-------|
| **Difficulty** | Predictable |
| **File** | `bots/honest_bot.py` |
| **Intelligence** | Card counting only — never bluffs |

**What it does:**

Never bluffs — only plays cards it actually holds. Calls bluff based on card counting (hypergeometric probability). Predictable — opponents learn to never call bluff, giving HonestBot free passes.

**Decision Logic:**

```
decide_play(hand, game_state):
    # Always play honest — only cards we actually have
    ranks_in_hand = count_ranks(hand)

    # Play the rank we have most of
    best_rank = max(ranks_in_hand, key=ranks_in_hand.get)
    claim_size = min(ranks_in_hand[best_rank], 4)
    cards = take_cards_of_rank(hand, best_rank, claim_size)
    return (cards, best_rank)

decide_call(last_action, game_state):
    # Use card counting to estimate bluff probability
    p_bluff = card_counter.bluff_probability(
        last_action.claimed_rank,
        len(last_action.cards_played)
    )
    return p_bluff > 0.6  # Conservative threshold

observe_action(action, opponent_hand_size):
    card_counter.record_play(action.cards_played)

save(path):
    pass

load(path):
    pass
```

**Strengths:**
- Perfect card counting — knows exact probabilities
- Never caught bluffing — no penalty for dishonest play
- Consistent — reliable baseline

**Weaknesses:**
- **Extremely predictable** — opponents learn to never call bluff
- Gives free passes — opponent can bluff without fear
- No deception — pure defensive play

**When to use it:**
- Special challenge mode — "Can you beat a bot that never bluffs?"
- Studying optimal defensive play
- Testing opponent model adaptation

---

### 3.4 CardCountBot

| Property | Value |
|----------|-------|
| **Difficulty** | Moderate |
| **File** | `bots/card_count_bot.py` |
| **Intelligence** | Hypergeometric P(bluff) — no opponent model |

**What it does:**

Uses exact hypergeometric probability to estimate bluff likelihood. No opponent modeling — assumes all plays are independent. Good baseline for "rational but non-adaptive" play.

**Decision Logic:**

```
decide_play(hand, game_state):
    ranks_in_hand = count_ranks(hand)
    best_score = -1
    best_play = None

    for rank in Rank:
        have_count = ranks_in_hand.get(rank, 0)
        max_claim = min(4, hand.size())

        for claim_size in range(1, max_claim + 1):
            p_honest = counter.probability_of_rank(rank, claim_size)

            if claim_size <= have_count:
                # Honest play — safe
                score = 0.3 + (have_count / 14.0) * 0.2
                if score > best_score:
                    cards = take_cards_of_rank(hand, rank, claim_size)
                    best_play = (cards, rank)
                    best_score = score
            else:
                # Bluff — score by probability of getting away
                p_get_away = p_honest * 0.7  # Assume 70% pass rate
                if claim_size >= 3:
                    p_get_away *= 0.8
                if p_get_away > best_score and p_get_away > 0.5:
                    cards = random_sample(hand, claim_size)
                    best_play = (cards, rank)
                    best_score = p_get_away

    return best_play if best_play else fallback_play(hand)

decide_call(last_action, game_state):
    p_bluff = counter.bluff_probability(
        last_action.claimed_rank,
        len(last_action.cards_played)
    )
    # Adjust for pile size — bigger pile = more valuable call
    pile_bonus = min(0.15, game_state["pile_size"] * 0.01)
    return p_bluff > (0.5 - pile_bonus)

observe_action(action, opponent_hand_size):
    counter.record_play(action.cards_played)

save(path):
    pass  # No persistent state

load(path):
    pass
```

**Strengths:**
- Exact probability calculations
- Rational decisions based on math
- Good baseline for "optimal non-adaptive" play

**Weaknesses:**
- No opponent modeling — assumes random play
- Doesn't adapt to opponent patterns
- Same strategy regardless of who it's playing

**When to use it:**
- Baseline for measuring adaptive bots
- Studying optimal play without learning
- Comparing against mathematical optimum

---

### 3.5 AdaptiveBot

| Property | Value |
|----------|-------|
| **Difficulty** | Strong |
| **File** | `bots/adaptive_bot.py` |
| **Intelligence** | Bayesian opponent model + card counting + decision fusion |

**What it does:**

The main bot. Combines card counting with a Bayesian opponent model that tracks bluff patterns across games. Learns per-user patterns — gets smarter the more it plays against the same opponent. Uses decision fusion to weight card counting vs opponent model based on data availability.

**Decision Logic:**

```
decide_play(hand, game_state):
    call_freq = model.estimate_call_frequency()
    ranks_in_hand = count_ranks(hand)
    best_play = None
    best_score = -1

    for rank in Rank:
        have_count = ranks_in_hand.get(rank, 0)
        max_claim = min(4, hand.size())

        for claim_size in range(1, max_claim + 1):
            p_honest = counter.probability_of_rank(rank, claim_size)

            if claim_size <= have_count:
                # Honest play
                score = 0.3
                if score > best_score:
                    cards = take_cards_of_rank(hand, rank, claim_size)
                    best_play = (cards, rank)
                    best_score = score
            else:
                # Bluff — factor in opponent's calling frequency
                p_get_away = p_honest * (1.0 - call_freq)

                # Desperation bonus (few cards left)
                if hand.size() <= 5:
                    p_get_away *= 1.3

                # Multi-card penalty (suspicious)
                if claim_size >= 3:
                    p_get_away *= 0.8
                if claim_size == 4:
                    p_get_away *= 0.7

                if (p_get_away > best_score and p_get_away > bluff_threshold):
                    cards = random_sample(hand, claim_size)
                    best_play = (cards, rank)
                    best_score = p_get_away

    return best_play if best_play else fallback_play(hand)

decide_call(last_action, game_state):
    # Card counting signal
    p_bluff_counting = counter.bluff_probability(
        last_action.claimed_rank,
        len(last_action.cards_played)
    )

    # Opponent model signal
    features = {
        "hand_size": game_state["opponent_hand_size"],
        "claimed_rank": last_action.claimed_rank,
        "claim_size": len(last_action.cards_played),
    }
    p_bluff_model = model.estimate_bluff_probability(features)

    # Decision fusion — weight model more as we get more data
    model_weight = min(1.0, model.total_actions_observed / 40)
    counting_weight = 1.0 - model_weight

    p_bluff = (p_bluff_counting * counting_weight +
               p_bluff_model * model_weight)

    # Risk/reward adjustments
    pile_bonus = min(0.15, game_state["pile_size"] * 0.01)
    hand_bonus = 0.0
    if hand.size() <= 5:
        hand_bonus = 0.1
    if hand.size() <= 3:
        hand_bonus = 0.2

    threshold = call_threshold - pile_bonus - hand_bonus
    return p_bluff > threshold

observe_action(action, opponent_hand_size):
    model.observe_action(action, opponent_hand_size)
    counter.record_play(action.cards_played)

    # Adapt parameters based on opponent model
    adapt_parameters()

save(path):
    model.save(path + "/opponent_model.json")

load(path):
    model.load(path + "/opponent_model.json")
```

**Strengths:**
- **Adapts to opponent** — learns bluff patterns per user
- **Cross-game learning** — model persists across sessions
- **Decision fusion** — combines multiple signals intelligently
- **Parameter adaptation** — adjusts thresholds based on opponent behavior

**Weaknesses:**
- Needs data — starts with prior, improves with observations
- Can be exploited by changing patterns mid-game
- No deep strategic planning — reactive, not proactive

**When to use it:**
- Main opponent for human players
- Benchmarking adaptive vs non-adaptive play
- Studying opponent modeling effectiveness

---

### 3.6 PureNNBot

| Property | Value |
|----------|-------|
| **Difficulty** | Strong |
| **File** | `bots/pure_nn_bot.py` |
| **Intelligence** | PPO self-play, ~100k parameters |

**What it does:**

Trained via Proximal Policy Optimization (PPO) on 200K self-play games in the terminal. Learns bluffing strategy from scratch through trial and error. No hand-crafted rules — pure neural network decision making.

**Decision Logic:**

```
decide_play(hand, game_state):
    # Encode state to tensor
    state = encoder.encode(hand, game_state)

    # Forward pass through policy network
    logits = policy_network(state)
    action_dist = Categorical(logits=logits)

    # Sample action (or use argmax for deployment)
    action_idx = action_dist.sample()  # Training: sample, Deployment: argmax
    cards, rank = decode_action(action_idx, hand)
    return (cards, rank)

decide_call(last_action, game_state):
    state = encoder.encode(hand, game_state)
    logits = call_network(state)
    return logits[1] > 0.5  # Threshold on "call" logit

observe_action(action, opponent_hand_size):
    # Store for training (not used during deployment)
    pass

save(path):
    torch.save(policy_network.state_dict(), path + "/policy.pt")
    torch.save(call_network.state_dict(), path + "/call.pt")

load(path):
    policy_network.load_state_dict(torch.load(path + "/policy.pt"))
    call_network.load_state_dict(torch.load(path + "/call.pt"))
```

**Network Architecture:**

```
Input: ~50-dim state vector
├── Hidden 1: 128 units, ReLU
├── Hidden 2: 64 units, ReLU
├── Policy Head: 52 × 13 action space (card selection × rank)
└── Value Head: 1 (state value for advantage estimation)

Total parameters: ~100K
```

**Training Details:**

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Self-play games | 200,000 |
| Training time | ~2 hours on CPU |
| Discount factor (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip epsilon | 0.2 |
| Learning rate | 3e-4 |
| Batch size | 64 |

**Strengths:**
- **No hand-crafted rules** — learns strategy from scratch
- **Deep strategic understanding** — can discover non-obvious patterns
- **Consistent play** — no randomness in deployment (argmax)

**Weaknesses:**
- **No opponent modeling** — plays same strategy against everyone
- **Fixed after training** — doesn't adapt to specific opponents
- **Black box** — decisions are hard to explain
- **Training required** — needs 200K games to converge

**When to use it:**
- Studying neural network approach to deception games
- Comparing hand-crafted vs learned strategies
- Baseline for hybrid approaches

---

### 3.7 HybridBot

| Property | Value |
|----------|-------|
| **Difficulty** | Strongest |
| **File** | `bots/hybrid_bot.py` |
| **Intelligence** | PPO base + Bayesian personalization layer |

**What it does:**

Research contribution — combines PPO neural network with Bayesian opponent modeling. The NN handles general strategy, while the Bayesian layer personalizes to specific opponents. Does hybrid outperform pure NN? This is the research question.

**Decision Logic:**

```
decide_play(hand, game_state):
    # Get NN base strategy
    state = encoder.encode(hand, game_state)
    nn_logits = policy_network(state)
    nn_probs = softmax(nn_logits)

    # Get opponent model adjustments
    call_freq = model.estimate_call_frequency()
    aggression = model.get_aggression_level()

    # Blend strategies based on confidence
    model_weight = min(0.4, model.total_actions_observed / 100)

    # Adjust NN probabilities based on opponent model
    if call_freq > 0.6:
        # Opponent calls a lot — reduce bluffs
        adjusted_probs = nn_probs * (1 - model_weight) + conservative_probs * model_weight
    elif call_freq < 0.3:
        # Opponent passes a lot — increase bluffs
        adjusted_probs = nn_probs * (1 - model_weight) + aggressive_probs * model_weight
    else:
        adjusted_probs = nn_probs

    # Sample from adjusted distribution
    action_dist = Categorical(probs=adjusted_probs)
    action_idx = action_dist.sample()
    cards, rank = decode_action(action_idx, hand)
    return (cards, rank)

decide_call(last_action, game_state):
    # NN base decision
    state = encoder.encode(hand, game_state)
    nn_call_prob = call_network(state)[1]

    # Opponent model adjustment
    features = {
        "hand_size": game_state["opponent_hand_size"],
        "claimed_rank": last_action.claimed_rank,
        "claim_size": len(last_action.cards_played),
    }
    model_bluff_prob = model.estimate_bluff_probability(features)

    # Card counting signal
    counting_bluff_prob = counter.bluff_probability(
        last_action.claimed_rank,
        len(last_action.cards_played)
    )

    # Three-way fusion
    model_weight = min(0.3, model.total_actions_observed / 40)
    counting_weight = 0.3
    nn_weight = 1.0 - model_weight - counting_weight

    final_prob = (nn_call_prob * nn_weight +
                  model_bluff_prob * model_weight +
                  counting_bluff_prob * counting_weight)

    return final_prob > 0.5

observe_action(action, opponent_hand_size):
    model.observe_action(action, opponent_hand_size)
    counter.record_play(action.cards_played)

save(path):
    torch.save(policy_network.state_dict(), path + "/policy.pt")
    torch.save(call_network.state_dict(), path + "/call.pt")
    model.save(path + "/opponent_model.json")

load(path):
    policy_network.load_state_dict(torch.load(path + "/policy.pt"))
    call_network.load_state_dict(torch.load(path + "/call.pt"))
    model.load(path + "/opponent_model.json")
```

**Network Architecture:**

Same as PureNNBot but with additional input features from opponent model:

```
Input: ~50-dim state vector + ~10-dim opponent model features
├── Hidden 1: 128 units, ReLU
├── Hidden 2: 64 units, ReLU
├── Policy Head: 52 × 13 action space
└── Value Head: 1

Total parameters: ~120K
```

**Strengths:**
- **Best of both worlds** — NN general strategy + Bayesian personalization
- **Adapts to opponents** — learns per-user patterns
- **Deep understanding** — NN captures complex patterns
- **Research contribution** — answers "does hybrid outperform pure NN?"

**Weaknesses:**
- **Most complex** — harder to debug and explain
- **More training data needed** — needs both self-play and opponent data
- **Potential overfitting** — might over-adapt to specific opponents
- **Computational cost** — runs both NN and Bayesian model

**When to use it:**
- Research experiments — comparing hybrid vs pure approaches
- Studying personalization in deception games
- Testing against adaptive humans

---

## 4. Win Rate Table

Estimated round-robin win rates (each pair plays 100 games):

| Bot | vs Random | vs Rule | vs CardCount | vs Honest | vs Adaptive |
|-----|-----------|---------|--------------|-----------|-------------|
| **Random** | — | ~30% | ~25% | ~35% | ~15% |
| **Rule** | ~70% | — | ~45% | ~55% | ~35% |
| **CardCount** | ~75% | ~55% | — | ~65% | ~45% |
| **Honest** | ~65% | ~45% | ~35% | — | ~25% |
| **Adaptive** | ~85% | ~65% | ~55% | ~75% | — |

### Interpretation

- **Random** loses to everyone — no strategy
- **Rule** beats Random consistently but struggles against adaptive play
- **CardCount** is the best non-adaptive bot — pure math wins
- **Honest** is predictable — opponents learn to never call bluff
- **Adaptive** dominates — learning compounds over games

### Notes

- Win rates are estimates — actual results depend on game parameters
- Adaptive's advantage grows with repeated games against same opponent
- Honest's weakness is exploitable — opponents bluffs freely
- CardCount is the "rational baseline" — hard to beat without adaptation

---

## 5. Bot Selection UX

Player sees bot difficulty levels in the lobby:

| Display Name | Bot | Difficulty | Description |
|--------------|-----|------------|-------------|
| **Beginner** | RandomBot | Trivial | Random moves — easy win |
| **Easy** | HonestBot | Predictable | Never bluffs — learnable |
| **Medium** | CardCountBot | Moderate | Mathematical play — challenging |
| **Hard** | AdaptiveBot | Strong | Learns your patterns — tough |
| **Expert** | PureNNBot | Strong | Neural network — very tough |
| **Master** | HybridBot | Strongest | NN + adaptation — hardest |
| **Challenge** | HonestBot | Predictable | Never bluffs — special mode |

### Lobby Flow

```
1. Player selects "Play vs Bot"
2. Bot selection screen appears
3. Player sees difficulty levels with descriptions
4. Player selects bot
5. Game starts
6. Bot's difficulty shown during game (subtle badge)
```

### Difficulty Badges

```
Beginner  ● (green)
Easy      ●● (yellow)
Medium    ●●● (orange)
Hard      ●●●● (red)
Expert    ★★★★★ (purple)
Master    👑 (gold)
Challenge ⚡ (special)
```

---

## 6. Benchmarking Protocol

Each bot plays 100 games vs scripted opponents with known bluff rates.

### Scripted Opponents

| Opponent | Bluff Rate | Call Rate | Description |
|----------|------------|-----------|-------------|
| Passive | 20% | 20% | Rarely bluffs, rarely calls |
| Balanced | 30% | 30% | Average player |
| Aggressive | 40% | 40% | Frequent bluffs and calls |
| Maniac | 50% | 50% | Maximum chaos |

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Win Rate** | % of games won against each opponent |
| **Bluff Success Rate** | % of bluffs that go uncalled |
| **Call Accuracy** | % of calls that are correct |
| **Adaptation Speed** | Turns to reach 60% win rate against same opponent |
| **Avg Game Length** | Turns per game (shorter = more aggressive) |
| **Pile Size on Win** | Cards in hand when winning (smaller = more efficient) |

### Benchmarking Script

```python
def benchmark(bot_class, num_games=100):
    results = {}
    scripted_opponents = [
        ("Passive", 0.2, 0.2),
        ("Balanced", 0.3, 0.3),
        ("Aggressive", 0.4, 0.4),
        ("Maniac", 0.5, 0.5),
    ]

    for name, bluff_rate, call_rate in scripted_opponents:
        opponent = ScriptedBot(bluff_rate, call_rate)
        bot = bot_class()

        wins = 0
        total_bluffs = 0
        successful_bluffs = 0
        total_calls = 0
        correct_calls = 0

        for game in range(num_games):
            game_result = play_game(bot, opponent)
            if game_result.winner == bot:
                wins += 1

            for action in game_result.actions:
                if action.player == bot:
                    if action.was_bluff:
                        total_bluffs += 1
                        if not action.bluff_called:
                            successful_bluffs += 1
                else:
                    if action.bluff_called:
                        total_calls += 1
                        if action.caller_was_right:
                            correct_calls += 1

        results[name] = {
            "win_rate": wins / num_games,
            "bluff_success": successful_bluffs / max(1, total_bluffs),
            "call_accuracy": correct_calls / max(1, total_calls),
        }

    return results
```

### Expected Results

| Bot | vs Passive | vs Balanced | vs Aggressive | vs Maniac |
|-----|------------|-------------|---------------|-----------|
| Random | ~40% | ~30% | ~25% | ~20% |
| Rule | ~75% | ~70% | ~60% | ~50% |
| CardCount | ~80% | ~75% | ~65% | ~55% |
| Honest | ~70% | ~65% | ~55% | ~45% |
| Adaptive | ~90% | ~85% | ~75% | ~65% |
| PureNN | ~85% | ~80% | ~70% | ~60% |
| Hybrid | ~92% | ~88% | ~78% | ~68% |

---

*This document describes all bot implementations. Update when adding new bots or changing decision logic.*
