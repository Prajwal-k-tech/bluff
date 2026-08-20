# Neural Network — Offensive Bluffing Engine

Phase 3 deliverable. PPO self-play for offensive bluffing strategy, deployed as pre-trained model weights.

> **Critical distinction:** The NN and Bayesian model solve *different* problems and are *not* redundant.
> - **Bayesian (defensive):** "Is this opponent bluffing?" → when to CALL
> - **PPO NN (offensive):** "When should I bluff?" → action selection / policy
> - **Integration:** NN's general strategy conditioned on Bayesian's per-opponent model

---

## Why a Neural Network

The Bayesian model is excellent at opponent modeling (detecting bluffs) but cannot learn optimal bluffing policy. The NN learns WHEN and HOW to bluff through self-play, discovering patterns like:

- Bluff aggressively when close to winning (few cards left)
- Establish honest reputation early, exploit it late
- Adjust bluff frequency based on pile size (risk/reward)
- Condition opponent's behavior through strategic bluffing

These are *strategic* decisions that require temporal reasoning across a full game — something a per-turn Bayesian classifier cannot capture. The NN learns a *policy* over actions given game state, while the Bayesian model provides *diagnostic* information about opponents. They complement, not duplicate.

---

## Algorithm: PPO

**Why PPO over alternatives (CFR, DQN, A2C):**

| Factor | PPO | DQN | CFR |
|--------|-----|-----|-----|
| Training speed | 20x faster than DQN for card games | Baseline | Slow convergence |
| Imperfect-info handling | Standard approach | Requires modifications | Natural fit but complex |
| Entropy bonus | Critical for stochastic bluffing | Less effective | N/A |
| Real-world precedent | Big 2 (3M games, surpasses amateurs) | Cheat RL Project (slower) | CFR for Cheat (ICGA 2021) |
| Clean Bayesian integration | Natural — NN outputs policy, Bayesian feeds features | Possible but awkward | Harder to combine |

**Key finding from literature:** Entropy coefficient 0.01–0.05 prevents convergence to "always bluff" or "never bluff" extremes. This is the critical hyperparameter for bluffing games — without it, the NN collapses to a deterministic (and exploitable) strategy.

**Key references:**
- Charlesworth (2018): PPO for Big 2 — 512→256→256 architecture, 3M games
- Patwa (2026): ent_coef=0.05 sweet spot — 90.1% vs Random, 43.5% vs Smart
- Big 2 findings: entropy coefficient is critical for bluffing games

---

## Network Architecture (~100k parameters)

```python
class BluffNet(nn.Module):
    """Actor-Critic network for Bluff. ~100k parameters."""

    def __init__(self, state_dim=50, action_dim=54):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Actor head (policy over actions)
        self.actor = nn.Linear(256, action_dim)

        # Critic head (state value)
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)
```

**Architecture rationale:**
- 256→256 hidden layers match Big 2 PPO (proven for card games)
- Shared backbone extracts common features for both policy and value
- Separate heads allow independent learning of action preferences and state evaluation
- ~100k parameters is small enough for fast CPU training (~30 min)

---

## State Encoding (~50 dimensions)

```python
def encode_state(hand, game_state) -> torch.Tensor:
    return torch.tensor([
        # Hand frequency vector (13 values: count of each rank)
        *[hand.count_rank(rank) for rank in Range(2, 15)],  # 2=0, ..., Ace=13

        # Game context (5 values, normalized)
        game_state['opponent_hand_size'] / 26,
        game_state['pile_size'] / 52,
        game_state['draw_pile_size'] / 24,
        game_state['current_rank'] / 14 if game_state['current_rank'] else 0,
        game_state['turn_number'] / 100,

        # Cards remaining per rank (13 values)
        *[game_state['cards_remaining'][rank] / 4 for rank in Range(2, 15)],

        # History features (8 values)
        game_state['last_action_type'],           # 0=call, 1=pass, 2=play
        game_state['last_claimed_rank'] / 14,
        game_state['last_quantity'] / 4,
        game_state['consecutive_passes'] / 10,

        # Bayesian integration features (3 values — CRITICAL)
        game_state['opponent_call_rate'],         # From Bayesian model
        game_state['my_bluff_rate'],               # Rolling average of own bluffs
        game_state['my_bluff_success_rate'],        # Rolling average of successful bluffs
    ])
    # Total: ~50 dimensions
```

**Encoding breakdown:**

| Feature Group | Dims | Normalization | Purpose |
|---------------|------|---------------|---------|
| Hand frequency | 13 | Raw count (0-4) | What cards we hold |
| Game context | 5 | /26, /52, /24, /14, /100 | Game state awareness |
| Cards remaining | 13 | /4 | Card counting |
| History features | 4 | Various | Recent game dynamics |
| Bayesian integration | 3 | 0-1 range | Opponent model + self-tracking |
| **Total** | **~38–50** | | |

**Why `opponent_call_rate` matters:** This is the bridge between the Bayesian model and the NN. When the Bayesian model estimates an opponent calls frequently, the NN learns to bluff *less* against that opponent. When the opponent rarely calls, the NN learns to bluff *more*. The NN doesn't need explicit integration code — it learns this conditioning automatically through training.

**Why `my_bluff_rate` and `my_bluff_success_rate`:** These let the NN track its own behavior over time, enabling strategic reputation management (e.g., "I've been honest for 5 rounds, time to bluff").

---

## Action Space (54 actions)

```
0:      Call bluff
1-4:    Play 1-4 cards claiming rank 2
5-8:    Play 1-4 cards claiming rank 3
9-12:   Play 1-4 cards claiming rank 4
...
49-52:  Play 1-4 cards claiming Ace
53:     Pass (draw 1 card)
```

**Encoding:** Action index = (rank - 2) × 4 + quantity + 1 for play actions.

**Why this encoding:** Compact, deterministic, maps cleanly to game rules. The NN learns to output a probability distribution over all 54 actions, then we mask illegal actions before sampling.

---

## Action Masking

Set logits of illegal actions to -inf before softmax to prevent the network from selecting impossible moves:

```python
def get_action(self, state, legal_actions):
    logits = self.policy(state)

    # Create mask
    mask = torch.zeros(self.action_space_size, dtype=torch.bool)
    mask[legal_actions] = True

    # Set illegal actions to -inf
    masked_logits = logits.clone()
    masked_logits[~mask] = float('-inf')

    # Sample from valid actions only
    probs = F.softmax(masked_logits, dim=-1)
    action = torch.multinomial(probs, 1)
    return action.item()
```

**Common illegal actions to mask:**
- Can't call if you're the last player to act (no one to challenge)
- Can't play cards you don't have in hand
- Can't play 4 cards of rank X if you only have 2
- Can't pass if the game rules force a call (empty draw pile)

---

## Reward Function

The reward function is shaped to incentivize winning while specifically rewarding effective bluffing:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Win game | +1 | Primary objective |
| Lose game | -1 | Primary objective |
| Cards shed per play | +0.01 × cards_played | Incentivizes efficient play |
| Bluff caught | -0.3 × pile_size_taken | Punishes failed bluffs proportionally to cost |
| Successful bluff | +0.1 × cards_dumped | Rewards risk-taking that pays off |

**Reward design rationale:**
- Win/lose signals are sparse but strong — they drive the overall strategy
- Card shedding bonus prevents the NN from stalling (playing safe but slow)
- Bluff penalties/rewards are calibrated: getting caught hurts more than a successful bluff helps, which prevents "always bluff" convergence
- The pile_size_taken multiplier means late-game bluffs (big piles) are punished harder, teaching the NN to bluff more carefully when stakes are high

---

## Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lr` | 3e-4 | Fixed (simpler than annealing for initial experiments) |
| `clip_epsilon` | 0.2 | PPO clipping (universal) |
| `epochs` | 4 | Per update |
| `batch_size` | 2048 | Rollout batch size |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE parameter |
| `ent_coef` | 0.01–0.05 | Ablation needed — critical for stochastic bluffing |
| `vf_coef` | 0.5 | Value function loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `total_episodes` | 200,000 | Training episodes |
| `eval_interval` | 5,000 | Evaluate every N episodes |
| `pool_size` | 30 | Opponent pool snapshots |

**Why these values:**
- `lr=3e-4`: Standard PPO learning rate for small networks. Can anneal to 1e-4 later.
- `ent_coef=0.01–0.05`: This is the *most important* hyperparameter for bluffing. Too low (0.0) collapses to deterministic play. Too high (0.1) becomes too random. Ablation sweep needed.
- `epochs=4`: Fewer epochs = more stable, matches Big 2 and Splendor configs.
- `batch_size=2048`: Large enough for stable gradient estimates, small enough for fast iteration.

---

## Self-Play Training Loop

```python
for episode in range(200_000):
    # 1. Select opponent from pool
    opponent = sample_opponent(pool, strategy="mixed")

    # 2. Play one game
    trajectory = play_game(agent, opponent)

    # 3. Compute rewards
    for step in trajectory:
        step.reward = compute_reward(step)  # See reward function above

    # 4. Compute GAE advantages
    advantages, returns = compute_gae(trajectory, gamma=0.99, lam=0.95)

    # 5. PPO update
    loss = ppo_loss(agent, trajectory, advantages, returns, clip=0.2)
    loss.backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
    optimizer.step()

    # 6. Snapshot to pool periodically
    if episode % 5000 == 0:
        pool.append(agent.snapshot())
        win_rate = evaluate(agent, n_games=1000)
        print(f"Episode {episode}: {win_rate:.1%} vs random")
```

---

## Opponent Sampling Strategy

```python
def sample_opponent(pool, strategy="mixed"):
    r = random.random()
    if r < 0.10:
        return RandomBot()              # 10% random (exploration)
    elif r < 0.20:
        return HonestBot()              # 10% honest (diversity)
    elif r < 0.50:
        return pool[-1]                 # 30% latest checkpoint
    elif r < 0.60:
        return random.choice(pool)      # 10% uniform from pool
    else:
        return elo_weighted_sample(pool) # 40% ELO-weighted
```

**Why this mix:**
- **10% Random:** Prevents overfitting to specific strategies, ensures exploration
- **10% Honest:** Provides a baseline skill level, prevents policy collapse
- **30% Latest:** Keeps training focused on current policy frontier
- **10% Uniform pool:** Exposes agent to diverse historical strategies
- **40% ELO-weighted:** Trains against strongest opponents, drives improvement

---

## Integration with Bayesian Model

The integration is *implicit*, not explicit. The Bayesian model's output is fed as input features to the NN, and the NN learns to condition on it automatically:

```python
# NN outputs base policy
logits, value = nn(observation)

# Bayesian provides opponent context
opponent_call_rate = bayesian_model.estimate_call_frequency()

# Observation includes opponent model output — THIS is the integration point
observation = encode_state(hand, game_state, opponent_call_rate, my_bluff_rate)

# NN conditions on opponent automatically through learned weights
action = sample_from.softmax(logits)
```

**Why implicit integration beats explicit fusion:**
- No hand-crafted weighting rules (e.g., "70% NN + 30% Bayesian")
- The NN learns *how* to use the Bayesian signal through backpropagation
- The NN might discover non-linear interactions we wouldn't think of
- Simpler code, fewer hyperparameters, more robust

**The data flow:**
```
Game State → Bayesian Model → opponent_call_rate → State Encoder → NN → Action
                                    ↑
                              (per-opponent)
```

---

## Evaluation Metrics

| Metric | What It Measures | Target |
|--------|------------------|--------|
| Bluff frequency distribution | Does NN bluff at Nash-equilibrium rates? | ~20-30% of plays (varies by game state) |
| Bluff success rate vs opponent type | Does NN adapt bluffing to opponent? | Higher vs honest, lower vs hawk |
| Adaptation speed | How fast does bluff rate change when opponent adjusts? | <10 rounds to shift strategy |
| Win rate vs baselines | Overall strength | >90% vs Random, >60% vs Honest |
| Ablation: NN alone vs NN+Bayesian | Proves personalization value | +5-10% win rate with Bayesian |

**Ablation study design:**
1. Train NN with `opponent_call_rate` feature removed → baseline
2. Train NN with `opponent_call_rate` feature included → personalized
3. Compare win rates against opponents with varying call tendencies
4. The delta proves the value of Bayesian integration

---

## Research Contribution

**First PPO-based Cheat AI that learns offensive bluffing policy conditioned on Bayesian opponent modeling.**

Key claims:
1. The NN discovers bluffing naturally through self-play with entropy regularization — no explicit bluffing rules needed
2. Conditioning on Bayesian opponent model output improves win rate vs. vanilla PPO
3. Entropy coefficient is the critical hyperparameter for stochastic bluffing games
4. The two-model architecture (defensive Bayesian + offensive NN) cleanly separates concerns while enabling synergy

---

## Expected Training Results

| Episode | Win Rate vs Random | Win Rate vs Honest |
|---------|-------------------|--------------------|
| 0 | ~25% (random) | ~25% |
| 5K | ~60% | ~40% |
| 10K | ~80% | ~55% |
| 50K | ~90% | ~65% |
| 100K+ | ~95% | ~70% |

**Training time:** ~30 min on CPU, <5 min on GPU.

**Convergence notes:**
- Bluff converges faster than Big 2 (2-player vs 4-player, simpler state space)
- Liar's Dice converges in 50K–100K episodes; Bluff should be similar
- Win rate plateaus around 100K episodes; diminishing returns after that

---

## File Structure

```
nn/
├── model.py          # BluffNet architecture, action masking
├── training.py       # PPO training loop, opponent sampling
├── state_encoder.py  # Game state → flat vector
└── checkpoints/      # Saved model weights
    ├── checkpoint_5k.pt
    ├── checkpoint_10k.pt
    ├── ...
    └── final.pt
```

**Key files:**
- `model.py`: Network definition, forward pass, action selection with masking
- `training.py`: Training loop, GAE computation, PPO loss, evaluation
- `state_encoder.py`: Encodes game state into ~50-dim vector (includes Bayesian features)
- `checkpoints/`: Model snapshots for opponent pool and deployment

---

## Deployment

Training runs in terminal. Model weights exported to web server for deployment.

```bash
# Train (terminal)
python -m nn.training --episodes 200000 --output nn/checkpoints/final.pt

# Export for web
python -m nn.export --input nn/checkpoints/final.pt --output static/models/bluffnet.pt
```

**Deployment flow:**
1. Train in terminal (fast, no GPU required)
2. Export model weights to static file
3. Web server loads weights at startup
4. During gameplay, server runs inference per action

**Inference cost:** ~1ms per action (CPU), negligible for real-time play.
