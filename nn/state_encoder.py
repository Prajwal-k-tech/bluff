"""StateEncoder — game state → ~50-dim feature vector for BluffNet.

Feature layout (docs/neural-network.md, extended v5 — ADR 2026-09-10):
    [ 0:13)  hand frequency per rank (count 0-4, no normalization — raw counts
             are already bounded and the network learns scale)
    [13:18)  game context: opponent_hand_size/26, pile_size/52,
             draw_pile_size/24, turn_number/100, can_pass flag
    [18:31)  cards remaining per rank (unseen count / 4)
    [31:35)  last action features: claim_size/4, claimed_rank/14,
             last action was bluff (unobservable — use 0), pile_size_before/52
    [35:39)  opponent-conditioning features (v5 — public observed signals):
             opponent_call_rate, opponent_bluff_revealed, my_bluff_rate,
             my_bluff_success_rate
    Total: 39 dims (docs allow "~38-50"; BluffNet input dim matches this)

v5 rationale: v4 filled opponent_call_rate with a CONSTANT 0.3, so the
policy could not tell opponents apart — it learned the average best response
(always call), which is optimal vs bluffing snapshots but catastrophic vs
HonestBot (0% win). The 4 features are now OBSERVED in-game public signals
(see opponent_signals_from_actions) — they vary per opponent, letting the
policy condition its bluffing AND calling on who it faces. This is the
cheap, self-play-learnable version of research claim #2; HybridBot later
replaces the estimates with real Bayesian posteriors.

All values are floats in roughly [0, 1]. Encoder is deterministic and
framework-free except for the output tensor, so unit tests can run without
loading checkpoints.
"""

from typing import List, Optional

import torch

from cards import Card, Rank

STATE_DIM = 39

# Population priors (Beta(3,7) mean 0.3 for call/bluff rates; 0.5 =
# uninformative for revealed-bluff and success rates).
POPULATION_PRIOR = (0.3, 0.5, 0.0, 0.5)

# Ablation switch (docs/neural-network.md §Evaluation Metrics — research
# claim #2: "conditioning on Bayesian opponent model output improves win
# rate vs vanilla PPO"). When False, the 4 conditioning features are filled
# with CONSTANT population priors — the features carry no opponent-specific
# information (the v4 condition, reproduced fairly) while input dimension
# and marginals stay comparable. Checkpoints remain loadable across settings.
USE_BAYESIAN_FEATURES = True


def opponent_signals_from_actions(actions, viewer: int) -> dict:
    """Observed public-signal opponent features from the action history.

    All four signals are legal per game-rules.md §5 (information model):
    - opponent_call_rate: fraction of the OPPONENT's plays that got called
      (in 2-player, every call on their play is ours — public flag).
    - opponent_bluff_revealed: among opponent plays REVEALED by our calls,
      fraction that were bluffs. Revealed cards are public. With no reveals
      this stays at the uninformative prior — exactly the uncertainty the
      Bayesian layer (HybridBot) will later model explicitly.
    - my_bluff_rate / my_bluff_success_rate: self-knowledge, always legal.
    """
    opp = 1 - viewer
    plays = [a for a in actions if a.cards_played]
    opp_plays = [a for a in plays if a.player == opp]
    my_plays = [a for a in plays if a.player == viewer]

    if opp_plays:
        opp_call_rate = sum(1 for a in opp_plays if a.bluff_called) / len(opp_plays)
    else:
        opp_call_rate = POPULATION_PRIOR[0]

    revealed = [a for a in opp_plays if a.bluff_called]
    opp_bluff_revealed = (
        sum(1 for a in revealed if a.was_bluff) / len(revealed)
        if revealed else POPULATION_PRIOR[1])

    my_bluffs = [a for a in my_plays if a.was_bluff]
    my_bluff_rate = len(my_bluffs) / len(my_plays) if my_plays else POPULATION_PRIOR[2]
    my_bluff_success = (
        sum(1 for a in my_bluffs if not a.bluff_called) / len(my_bluffs)
        if my_bluffs else POPULATION_PRIOR[3])

    return {
        "opponent_call_rate": opp_call_rate,
        "opponent_bluff_revealed": opp_bluff_revealed,
        "my_bluff_rate": my_bluff_rate,
        "my_bluff_success_rate": my_bluff_success,
    }


class StateEncoder:
    """Encodes game state into a fixed-size feature vector."""

    def __init__(self, use_bayesian_features: bool = None):
        # Per-instance override wins over the module-level switch.
        self.use_bayesian = (USE_BAYESIAN_FEATURES
                             if use_bayesian_features is None
                             else use_bayesian_features)

    def encode(self, hand: List[Card], game_state: dict) -> torch.Tensor:
        hand_counts = {rank: 0 for rank in Rank}
        for card in hand:
            hand_counts[card.rank] += 1

        features: List[float] = []

        # 13 dims — hand frequency per rank
        for rank in Rank:
            features.append(float(hand_counts[rank]))

        # 5 dims — game context
        features.append(game_state.get("opponent_hand_size", 14) / 26.0)
        features.append(game_state.get("pile_size", 0) / 52.0)
        features.append(game_state.get("draw_pile_size", 24) / 24.0)
        features.append(game_state.get("turn_number", 0) / 100.0)
        features.append(1.0 if game_state.get("can_pass", True) else 0.0)

        # 13 dims — cards remaining per rank (unseen to this player, / 4)
        remaining = game_state.get("cards_remaining") or {}
        for rank in Rank:
            count = remaining.get(rank, 4)
            features.append(min(4, max(0, count)) / 4.0)

        # 4 dims — last action features
        last_action = game_state.get("last_action")
        if last_action is not None:
            features.append(len(last_action.cards_played) / 4.0)
            features.append(int(last_action.claimed_rank) / 14.0)
            # was_bluff is hidden information — never encode it
            features.append(0.0)
            features.append(last_action.pile_size_before / 52.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])

        # 4 dims — opponent-conditioning features (v5). Under ablation these
        # become CONSTANT priors (no opponent info) rather than zeros, so the
        # control condition matches v4's marginal distributions.
        if self.use_bayesian:
            features.append(game_state.get("opponent_call_rate", POPULATION_PRIOR[0]))
            features.append(game_state.get("opponent_bluff_revealed", POPULATION_PRIOR[1]))
            features.append(game_state.get("my_bluff_rate", POPULATION_PRIOR[2]))
            features.append(game_state.get("my_bluff_success_rate", POPULATION_PRIOR[3]))
        else:
            features.extend(POPULATION_PRIOR)

        assert len(features) == STATE_DIM, (
            f"Encoder produced {len(features)} dims, expected {STATE_DIM}")
        return torch.tensor(features, dtype=torch.float32)
