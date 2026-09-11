"""BluffNet — actor-critic network + 54-action space (docs/neural-network.md).

Action space (54 actions):
    0        call bluff
    1-52     play 1-4 cards claiming rank r:  a = (r-2)*4 + qty
             r=2 → 1-4, r=3 → 5-8, ..., r=A → 49-52
    53       pass (draw 1 card)

The network never samples unmasked: decode_action() re-derives a legal
(cards, rank) pair from the raw hand, so even an out-of-distribution action
index maps to a legal move (fallback = honest play of most-common rank).
"""

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cards import Card, Rank

ACTION_DIM = 54
PASS_ACTION = 53
CALL_ACTION = 0


def action_index(rank: Rank, quantity: int) -> int:
    """(rank, quantity) → action index in [1, 52]."""
    return (int(rank) - 2) * 4 + quantity


def decode_action_index(idx: int) -> Tuple[str, int]:
    """Action index → ('call',) | ('pass',) | ('play', rank_value).

    Play encoding: idx = (rank_value - 2) * 4 + quantity, quantity 1-4.
    (Docs write '+ 1' but that collides quantity 4 with PASS; this layout
    keeps play actions in [1, 52] with no collisions.)
    """
    if idx == CALL_ACTION:
        return ("call", 0)
    if idx == PASS_ACTION:
        return ("pass", 0)
    rank_val = (idx - 1) // 4 + 2
    return ("play", rank_val)


class BluffNet(nn.Module):
    """Actor-critic for Bluff. ~100k parameters (docs/neural-network.md)."""

    def __init__(self, state_dim: int = 39, action_dim: int = ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor):
        """Returns (logits[action_dim], value[1])."""
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

    def act(self, state: torch.Tensor, legal_mask: torch.Tensor,
            deterministic: bool = False) -> int:
        """Sample a legal action index given a boolean mask (True = legal).

        Inputs are moved to the network's device, so callers can keep
        encoder outputs on CPU even when the net lives on CUDA.
        """
        device = next(self.parameters()).device
        state = state.to(device)
        legal_mask = legal_mask.to(device)
        logits, _ = self.forward(state)
        masked = logits.clone()
        masked[~legal_mask] = float("-inf")
        probs = F.softmax(masked, dim=-1)
        if deterministic:
            return int(torch.argmax(probs).item())
        return int(torch.multinomial(probs, 1).item())


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class BluffNetXL(nn.Module):
    """Deep residual actor-critic architecture for Bluff (~650k parameters).
    
    Features:
    - 512-dim embedding with LayerNorm and GELU activations.
    - Dual residual highway blocks preventing vanishing gradients during deep RL.
    - Decoupled actor and critic projection heads with non-linear bottlenecks.
    """

    def __init__(self, state_dim: int = 39, action_dim: int = ACTION_DIM, hidden_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(state)
        features = self.blocks(features)
        return self.actor(features), self.critic(features)

    def act(self, state: torch.Tensor, legal_mask: torch.Tensor,
            deterministic: bool = False) -> int:
        device = next(self.parameters()).device
        state = state.to(device)
        legal_mask = legal_mask.to(device)
        logits, _ = self.forward(state)
        masked = logits.clone()
        masked[~legal_mask] = float("-inf")
        probs = F.softmax(masked, dim=-1)
        if deterministic:
            return int(torch.argmax(probs).item())
        return int(torch.multinomial(probs, 1).item())


def build_legal_actions(hand: List[Card], can_call: bool,
                        can_pass: bool,
                        respond_only: bool = False) -> torch.Tensor:
    """Boolean mask over the 54-action space.

    Legal play actions: any (rank, qty) with 1 <= qty <= min(4, len(hand)).
    NOTE: we intentionally allow claiming ranks we don't hold — that's bluffing,
    the whole point of the game. Card availability is resolved by
    decode_action_into(), which plays honest cards when it holds the rank and
    dumps arbitrary cards when it doesn't.

    respond_only=True restricts the mask to {call, pass}: used when the agent
    must respond to an opponent's play. Without it, play actions dominate the
    softmax (52 of 54), so the agent samples a meaningless "play" index ~96%
    of the time and the harness coerces it into a call — corrupting both
    behavior and PPO gradients (the v1 draw-rate bug, see docs/decisions.md).
    """
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    n = len(hand)
    if not respond_only and n > 0:
        for rank in Rank:
            for qty in range(1, min(4, n) + 1):
                mask[action_index(rank, qty)] = True
    if can_call:
        mask[CALL_ACTION] = True
    if can_pass:
        mask[PASS_ACTION] = True
    return mask


def decode_action_into(idx: int, hand: List[Card]) -> Optional[Tuple[List[Card], Rank]]:
    """Convert an action index into (cards, claimed_rank) from the raw hand.

    - play N of rank r: if we hold >= N cards of r, play those (honest).
      Otherwise play N arbitrary cards (a bluff) — engine validates count.
    - call/pass: returns None (handled by the caller).
    Always returns a legal play if hand is non-empty and idx is a play action.
    """
    kind, val = decode_action_index(idx)
    if kind != "play":
        return None
    rank = Rank(val)
    qty = (idx - 1) % 4 + 1
    qty = min(qty, len(hand))
    if qty <= 0:
        return None

    matching = [c for c in hand if c.rank == rank]
    if len(matching) >= qty:
        return (matching[:qty], rank)
    # Bluff: dump lowest cards (arbitrary but deterministic)
    return (sorted(hand)[:qty], rank)


def legal_call_and_pass(game, responder: int) -> Tuple[bool, bool]:
    """Whether `responder` may call bluff / pass on the last play.

    Mirrors the rules enforced by game.py + the tournament harness:
    - call: legal if there is an uncalled play and responder != that play's owner
    - pass: legal if the draw pile is non-empty (forced call when empty)
    """
    if game.game_over or not game.actions:
        return False, False
    last = game.actions[-1]
    can_call = (not last.bluff_called) and (last.player != responder)
    can_pass = len(game.draw_pile) > 0 and can_call
    return can_call, can_pass
