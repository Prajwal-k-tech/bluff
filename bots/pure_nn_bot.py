"""PureNNBot — PPO-trained neural network bot (Phase 3).

Implements BotInterface on top of a trained BluffNet. Loads weights from a
checkpoint at construction (default: nn/checkpoints/final.pt, overridable via
BLUFF_NN_CHECKPOINT env var or constructor arg).

Decisions:
- Deterministic (argmax over legal actions) for reproducible play
- No opponent modeling — plays the same strategy against everyone
- Respond decisions use a respond-only action mask ({call, pass}); play
  actions are never offered in the respond phase (v1 mask bug, see
  docs/decisions.md ADR 2026-09-10)
- Card counting uses the public-info pool model (pending claims), not
  revealed-card counts — uncalled pile cards stay hidden per game-rules.md §5
- Falls back to random-legal play if the checkpoint is missing (with a
  warning) so the server can register the bot before training completes
"""

import os
import random
from typing import List, Tuple, Optional

from cards import Card, Rank
from game import Action
from bots.base import (BotInterface, pending_claims_from_actions,
                       pool_by_rank)
from nn.state_encoder import opponent_signals_from_actions, POPULATION_PRIOR

from nn.model import (BluffNet, build_legal_actions, decode_action_into,
                      legal_call_and_pass, CALL_ACTION, PASS_ACTION)
from nn.state_encoder import StateEncoder, STATE_DIM

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "nn", "checkpoints", "final.pt")


class PureNNBot(BotInterface):
    """PPO-trained policy. Argmax over legal actions."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        path = checkpoint_path or os.environ.get(
            "BLUFF_NN_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.net: Optional[BluffNet] = None
        self.encoder = StateEncoder()
        self.checkpoint_path = path
        self.player_id: Optional[int] = None
        if os.path.exists(path):
            from nn.training import load_checkpoint
            self.net = load_checkpoint(path)
            # HOTFIX by Muse 2026-09-10 (Buffy to review): encoder is now
            # 39-dim (v5) but older checkpoints (e.g. v4.pt) hold 38-dim nets —
            # forward would crash (mat1 1x39 vs 38x256). Degrade to the
            # documented random fallback instead of tracebacks; proper fix is
            # a 38-dim legacy loader path for v4-vs-v5 eval (Buffy's call).
            if self.net.shared[0].in_features != STATE_DIM:
                print(f"[PureNNBot] WARNING: checkpoint dim "
                      f"{self.net.shared[0].in_features} != encoder dim "
                      f"{STATE_DIM}; falling back to random-legal play.")
                self.net = None
        else:
            print(f"[PureNNBot] WARNING: checkpoint not found at {path}; "
                  f"falling back to random-legal play until trained.")

    # -- helpers ---------------------------------------------------------------

    def _remaining_counts(self, hand: List[Card],
                          game_state: dict) -> dict:
        """Public-info pool model: pool[r] = 4 − own copies − pending claims.

        Uses the same pending-claims accounting as the rule-based bots
        (bots/base.py). The old version subtracted all played cards incl.
        uncalled face-down plays — that leaked hidden information.
        """
        pending = game_state.get("pending_claims")
        if pending is None:
            pending = pending_claims_from_actions(game_state.get("actions", []))
        return pool_by_rank(hand, pending)

    def _context(self, hand: List[Card], game_state: dict) -> dict:
        """Adapt the BotInterface game_state dict to encoder context."""
        # v5: observed public signals from the action history (deployment
        # parity with training). BotInterface doesn't pass `actions`, so use
        # population priors when the harness omits them.
        actions = game_state.get("actions") or []
        viewer = getattr(self, "player_id", None)
        if actions and viewer is not None:
            signals = opponent_signals_from_actions(actions, viewer)
        else:
            keys = ("opponent_call_rate", "opponent_bluff_revealed",
                    "my_bluff_rate", "my_bluff_success_rate")
            signals = dict(zip(keys, POPULATION_PRIOR))
        return {
            "opponent_hand_size": game_state.get("opponent_hand_size", 14),
            "pile_size": game_state.get("pile_size", 0),
            "draw_pile_size": game_state.get("draw_pile_size", 24),
            "turn_number": game_state.get("turn_number", 0),
            "can_pass": game_state.get("can_pass", True),
            "last_action": game_state.get("last_action"),
            "cards_remaining": self._remaining_counts(hand, game_state),
            **signals,
        }

    def _act(self, hand: List[Card], game_state: dict, can_call: bool,
             can_pass: bool, respond_only: bool = False) -> int:
        state = self.encoder.encode(hand, self._context(hand, game_state))
        mask = build_legal_actions(hand, can_call=can_call, can_pass=can_pass,
                                   respond_only=respond_only)
        if self.net is None:
            legal = [i for i in range(54) if mask[i]]
            return random.choice(legal)
        return self.net.act(state, mask, deterministic=True)

    # -- BotInterface ------------------------------------------------------------

    def decide_play(self, hand: List[Card],
                    game_state: dict) -> Tuple[List[Card], Rank]:
        if not hand:
            return ([], Rank.TWO)
        idx = self._act(hand, game_state, can_call=False, can_pass=False)
        result = decode_action_into(idx, hand)
        if result is None:
            # Degenerate fallback: honest play of most-common rank
            counts = {}
            for c in hand:
                counts[c.rank] = counts.get(c.rank, 0) + 1
            best = max(counts, key=counts.get)
            cards = [c for c in hand if c.rank == best][:min(4, len(hand))]
            return (cards, best)
        return result

    def decide_call(self, last_action: Action, game_state: dict) -> bool:
        can_call, can_pass = True, game_state.get("draw_pile_size", 0) > 0
        idx = self._act(game_state.get("hand", []),
                        game_state, can_call=True, can_pass=can_pass,
                        respond_only=True)
        if idx == CALL_ACTION:
            return True
        if idx == PASS_ACTION and can_pass:
            return False
        # Sampled play/illegal pass when forced to decide → call only if
        # passing is impossible (mirrors harness forced-call rule)
        return not can_pass

    def observe_action(self, action: Action, opponent_hand_size: int):
        pass  # No opponent model

    def save(self, path: str):
        if self.net is None:
            return
        from nn.training import save_checkpoint
        save_checkpoint(self.net, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        from nn.training import load_checkpoint
        self.net = load_checkpoint(path)
