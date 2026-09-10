"""JSONL game logger — Phase 2 data pipeline (terminal source).

Logs every action of every game to a JSONL file using the unified record
format from docs/data-pipeline.md. Used by bot-vs-bot tournaments and
future PPO self-play training runs.

Design:
- One GameLogger per game (one JSON object per line, appended on flush).
- Action records are written as they happen (crash-safe).
- Game header/footer records bracket the action stream for easy grouping.

Record types:
    {"record": "game_start", ...}   game metadata
    {"record": "action", ...}       one per play/call/pass
    {"record": "game_end", ...}     winner + final stats
"""

import json
import os
import time
import uuid
from typing import List, Optional

from cards import Card, Rank
from game import Action, GameState


def _card_str(card: Card) -> str:
    """Compact card representation, e.g. '3H' (rank letter + suit letter)."""
    suit_letters = {0: "H", 1: "D", 2: "C", 3: "S"}
    return f"{card.rank.display()}{suit_letters[int(card.suit)]}"


class GameLogger:
    """Logs one game to JSONL in the unified data-pipeline format."""

    def __init__(self, output_path: str, source: str = "terminal",
                 bot_mode_a: str = "", bot_mode_b: str = ""):
        self.game_id = str(uuid.uuid4())
        self.source = source
        self.bot_mode_a = bot_mode_a
        self.bot_mode_b = bot_mode_b
        self.started_at = time.time()
        self.action_count = 0

        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Keep one file handle open for the game; append mode is crash-safe.
        self._fh = open(output_path, "a", encoding="utf-8")
        self._write(self._game_start_record())

    # -- record builders ----------------------------------------------------

    def _game_start_record(self) -> dict:
        return {
            "record": "game_start",
            "format_version": 2,
            "game_id": self.game_id,
            "source": self.source,
            "bot_mode_a": self.bot_mode_a,
            "bot_mode_b": self.bot_mode_b,
            "timestamp": self.started_at,
        }

    def log_action(self, action: Action, game: GameState,
                   player_type: str = "bot",
                   caller: Optional[int] = None,
                   pile_size_at_play: Optional[int] = None) -> None:
        """Log a play (and optionally the call against it) with full context.

        Format v2 (2026-09-10, fixes report.py semantics — see ADR):
        - The PLAY is always logged as action_type="play", owned by the
          player who played, with bluff_called/caller_was_right outcome
          fields. (v1 merged called plays into "call" records owned by the
          play-er, which made bluff-success trivially 100% and mislabeled
          call accuracy — Muse catch.)
        - If a call happened, a SECOND record action_type="call" is emitted,
          owned by the CALLER (cards_played empty, was_bluff = whether the
          call was right for the caller).

        `caller` is the player index who called bluff (required when
        action.bluff_called — the engine's Action doesn't record who called).
        `pile_size_at_play` is the pile size BEFORE the call resolved; a call
        empties the pile, so callers must pass it explicitly for the play
        record's pile_size to reflect the stakes at decision time.
        """
        self.action_count += 1
        rec = {
            "record": "action",
            "format_version": 2,
            "game_id": self.game_id,
            "source": self.source,
            "player_type": player_type,
            "bot_mode": (self.bot_mode_a if action.player == 0
                         else self.bot_mode_b),
            "action_type": "play",
            "cards_played": [_card_str(c) for c in action.cards_played],
            "claimed_rank": int(action.claimed_rank),
            "was_bluff": action.was_bluff,
            "bluff_called": action.bluff_called,
            "caller_was_right": action.caller_was_right if action.bluff_called else None,
            "hand_size": game.get_hand(action.player).size(),
            "opponent_hand_size": game.get_hand(1 - action.player).size(),
            "pile_size": (pile_size_at_play if pile_size_at_play is not None
                          else game.get_pile_size()),
            "turn_number": game.turn_count,
            "timestamp": time.time(),
        }
        self._write(rec)

        if action.bluff_called and caller is not None:
            self.action_count += 1
            call_rec = {
                "record": "action",
                "format_version": 2,
                "game_id": self.game_id,
                "source": self.source,
                "player_type": player_type,
                "bot_mode": (self.bot_mode_a if caller == 0
                             else self.bot_mode_b),
                "action_type": "call",
                "cards_played": [],
                "claimed_rank": int(action.claimed_rank),
                "was_bluff": action.caller_was_right,  # from the caller's POV
                "bluff_called": False,
                "caller_was_right": action.caller_was_right,
                "hand_size": game.get_hand(caller).size(),
                "opponent_hand_size": game.get_hand(1 - caller).size(),
                "pile_size": 0,
                "turn_number": game.turn_count,
                "timestamp": time.time(),
            }
            self._write(call_rec)

    def log_pass(self, game: GameState, player: int) -> None:
        """Log a pass action (no Action object exists for passes)."""
        self.action_count += 1
        rec = {
            "record": "action",
            "game_id": self.game_id,
            "source": self.source,
            "player_type": "bot",
            "bot_mode": (self.bot_mode_a if player == 0 else self.bot_mode_b),
            "action_type": "pass",
            "cards_played": [],
            "claimed_rank": None,
            "was_bluff": None,
            "bluff_called": False,
            "caller_was_right": None,
            "hand_size": game.get_hand(player).size(),
            "opponent_hand_size": game.get_hand(1 - player).size(),
            "pile_size": game.get_pile_size(),
            "turn_number": game.turn_count,
            "timestamp": time.time(),
        }
        self._write(rec)

    def log_game_end_simple(self, winner: Optional[int], agent_seat: int,
                            num_turns: int = 0) -> None:
        """Lightweight end-of-game record (used by PPO training, which does
        not keep a GameState reference at logging time)."""
        duration = time.time() - self.started_at
        rec = {
            "record": "game_end",
            "game_id": self.game_id,
            "source": self.source,
            "bot_mode_a": self.bot_mode_a,
            "bot_mode_b": self.bot_mode_b,
            "game_winner": ("draw" if winner is None
                            else (self.bot_mode_a if winner == agent_seat
                                  else self.bot_mode_b)),
            "winner_player": winner,
            "num_turns": num_turns,
            "num_actions": self.action_count,
            "duration_seconds": round(duration, 2),
        }
        self._write(rec)
        self.close()

    def log_game_end(self, game: GameState, winner: Optional[int]) -> None:
        """Log the final game record and close the file handle."""
        duration = time.time() - self.started_at
        rec = {
            "record": "game_end",
            "game_id": self.game_id,
            "source": self.source,
            "bot_mode_a": self.bot_mode_a,
            "bot_mode_b": self.bot_mode_b,
            "game_winner": ("draw" if winner is None
                            else (self.bot_mode_a if winner == 0
                                  else self.bot_mode_b)),
            "winner_player": winner,
            "num_turns": game.turn_count,
            "num_actions": self.action_count,
            "final_hand_size_a": game.get_hand(0).size(),
            "final_hand_size_b": game.get_hand(1).size(),
            "duration_seconds": round(duration, 2),
        }
        self._write(rec)
        self.close()

    # -- internals ------------------------------------------------------------

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()


class TeeLogger:
    """Aggregates stats across many games without holding per-game state.

    Usage:
        tee = TeeLogger("data/terminal/tournament.jsonl")
        for game in games:
            logger = tee.new_game("CardCount", "Bayesian")
            ... play game, call logger.log_action(...) ...
            logger.log_game_end(game, winner)
        tee.summary()  # prints aggregate stats
    """

    def __init__(self, output_path: str, source: str = "terminal"):
        self.output_path = output_path
        self.source = source
        self.games = 0

    def new_game(self, bot_a: str, bot_b: str) -> GameLogger:
        self.games += 1
        return GameLogger(self.output_path, source=self.source,
                          bot_mode_a=bot_a, bot_mode_b=bot_b)

    def summary(self) -> dict:
        return {"games_logged": self.games, "output": self.output_path}
