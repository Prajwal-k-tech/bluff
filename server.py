"""FastAPI backend — WebSocket game room + bot integration.

Run:
    source .venv/bin/activate
    uvicorn server:app --reload --port 8000

WebSocket protocol (JSON messages):

Server → Client:
    game_state    {hand, opponent_hand_size, pile_size, draw_pile_size, turn, current_player, last_action, message, phase}
    game_over     {winner, message}
    error         {message}

Client → Server:
    play          {cards: [index...], rank: "K"}
    call_bluff    {}
    pass          {}
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from cards import Card, Rank
from game import GameState, Action
from bots.base import BotInterface, build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from bots.pure_nn_bot import PureNNBot
from bots.hybrid_bot import HybridBot
from bots.academic_beast_bot import AcademicBeastBot
from db import pg as db

# Opponent-model persistence helpers (imported here to avoid polluting the
# module-level namespace; only used inside GameRoom when the bot is Bayesian).
_OpponentModel = None
_BluffTracker = None
try:
    from bots.bayesian_bot import OpponentModel as _OpponentModel  # type: ignore[no-redef]
    from bots.bayesian_bot import BluffTracker as _BluffTracker  # type: ignore[no-redef]
except ImportError:
    pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Bluff Bot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------

BOT_CLASSES = {
    "random": RandomBot,
    "honest": HonestBot,
    "cardcount": CardCountBot,
    "bayesian": BayesianBot,
    "purenn": PureNNBot,
    "hybrid": HybridBot,
    "beast": AcademicBeastBot,
}


def create_bot(name: str) -> BotInterface:
    cls = BOT_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown bot: {name}")
    return cls()


# ---------------------------------------------------------------------------
# Game room
# ---------------------------------------------------------------------------

class GameRoom:
    """Manages a single game session between a human (WebSocket) and a bot."""

    def __init__(self, room_id: str, bot_name: str = "bayesian"):
        self.room_id = room_id
        self.game = GameState(num_players=2)
        self.bot: BotInterface = create_bot(bot_name)
        self.bot_name = bot_name
        self.human_id = 0
        self.bot_id = 1
        self.ws: Optional[WebSocket] = None
        self.gamestarted = False
        self.waiting_for_human = False
        # S3 web logging (best-effort, see db/pg.py — never breaks gameplay)
        self.db_session_id: Optional[str] = None
        self._prompt_ts: Optional[float] = None
        self._game_start_ts: Optional[float] = None
        # Per-user opponent-model persistence (session-level identity until
        # Clerk lands — see docs/clerk-plan.md).
        self.session_user_id: Optional[str] = None

    async def db_start(self):
        """Open a Neon session row for the current game (await once)."""
        self._game_start_ts = time.time()
        self._prompt_ts = None
        # Identity: create/lookup the users row for every game (not just
        # Bayesian) so game_sessions.player_id is always attributable.
        self._model_loaded = False
        if self.session_user_id:
            try:
                await db.ensure_user(self.session_user_id)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.db_session_id = await db.log_session_start(
                self.bot_name, player_id=self.session_user_id)
        except Exception:  # noqa: BLE001 — db layer already swallows; belt first
            self.db_session_id = None
        # Load persisted opponent model (best-effort: BayesianBot, HybridBot, AcademicBeastBot)
        if (self.session_user_id and _OpponentModel is not None
                and isinstance(self.bot, (BayesianBot, HybridBot, AcademicBeastBot))):
            try:
                saved = await db.load_opponent_model(
                    self.session_user_id, self.bot_name)
                if saved is not None:
                    self.bot.model = _OpponentModel.from_dict(saved["model"])
                    if "bluff_tracker" in saved and _BluffTracker is not None:
                        self.bot.bluff_tracker = _BluffTracker.from_dict(
                            saved["bluff_tracker"])
                    if "bluff_threshold" in saved:
                        self.bot.bluff_threshold = saved["bluff_threshold"]
                    if "call_threshold" in saved:
                        self.bot.call_threshold = saved["call_threshold"]
                    self._model_loaded = True
            except Exception:  # noqa: BLE001
                pass  # fresh model is fine
            # Ablation flag for the claim-(b) experiment (s3-design §4)
            await db.set_model_loaded(self.db_session_id, self._model_loaded)

    def _decision_ms(self) -> Optional[int]:
        """Human decision latency since the last actionable prompt."""
        if self._prompt_ts is None:
            return None
        return max(0, int((time.time() - self._prompt_ts) * 1000))

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.ws = ws

    def start_game(self, cards_per_player: int = 14):
        self.game = GameState(num_players=2)
        self.game.deal(cards_per_player)
        self.bot.reset()
        # Attribute the bot's own bluffs to itself so BluffTracker records
        # outcomes in web play (was always None → tracker never learned).
        self.bot.player_id = self.bot_id
        self.gamestarted = True

    # -- Sending to client ---------------------------------------------------

    def _card_to_dict(self, card: Card) -> dict:
        return {"rank": card.rank.display(), "suit": card.suit.display(),
                "rank_value": int(card.rank), "suit_value": int(card.suit)}

    def _action_to_dict(self, action: Optional[Action]) -> Optional[dict]:
        if action is None:
            return None
        return {
            "player": action.player,
            "cards": [self._card_to_dict(c) for c in action.cards_played],
            "claimed_rank": action.claimed_rank.display(),
            "was_bluff": action.was_bluff,
            "bluff_called": action.bluff_called,
            "caller_was_right": action.caller_was_right,
            "pile_size_before": action.pile_size_before,
        }

    @staticmethod
    def _db_cards(cards: List[Card]) -> list:
        """Compact strings for the DB, mirroring analysis/logger.py."""
        suit_letters = {0: "H", 1: "D", 2: "C", 3: "S"}
        return [f"{c.rank.display()}{suit_letters[int(c.suit)]}" for c in cards]

    async def send_game_state(self, message: str = "", phase: str = ""):
        if self.ws is None:
            return
        hand = self.game.get_hand(self.human_id)
        # Auto-detect phase if not provided
        if not phase:
            if self.game.current_player == self.human_id:
                phase = "play"  # Human's turn to play cards
            elif self.game.can_call_bluff():
                phase = "call_or_pass"  # Human needs to call bluff or pass
            else:
                phase = "waiting"
        if phase in ("play", "call_or_pass"):
            # Human decision latency starts at the actionable prompt (S3).
            self._prompt_ts = time.time()
        payload = {
            "type": "game_state",
            "phase": phase,
            "hand": [self._card_to_dict(c) for c in hand.cards],
            "hand_size": hand.size(),
            "opponent_hand_size": self.game.get_hand(self.bot_id).size(),
            "pile_size": self.game.get_pile_size(),
            "draw_pile_size": len(self.game.draw_pile),
            "turn": self.game.turn_count,
            "current_player": self.game.current_player,
            "last_action": self._action_to_dict(
                self.game.actions[-1] if self.game.actions else None
            ),
            "can_call_bluff": self.game.can_call_bluff(),
            "can_pass": self.game.can_pass(),
            "message": message,
        }

        # Real-time adaptation metrics from OpponentModel (BayesianBot, HybridBot, AcademicBeastBot)
        bot_adaptation = None
        if hasattr(self.bot, "model") and hasattr(self.bot.model, "overall_bluff"):
            bot_adaptation = {
                "estimated_bluff_rate": round(self.bot.model.overall_bluff.mean(), 3),
                "estimated_call_frequency": round(self.bot.model.call_frequency.mean(), 3),
                "actions_observed": self.bot.model.total_actions_observed,
                "model_loaded": getattr(self, "_model_loaded", False),  # HOTFIX Tess 2026-09-11: was reading non-existent attr (always False)
            }
            if hasattr(self.bot, "classifier") and hasattr(self.bot, "opp_bluffs"):
                top_arch, conf = self.bot.classifier.top_archetype(
                    self.bot.opp_bluffs, self.bot.opp_honest, self.bot.opp_calls, self.bot.opp_passes
                )
                bot_adaptation["inferred_archetype"] = top_arch.replace("_", " ")
                bot_adaptation["archetype_confidence"] = round(conf, 3)
        payload["bot_adaptation"] = bot_adaptation

        await self.ws.send_json(payload)

    async def send_game_over(self):
        if self.ws is None:
            return
        winner = self.game.winner
        if winner is None:
            human_won, message = False, "Draw — turn limit reached."
            result = "draw"
        else:
            human_won = winner == self.human_id
            message = "You win!" if human_won else "Bot wins!"
            result = "win" if human_won else "loss"
        # Persist opponent model before closing (best-effort: BayesianBot, HybridBot, AcademicBeastBot)
        if (self.session_user_id
                and isinstance(self.bot, (BayesianBot, HybridBot, AcademicBeastBot))):
            try:
                model_data = {
                    "model": self.bot.model.to_dict(),
                    "bluff_tracker": self.bot.bluff_tracker.to_dict(),
                    "bluff_threshold": getattr(self.bot, "bluff_threshold", 0.30),
                    "call_threshold": getattr(self.bot, "call_threshold", 0.60),
                }
                await db.save_opponent_model(
                    self.session_user_id, self.bot_name, model_data)
            except Exception:  # noqa: BLE001
                pass
        await db.log_session_end(
            self.db_session_id, result, self.game.turn_count,
            int(time.time() - (self._game_start_ts or time.time())),
        )
        await self.ws.send_json({
            "type": "game_over",
            "winner": winner,
            "human_won": human_won,
            "draw": winner is None,
            "message": message,
        })

    async def send_error(self, msg: str):
        if self.ws is None:
            return
        await self.ws.send_json({"type": "error", "message": msg})

    # -- Bot turn ------------------------------------------------------------

    async def run_bot_turn(self):
        """Have the bot play its turn, then ask human to call or pass."""
        if not self.gamestarted or self.game.game_over:
            return

        bot_hand = self.game.get_hand(self.bot_id)
        gs = build_game_state(
            hand_size=bot_hand.size(),
            opponent_hand_size=self.game.get_hand(self.human_id).size(),
            pile_size=self.game.get_pile_size(),
            draw_pile_size=len(self.game.draw_pile),
            turn_number=self.game.turn_count,
            last_action=self.game.actions[-1] if self.game.actions else None,
            cards_played=self.game.get_cards_played(),
            actions=self.game.actions,
        )

        cards, rank = self.bot.decide_play(bot_hand.cards, gs)
        if not cards:
            return

        success, msg = self.game.play_cards(self.bot_id, cards, rank)
        if not success:
            return

        # Bot may have emptied its hand → game over immediately (win is
        # checked in play_cards before any respond phase; without this the
        # human would hang waiting for a turn that never comes).
        if self.game.game_over:
            await self.send_game_over()
            return

        # Update card counter with bot's own play (learn card distribution).
        # NOTE: we do NOT call observe_action here because that would feed
        # the bot's own play into the *opponent* model, corrupting it.
        if isinstance(self.bot, (BayesianBot, HybridBot, AcademicBeastBot)):
            self.bot.counter.update_with_play(cards)

        await db.log_action(
            self.db_session_id, self.game.turn_count, "bot", "play",
            cards_played=self._db_cards(cards), claimed_rank=int(rank),
            was_bluff=not all(c.rank == rank for c in cards),
            hand_size=bot_hand.size(),  # live ref: already post-play size
            opponent_hand_size=self.game.get_hand(self.human_id).size(),
            pile_size=self.game.get_pile_size(),
        )

        # Notify human about bot's play and ask for call/pass
        self.waiting_for_human = True
        await self.send_game_state(
            message=f"Bot played {len(cards)} card(s) as {rank.display()}. "
                    f"Call bluff or pass?"
        )

    # -- Human turn processing -----------------------------------------------

    async def handle_human_play(self, data: dict):
        """Process human's play action."""
        if self.game.current_player != self.human_id:
            await self.send_error("Not your turn.")
            return

        card_indices = data.get("cards", [])
        rank_str = data.get("rank", "")

        if not card_indices or not rank_str:
            await self.send_error("Must specify cards and rank.")
            return

        # Resolve rank
        rank = self._parse_rank(rank_str)
        if rank is None:
            await self.send_error(f"Invalid rank: {rank_str}")
            return

        # Resolve cards from hand
        hand = self.game.get_hand(self.human_id)
        if not all(0 <= i < hand.size() for i in card_indices):
            await self.send_error("Invalid card indices.")
            return

        cards = [hand.cards[i] for i in card_indices]
        if len(cards) == 0:
            await self.send_error("Must play at least 1 card.")
            return
        if len(cards) > 4:
            await self.send_error("Cannot play more than 4 cards.")
            return

        success, msg = self.game.play_cards(self.human_id, cards, rank)
        if not success:
            await self.send_error(msg)
            return

        decision_ms = self._decision_ms()
        await db.log_action(
            self.db_session_id, self.game.turn_count, "human", "play",
            cards_played=self._db_cards(cards), claimed_rank=int(rank),
            was_bluff=not all(c.rank == rank for c in cards),
            hand_size=hand.size(),
            opponent_hand_size=self.game.get_hand(self.bot_id).size(),
            pile_size=self.game.get_pile_size(),
            decision_ms=decision_ms,
        )

        # Tell the client what they played (frontend logs this from game_state)
        await self.send_game_state(
            message=f"You played {len(cards)} card(s) as {rank.display()}."
        )

        self.waiting_for_human = False

        # Check win
        if self.game.game_over:
            await self.send_game_over()
            return

        # Bot decides to call bluff or pass
        await asyncio.sleep(0.3)  # Brief pause for UX
        bot_wants_to_call = self.bot.decide_call(
            last_action=self.game.actions[-1],
            game_state=build_game_state(
                hand_size=self.game.get_hand(self.bot_id).size(),
                opponent_hand_size=self.game.get_hand(self.human_id).size(),
                pile_size=self.game.get_pile_size(),
                draw_pile_size=len(self.game.draw_pile),
                turn_number=self.game.turn_count,
                last_action=self.game.actions[-1],
                cards_played=self.game.get_cards_played(),
                hand=self.game.get_hand(self.bot_id).cards,
                actions=self.game.actions,
            ),
        )

        result_msg = ""
        # §2: empty draw pile → bot MUST call bluff (pass is not allowed)
        if not self.game.can_pass():
            bot_wants_to_call = True

        if bot_wants_to_call:
            success, result_msg, action = self.game.call_bluff(self.bot_id)
            if action:
                self.bot.observe_action(action, self.game.get_hand(self.human_id).size())
                await db.log_action(
                    self.db_session_id, self.game.turn_count, "bot", "call",
                    cards_played=self._db_cards(action.cards_played),
                    claimed_rank=int(action.claimed_rank),
                    was_bluff=action.was_bluff, bluff_called=True,
                    caller_was_right=action.caller_was_right,
                    hand_size=self.game.get_hand(self.bot_id).size(),
                    opponent_hand_size=self.game.get_hand(self.human_id).size(),
                    pile_size=self.game.get_pile_size(),
                )
        else:
            success, result_msg = self.game.pass_turn(passer=self.bot_id)
            # Observe human's play for opponent modeling (the bot is passing
            # on this play — but the model should still learn from it).
            self.bot.observe_action(
                self.game.actions[-1],
                self.game.get_hand(self.bot_id).size(),
            )
            await db.log_action(
                self.db_session_id, self.game.turn_count, "bot", "pass",
                hand_size=self.game.get_hand(self.bot_id).size(),
                opponent_hand_size=self.game.get_hand(self.human_id).size(),
                pile_size=self.game.get_pile_size(),
            )

        if self.game.game_over:
            await self.send_game_over()
            return

        # Now it's bot's turn
        await self.run_bot_turn()

    async def handle_human_call_bluff(self):
        """Process human calling bluff on bot's last play."""
        if not self.game.can_call_bluff():
            await self.send_error("Cannot call bluff right now.")
            return

        success, result_msg, action = self.game.call_bluff(self.human_id)
        if not success:
            await self.send_error(result_msg)
            return

        self.waiting_for_human = False

        if action:
            self.bot.observe_action(action, self.game.get_hand(self.bot_id).size())
            await db.log_action(
                self.db_session_id, self.game.turn_count, "human", "call",
                cards_played=self._db_cards(action.cards_played),
                claimed_rank=int(action.claimed_rank),
                was_bluff=action.was_bluff, bluff_called=True,
                caller_was_right=action.caller_was_right,
                hand_size=self.game.get_hand(self.human_id).size(),
                opponent_hand_size=self.game.get_hand(self.bot_id).size(),
                pile_size=self.game.get_pile_size(),
                decision_ms=self._decision_ms(),
            )

        if self.game.game_over:
            await self.send_game_over()
            return

        # After call_bluff, turn advanced — send state, then it's whoever's turn
        await self.send_game_state(message=result_msg)
        if self.game.current_player == self.bot_id:
            await asyncio.sleep(0.3)
            await self.run_bot_turn()

    async def handle_human_pass(self):
        """Process human passing (draw 1 card, then human plays again).

        Per game-rules.md §2: cannot pass when the draw pile is empty —
        the player MUST call bluff instead.
        """
        if not self.game.can_call_bluff():
            await self.send_error("Cannot pass right now.")
            return

        # §2: empty draw pile → must call bluff, pass is not allowed
        if not self.game.can_pass():
            await self.send_error(
                "Cannot pass — draw pile is empty. You must call bluff."
            )
            return

        success, result_msg = self.game.pass_turn(passer=self.human_id)

        self.waiting_for_human = False

        await db.log_action(
            self.db_session_id, self.game.turn_count, "human", "pass",
            hand_size=self.game.get_hand(self.human_id).size(),
            opponent_hand_size=self.game.get_hand(self.bot_id).size(),
            pile_size=self.game.get_pile_size(),
            decision_ms=self._decision_ms(),
        )

        # Observe human's pass for opponent modeling — a pass tells the
        # Bayesian layer the human chose NOT to call (updates call_frequency).
        # We fabricate a minimal Action because pass_turn doesn't create one.
        self.bot.observe_action(
            Action(
                player=self.human_id,
                cards_played=[],
                claimed_rank=Rank.TWO,   # placeholder — pass has no claim
                was_bluff=False,
                bluff_called=False,
                caller_was_right=False,
                pile_size_before=self.game.get_pile_size(),
            ),
            self.game.get_hand(self.bot_id).size(),
        )

        # After human passes, human plays again
        await self.send_game_state(message=result_msg, phase="play")

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _parse_rank(s: str) -> Optional[Rank]:
        s = s.strip().upper()
        mapping = {
            "2": Rank.TWO, "3": Rank.THREE, "4": Rank.FOUR, "5": Rank.FIVE,
            "6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
            "10": Rank.TEN, "J": Rank.JACK, "Q": Rank.QUEEN, "K": Rank.KING,
            "A": Rank.ACE,
        }
        return mapping.get(s)


# ---------------------------------------------------------------------------
# Active rooms
# ---------------------------------------------------------------------------

rooms: Dict[str, GameRoom] = {}


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Bluff Bot API is running."}


@app.get("/bots")
def list_bots():
    return {"bots": list(BOT_CLASSES.keys())}


def _normalize_user_id(raw_id: Optional[str]) -> Optional[str]:
    """Convert any user identifier (UUID, Clerk sub, device ID) to a valid UUID string."""
    if not raw_id:
        return None
    try:
        return str(uuid.UUID(raw_id))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))


@app.post("/rooms")
def create_room(bot_name: str = "bayesian", user_id: Optional[str] = None):
    room_id = str(uuid.uuid4())[:8]
    room = GameRoom(room_id, bot_name)
    if user_id:
        room.session_user_id = _normalize_user_id(user_id)
    rooms[room_id] = room
    return {"room_id": room_id, "bot": bot_name}


@app.get("/rooms/{room_id}")
def get_room(room_id: str):
    room = rooms.get(room_id)
    if room is None:
        return {"error": "Room not found"}
    return {
        "room_id": room.room_id,
        "bot": room.bot_name,
        "started": room.gamestarted,
        "game_over": room.game.game_over,
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(ws: WebSocket, room_id: str, user_id: Optional[str] = None):
    room = rooms.get(room_id)
    if room is None:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "Room not found"})
        await ws.close()
        return

    await room.connect(ws)

    # Stable per-user identity for opponent-model persistence.
    # Prioritizes query param user_id (Clerk / client UUID), then pre-set room.session_user_id, then random UUID.
    norm_user = _normalize_user_id(user_id)
    if norm_user:
        room.session_user_id = norm_user
    elif not room.session_user_id:
        room.session_user_id = str(uuid.uuid4())

    try:
        # Start game and send initial state
        room.start_game()
        await room.db_start()
        await room.send_game_state(message="Game started! Your turn to play.")

        # If bot goes first (random), run bot turn
        if room.game.current_player == room.bot_id:
            await room.run_bot_turn()

        # Message loop
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await room.send_error("Invalid JSON.")
                continue

            action = data.get("action", "")

            if action == "play":
                await room.handle_human_play(data)
            elif action == "call_bluff":
                await room.handle_human_call_bluff()
            elif action == "pass":
                await room.handle_human_pass()
            elif action == "new_game":
                room.start_game()
                await room.db_start()
                await room.send_game_state(message="New game! Your turn to play.")
                if room.game.current_player == room.bot_id:
                    await room.run_bot_turn()
            else:
                await room.send_error(f"Unknown action: {action}")

    except WebSocketDisconnect:
        pass
