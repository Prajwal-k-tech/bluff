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

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.ws = ws

    def start_game(self, cards_per_player: int = 14):
        self.game = GameState(num_players=2)
        self.game.deal(cards_per_player)
        self.bot.reset()
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
        await self.ws.send_json(payload)

    async def send_game_over(self):
        if self.ws is None:
            return
        winner = self.game.winner
        if winner is None:
            human_won, message = False, "Draw — turn limit reached."
        else:
            human_won = winner == self.human_id
            message = "You win!" if human_won else "Bot wins!"
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
        else:
            success, result_msg = self.game.pass_turn(passer=self.bot_id)

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


@app.post("/rooms")
def create_room(bot_name: str = "bayesian"):
    room_id = str(uuid.uuid4())[:8]
    rooms[room_id] = GameRoom(room_id, bot_name)
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
async def websocket_endpoint(ws: WebSocket, room_id: str):
    room = rooms.get(room_id)
    if room is None:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "Room not found"})
        await ws.close()
        return

    await room.connect(ws)

    try:
        # Start game and send initial state
        room.start_game()
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
                await room.send_game_state(message="New game! Your turn to play.")
                if room.game.current_player == room.bot_id:
                    await room.run_bot_turn()
            else:
                await room.send_error(f"Unknown action: {action}")

    except WebSocketDisconnect:
        pass
