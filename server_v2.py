"""Local adapter for the corrected fixed-rank Bluff prototype.

Run with ``uvicorn server_v2:app --port 8768``.  This adapter intentionally
uses guest cookie ownership with separate profile and research opt-ins.
Historical account hooks remain disabled and are not used by the guest site.
"""

import asyncio
import json
import os
import time
import uuid
from typing import Annotated, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, StrictBool

from bots.base import public_action_for_viewer
from bots.round_observation import round_player_view
from bots.round_policy import (
    RoundAdaptiveBot,
    RoundHonestBot,
    RoundMathBot,
    RoundRandomBot,
)
from cards import Card, Rank
from game_v2 import GameState, RULESET_ID
from bots.round_profile_session import RoundProfileSession
from profile_auth import profile_id_from_authorization
from db import pg as profile_storage
from db import guest as guest_storage
from guest_identity import COOKIE_NAME, CONSENT_VERSION, check_origin, ensure_guest, guest_id
from web_origins import configured_cors_origins


BOT_FACTORIES = {
    "flagship": RoundAdaptiveBot,
    "math": RoundMathBot,
    "honest": RoundHonestBot,
    "random": RoundRandomBot,
}
BOT_DECISION_BURST_LIMIT = 16
BOT_MOVE_DELAY_SECONDS = 0.75
RANKS_BY_DISPLAY = {rank.display(): rank for rank in Rank}

app = FastAPI(title="Bluff Bot Corrected Rules Prototype", version="2.0.0")
CORS_ALLOWED_ORIGINS = configured_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)
rooms: dict[str, "GameRoom"] = {}


def account_profiles_enabled() -> bool:
    return os.environ.get("BLUFF_ENABLE_ACCOUNT_PROFILES", "") == "1"


def create_bot(name: str):
    factory = BOT_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown bot: {name}")
    return factory()


def card_to_dict(card: Card) -> dict:
    return {
        "rank": card.rank.display(),
        "suit": card.suit.display(),
        "rank_value": int(card.rank),
        "suit_value": int(card.suit),
    }


def action_to_dict(action, viewer: int) -> Optional[dict]:
    if action is None:
        return None
    revealed = bool(action.bluff_called)
    owns_action = action.player == viewer
    return {
        "player": action.player,
        "cards": ([card_to_dict(card) for card in action.cards_played]
                  if revealed or owns_action else []),
        "cards_played_count": len(action.cards_played),
        "revealed": revealed,
        "claimed_rank": action.claimed_rank.display(),
        "was_bluff": action.was_bluff if revealed or owns_action else None,
        "bluff_called": action.bluff_called,
        "caller_was_right": action.caller_was_right if revealed else None,
        "pile_size_before": action.pile_size_before,
    }


class GameRoom:
    def __init__(self, room_id: str, bot_name: str, *, verified_user_id=None,
                 guest_user_id=None):
        self.room_id = room_id
        self.bot_name = bot_name
        self.bot = create_bot(bot_name)
        self.human_id = 0
        self.bot_id = 1
        self.bot.player_id = self.bot_id
        self.game = GameState(num_players=2)
        self.ws: Optional[WebSocket] = None
        self.guest_user_id = guest_user_id
        self.session_user_id = guest_user_id or verified_user_id
        self.profile_session = (
            RoundProfileSession(self.session_user_id, profile_storage)
            if self.session_user_id is not None and bot_name == "flagship" else None
        )
        self._profile_loaded = False
        self._refresh_guest_profile = False
        self.log_status = "not_consented"
        self.game_id = None
        self.event_index = 0
        self.game_started_at = 0.0
        self._game_finalized = False

    async def begin_profile_game(self):
        if self.profile_session is not None:
            if (not self.guest_user_id or not self._profile_loaded
                    or self._refresh_guest_profile
                    or self.profile_session.status == "saved"):
                await self.profile_session.begin_game(self.bot)
                self._profile_loaded = True
                self._refresh_guest_profile = False
        if self.guest_user_id:
            self.game_id = str(uuid.uuid4())
            self.event_index = 0
            self.game_started_at = time.monotonic()
            self._game_finalized = False
            try:
                started = await guest_storage.start_game(
                    self.guest_user_id, self.bot_name, self.game_id,
                    {"ruleset_id": RULESET_ID, "policy": type(self.bot).__name__,
                     "transcript_schema": "guest-events-v1"}, CONSENT_VERSION)
                self.log_status = "recording" if started else "not_consented"
            except Exception:
                self.log_status = "unavailable"
            await self.record_guest_event("start", self.game.current_player)

    def _audit_claim(self, action):
        if action is None:
            return None
        # Privileged research transcript only. Never use this for observations
        # or send it to a browser; older hidden packets remain hidden in play.
        return {"action_index": getattr(action, "action_index", None),
                "player": action.player,
                "cards": [card_to_dict(card) for card in action.cards_played],
                "rank": int(action.claimed_rank), "was_bluff": action.was_bluff,
                "bluff_called": action.bluff_called,
                "caller_was_right": action.caller_was_right,
                "response_was_call": getattr(action, "response_was_call", None),
                "response_was_forced": getattr(action, "response_was_forced", None)}

    async def record_guest_event(self, kind, actor, previous=None):
        if not self.guest_user_id or self.log_status != "recording":
            return
        game = self.game
        event = {
            "ruleset_id": RULESET_ID, "event_index": self.event_index,
            "player_type": "human" if actor == self.human_id else "bot",
            "action_type": kind, "round_number": game.round_number,
            "turn": game.turn_count, "current_player": game.current_player,
            "active_rank": int(game.active_rank) if game.active_rank else None,
            "hands": [[card_to_dict(c) for c in hand.cards] for hand in game.hands],
            "pile": [card_to_dict(c) for c in game.pile],
            "draw_supply": [card_to_dict(c) for c in game.draw_pile],
            "latest_claim": self._audit_claim(game.last_action),
            "previous_claim": self._audit_claim(previous),
            "winner": game.winner if game.game_over else None,
        }
        try:
            saved = await guest_storage.append_event(
                self.guest_user_id, self.game_id, self.event_index, event)
            if saved:
                self.event_index += 1
            else:
                self.log_status = "not_consented"
        except Exception:
            self.log_status = "save_failed"

    async def close_guest_game(self):
        if not self.guest_user_id or not self.game_id or self._game_finalized:
            return
        self._game_finalized = True
        completed = self.game.game_over and self.game.winner in (0, 1)
        if self.profile_session is not None:
            await self.profile_session.finish_game(
                self.bot, completed=completed, abandoned=not completed)
        if self.log_status == "recording":
            await self.record_guest_event("end", self.game.current_player)
            if self.log_status != "recording":
                return
            result = ("win" if self.game.winner == self.human_id else "loss") if completed else "abandoned"
            try:
                saved = await guest_storage.finish_game(
                    self.guest_user_id, self.game_id, result, self.game.turn_count,
                    int(time.monotonic() - self.game_started_at))
                self.log_status = "saved" if saved else "not_consented"
            except Exception:
                self.log_status = "save_failed"

    async def finish_profile_game(self):
        if self.guest_user_id:
            await self.close_guest_game()
            return
        if self.profile_session is not None:
            await self.profile_session.finish_game(
                self.bot, completed=self.game.game_over and self.game.winner in (0, 1))

    def adaptation_status(self):
        status = self.profile_session.status if self.profile_session else "room_only"
        available = status in {"new", "restored", "saved"}
        return {
            "persistent_profiles_available": available,
            "profile_status": status,
            "research_logging": self.log_status in {"recording", "saved"},
            "log_status": self.log_status,
            "scope": ("guest" if self.guest_user_id else "account") if available else "room_only",
        }

    def start_game(self):
        self.game = GameState(num_players=2)
        self.game.deal(14)
        self.bot.reset()
        self.bot.player_id = self.bot_id
        # Guest rematches retain room-local behavior. Account games reload the
        # consented snapshot separately before their first action.

    async def connect(self, ws: WebSocket) -> bool:
        if self.ws is not None:
            await ws.accept()
            await ws.send_json({
                "type": "error",
                "message": "This room already has an active connection.",
            })
            await ws.close(code=1008)
            return False
        self.ws = ws
        try:
            await ws.accept()
        except BaseException:
            if self.ws is ws:
                self.ws = None
            raise
        return True

    def _bot_state(self) -> dict:
        return round_player_view(self.game, self.bot_id)

    def _observe(self, action):
        if action is None:
            return
        # Each notification is a detached view. Uncalled human packets retain
        # quantity/rank but never expose their actual cards or honesty label.
        public = public_action_for_viewer(action, self.bot_id)
        self.bot.observe_action(public, self.game.hands[self.human_id].size())

    async def send_error(self, message: str):
        if self.ws is not None:
            await self.ws.send_json({"type": "error", "message": message})

    async def send_game_state(self, message: str = "", resolution_action=None):
        if self.game.game_over:
            await self.finish_profile_game()
        if self.ws is None:
            return
        game = self.game
        human_turn = game.current_player == self.human_id
        active = game.latest_pending_claim
        payload = {
            "type": "game_state",
            "phase": "play" if human_turn else "waiting",
            "hand": [card_to_dict(card) for card in game.hands[self.human_id].cards],
            "hand_size": game.hands[self.human_id].size(),
            "opponent_hand_size": game.hands[self.bot_id].size(),
            "pile_size": len(game.pile),
            "draw_pile_size": len(game.draw_pile),
            "turn": game.turn_count,
            "current_player": game.current_player,
            "last_action": action_to_dict(active, self.human_id),
            "resolution_action": action_to_dict(resolution_action, self.human_id),
            "can_play": game.can_play(self.human_id),
            "can_call_bluff": game.can_call_bluff(self.human_id),
            "can_pass": game.can_pass(self.human_id),
            "active_rank": game.active_rank.display() if game.active_rank else None,
            "round_number": game.round_number,
            "ruleset_id": RULESET_ID,
            "bot_adaptation": self.adaptation_status(),
            "message": message,
        }
        await self.ws.send_json(payload)

    async def send_game_over(self, resolution_action=None):
        await self.finish_profile_game()
        if self.ws is None:
            return
        human_won = self.game.winner == self.human_id
        message = "You won!" if human_won else "Bot wins!"
        await self.ws.send_json({
            "type": "game_over",
            "winner": self.game.winner,
            "human_won": human_won,
            "draw": False,
            "resolution_action": action_to_dict(resolution_action, self.human_id),
            "bot_adaptation": self.adaptation_status(),
            "message": message,
        })

    async def _bot_action(self) -> tuple[bool, str, object | None]:
        game = self.game
        state = self._bot_state()
        choice = self.bot.choose_action(list(game.hands[self.bot_id].cards), state)
        if not isinstance(choice, dict):
            return False, "Bot returned an invalid decision object.", None
        kind = choice.get("action", choice.get("kind"))
        previous = game.latest_pending_claim
        resolution = None
        if kind == "play":
            cards = choice.get("cards")
            rank = choice.get("rank")
            if not isinstance(cards, (list, tuple)) or not isinstance(rank, Rank):
                return False, "Bot returned an invalid play decision.", None
            ok, message = game.play_cards(self.bot_id, list(cards), rank)
            if not ok:
                return False, f"Bot play rejected: {message}", None
            current = game.last_action
            if previous is not None and previous is not current:
                self._observe(previous)
            self._observe(current)
            resolution = current if current.bluff_called else None
            safe_message = f"Bot played {len(cards)} card(s) as {rank.display()}."
            if current.bluff_called:
                safe_message = "Bot's final packet was automatically challenged."
                safe_message += " Bluff caught." if current.was_bluff else " It was truthful."
            await self.record_guest_event("play", self.bot_id, previous)
            return True, safe_message, resolution
        if kind in ("call", "call_bluff", "challenge"):
            if not game.can_call_bluff(self.bot_id):
                return False, "Bot tried to challenge when it was unavailable.", None
            ok, message, resolved = game.call_bluff(self.bot_id)
            if not ok or resolved is None:
                return False, f"Bot challenge failed: {message}", None
            self._observe(resolved)
            await self.record_guest_event("call", self.bot_id, resolved)
            return True, message, resolved
        if kind == "pass":
            if not game.can_pass(self.bot_id):
                return False, "Bot tried to pass when it was unavailable.", None
            ok, _message = game.pass_turn(self.bot_id)
            if not ok:
                return False, "Bot pass failed.", None
            self._observe(previous)
            await self.record_guest_event("pass", self.bot_id, previous)
            return True, "Bot passed and drew 1 card.", None
        return False, f"Bot returned an unknown action: {kind!r}.", None

    async def run_bot_turn(self):
        for _ in range(BOT_DECISION_BURST_LIMIT):
            if self.game.game_over or self.game.current_player != self.bot_id:
                return
            await asyncio.sleep(BOT_MOVE_DELAY_SECONDS)
            try:
                ok, message, resolution = await self._bot_action()
            except Exception as exc:  # noqa: BLE001 — expose policy failures to the room
                ok, message, resolution = (
                    False, f"Bot decision failed: {type(exc).__name__}: {exc}", None
                )
            if not ok:
                await self.send_error(message)
                await self.send_game_state(message=message)
                return
            await self.send_game_state(message=message, resolution_action=resolution)
            if self.game.game_over:
                await self.send_game_over(resolution)
                return
        if not self.game.game_over and self.game.current_player == self.bot_id:
            message = ("Bot decision burst exceeded the safety bound of "
                       f"{BOT_DECISION_BURST_LIMIT} actions.")
            await self.send_error(message)

    async def handle_human_action(self, data: dict):
        game = self.game
        if data.get("action") == "new_game":
            await self.close_guest_game()
            self.start_game()
            await self.begin_profile_game()
            await self.send_game_state(message="New game started.")
            if self.game.current_player == self.bot_id:
                await self.run_bot_turn()
            return
        if game.game_over:
            await self.send_error("Game is over. Start a new game to continue.")
            return
        if game.current_player != self.human_id:
            await self.send_error("It is not your turn.")
            return

        kind = data.get("action")
        previous = game.latest_pending_claim
        resolution = None
        if kind == "play":
            if not game.can_play(self.human_id):
                await self.send_error("You cannot play right now.")
                return
            indices = data.get("cards")
            if (not isinstance(indices, list) or not indices
                    or any(type(index) is not int for index in indices)
                    or len(indices) != len(set(indices))
                    or any(index < 0 or index >= len(game.hands[self.human_id].cards)
                           for index in indices)):
                await self.send_error("Choose one to four distinct valid hand indexes.")
                return
            if len(indices) > 4:
                await self.send_error("Choose one to four distinct valid hand indexes.")
                return
            rank_text = data.get("rank")
            rank = RANKS_BY_DISPLAY.get(rank_text.upper()) if isinstance(rank_text, str) else None
            if rank is None:
                await self.send_error("Choose a valid rank.")
                return
            cards = [game.hands[self.human_id].cards[index] for index in indices]
            ok, message = game.play_cards(self.human_id, cards, rank)
            if not ok:
                await self.send_error(message)
                return
            current = game.last_action
            if previous is not None and previous is not current:
                self._observe(previous)
            self._observe(current)
            resolution = current if current.bluff_called else None
            safe_message = f"You played {len(cards)} card(s) as {rank.display()}."
            if current.bluff_called:
                safe_message = "Your final packet was automatically challenged."
                safe_message += " Bluff caught." if current.was_bluff else " It was truthful."
        elif kind == "call_bluff":
            if not game.can_call_bluff(self.human_id):
                await self.send_error("There is no available claim to challenge.")
                return
            ok, message, resolution = game.call_bluff(self.human_id)
            if not ok or resolution is None:
                await self.send_error(message)
                return
            self._observe(resolution)
            safe_message = message
        elif kind == "pass":
            if not game.can_pass(self.human_id):
                await self.send_error("Passing is unavailable right now.")
                return
            ok, _message = game.pass_turn(self.human_id)
            if not ok:
                await self.send_error("Pass failed.")
                return
            self._observe(previous)
            safe_message = f"You passed and drew 1 card. Bot starts round {game.round_number}."
        else:
            await self.send_error(f"Unknown action: {kind!r}.")
            return

        await self.record_guest_event("call" if kind == "call_bluff" else kind,
                                      self.human_id, previous)
        await self.send_game_state(message=safe_message, resolution_action=resolution)
        if game.game_over:
            await self.send_game_over(resolution)
        elif game.current_player == self.bot_id:
            await self.run_bot_turn()


@app.get("/")
def root():
    return {
        "status": "ok",
        "ruleset_id": RULESET_ID,
        "prototype": True,
        "persistent_profiles_available": False,
        "account_profiles_enabled": account_profiles_enabled(),
        "research_logging": False,
        "guest_persistence": True,
        "message": "Fixed-rank Bluff. Guest storage requires visitor opt-in and an available database.",
    }


@app.get("/bots")
def list_bots():
    return {"bots": list(BOT_FACTORIES)}


@app.post("/rooms")
def create_room(bot_name: str = "flagship",
                authorization: Annotated[Optional[str], Header()] = None,
                request: Request = None, response: Response = None):
    if bot_name not in BOT_FACTORIES:
        raise HTTPException(status_code=400, detail=f"Unknown bot: {bot_name}")
    # An Authorization header is never silently downgraded to guest identity.
    # Explicit opt-in prevents available credentials from enabling DB traffic.
    identity = None
    if authorization:
        if not account_profiles_enabled():
            raise HTTPException(status_code=503, detail="Account profiles are disabled")
        identity = profile_id_from_authorization(authorization)
    room_id = str(uuid.uuid4())
    guest_identity = None
    if request is not None and identity is None:
        check_origin(request, CORS_ALLOWED_ORIGINS)
        guest_identity = ensure_guest(request, response)
    rooms[room_id] = GameRoom(room_id, bot_name, verified_user_id=identity,
                              guest_user_id=guest_identity)
    return {
        "room_id": room_id,
        "bot": bot_name,
        "profile_identity_bound": identity is not None or guest_identity is not None,
        "persistent_profiles_available": False,
        "research_logging": False,
    }


class ProfilePreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cross_session_profile: StrictBool


class GuestPreferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cross_session_profile: StrictBool
    research_logging: StrictBool


async def guest_status(identity):
    try:
        preferences = await guest_storage.get_preferences(identity)
    except Exception:
        return {"preferences": {"cross_session_profile": False, "research_logging": False},
                "consent_version": CONSENT_VERSION,
                "storage": {"database_available": False, "profile_status": "unavailable",
                            "research_logging": False, "scope": "room_only"}}
    profile = preferences.get("cross_session_profile") is True
    logging = (preferences.get("research_logging") is True
               and preferences.get("consent_version") == CONSENT_VERSION)
    return {"preferences": {"cross_session_profile": profile, "research_logging": logging},
            "consent_version": CONSENT_VERSION,
            "storage": {"database_available": True,
                        "profile_status": "enabled" if profile else "not_consented",
                        "research_logging": logging,
                        "scope": "guest" if profile else "room_only"}}


@app.get("/guest")
async def get_guest(request: Request, response: Response):
    check_origin(request, CORS_ALLOWED_ORIGINS)
    identity = ensure_guest(request, response)
    return await guest_status(identity)


@app.put("/guest")
async def put_guest(preferences: GuestPreferenceUpdate, request: Request, response: Response):
    check_origin(request, CORS_ALLOWED_ORIGINS)
    identity = ensure_guest(request, response)
    try:
        saved = await guest_storage.set_preferences(
            identity, cross_session_profile=preferences.cross_session_profile,
            research_logging=preferences.research_logging, consent_version=CONSENT_VERSION)
        if (saved.get("cross_session_profile") is not preferences.cross_session_profile
                or saved.get("research_logging") is not preferences.research_logging):
            raise RuntimeError("preferences not acknowledged")
    except Exception as exc:
        raise HTTPException(503, "Guest preferences could not be saved") from exc
    for room in tuple(rooms.values()):
        if room.guest_user_id != identity:
            continue
        room._refresh_guest_profile = True
        if not preferences.cross_session_profile and room.profile_session is not None:
            room.bot.reset_profile()
            room.profile_session.status = "not_consented"
        if not preferences.research_logging:
            room.log_status = "not_consented"
    result = await guest_status(identity)
    result["changes_apply_to_new_games"] = True
    return result


@app.delete("/guest")
async def delete_guest(request: Request, response: Response):
    check_origin(request, CORS_ALLOWED_ORIGINS)
    identity = guest_id(request.cookies.get(COOKIE_NAME))
    if identity is not None:
        try:
            await guest_storage.delete_data(identity)
        except Exception as exc:
            raise HTTPException(503, "Guest data could not be deleted") from exc
        for room in tuple(rooms.values()):
            if room.guest_user_id == identity:
                room.log_status = "not_consented"
                if room.profile_session is not None:
                    room.bot.reset_profile()
                    room.profile_session.status = "not_consented"
                # Rotate ownership without reviving the deleted identity on a
                # rematch or later opt-in in this still-open WebSocket.
                room.guest_user_id = None
                room.session_user_id = None
                room.profile_session = None
    new_identity = ensure_guest(request, response, rotate=True)
    result = await guest_status(new_identity)
    result["deleted"] = True
    return result


def required_profile_identity(authorization: Optional[str]) -> str:
    if not account_profiles_enabled():
        raise HTTPException(status_code=503, detail="Account profiles are disabled")
    identity = profile_id_from_authorization(authorization)
    if identity is None:
        raise HTTPException(status_code=401, detail="Sign in to manage profile memory")
    return identity


@app.get("/privacy/profile")
async def get_profile_preference(
        authorization: Annotated[Optional[str], Header()] = None):
    identity = required_profile_identity(authorization)
    try:
        preferences = await profile_storage.get_privacy_preferences(identity)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Profile preferences unavailable") from exc
    return {"cross_session_profile": preferences.get("cross_session_profile") is True,
            "research_logging": False}


@app.put("/privacy/profile")
async def put_profile_preference(
        preferences: ProfilePreferenceUpdate,
        authorization: Annotated[Optional[str], Header()] = None):
    identity = required_profile_identity(authorization)
    try:
        saved = await profile_storage.update_profile_preference(
            identity, enabled=preferences.cross_session_profile)
        if saved.get("cross_session_profile") is not preferences.cross_session_profile:
            raise RuntimeError("preference update was not acknowledged")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Profile preference could not be saved") from exc
    if not preferences.cross_session_profile:
        for room in tuple(rooms.values()):
            if room.session_user_id == identity and room.profile_session is not None:
                room.profile_session.status = "not_consented"
                room.bot.reset_profile()
    return {"cross_session_profile": preferences.cross_session_profile,
            "research_logging": False, "changes_apply_to_new_games": True}


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(ws: WebSocket, room_id: str):
    # CORS middleware does not protect WebSocket handshakes. Origin is a
    # browser boundary, not authentication; non-browser clients may omit it.
    origins = ws.headers.getlist("origin")
    if len(origins) > 1 or (origins and origins[0] not in CORS_ALLOWED_ORIGINS):
        await ws.close(code=1008)
        return
    room = rooms.get(room_id)
    if room is None:
        await ws.accept()
        await ws.send_json({"type": "error", "message": "Room not found"})
        await ws.close()
        return
    if room.guest_user_id is not None and guest_id(ws.cookies.get(COOKIE_NAME)) != room.guest_user_id:
        await ws.close(code=1008)
        return
    if not await room.connect(ws):
        return
    try:
        room.start_game()
        await room.begin_profile_game()
        await room.send_game_state(message="Game started.")
        if room.game.current_player == room.bot_id:
            await room.run_bot_turn()
        while True:
            try:
                data = json.loads(await ws.receive_text())
            except json.JSONDecodeError:
                await room.send_error("Invalid JSON.")
                continue
            if not isinstance(data, dict):
                await room.send_error("Expected a JSON object.")
                continue
            await room.handle_human_action(data)
    except WebSocketDisconnect:
        pass
    finally:
        if room.ws is ws:
            await room.close_guest_game()
            room.ws = None
            # Guest profiles are room-local: closing the owning connection
            # must not retain its behavioral data indefinitely in `rooms`.
            if rooms.get(room_id) is room:
                rooms.pop(room_id)
