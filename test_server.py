"""Integration test — starts server, runs full WebSocket game, shuts down."""

import asyncio
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

import websockets


def wait_for_server(url: str, timeout: float = 10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url)
            return True
        except Exception:
            time.sleep(0.2)
    return False


async def recv_human_turn(ws, timeout=5):
    """Drain messages until we get a game_state where the human should act.
    
    After sending an action, the server may send multiple messages:
    - Play confirmation ("You played...") — just an ack, human already acted
    - Bot's response ("Bot played...") — now human can call bluff or pass
    - Pass result ("You passed...") — human drew a card, now should play
    - Bluff result ("BLUFF CAUGHT..." / "WRONG CALL...") — game state update
    
    We skip only the play confirmations since they're echoes of the human's action.
    """
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("type") in ("game_over", "error"):
            return msg
        if msg.get("type") != "game_state":
            continue
        message = msg.get("message", "")
        # Skip confirmations of the human's own play ("You played...")
        # These are just acks — the actionable state comes next
        if message.startswith("You played"):
            continue
        return msg


async def _websocket_impl():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "8768"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server("http://localhost:8768/"):
            print("FAIL: Server didn't start")
            return False
        print("[OK] Server started")

        # REST endpoints
        data = json.loads(urllib.request.urlopen("http://localhost:8768/").read())
        assert data["status"] == "ok"
        print("[OK] GET /")

        bots = json.loads(urllib.request.urlopen("http://localhost:8768/bots").read())
        assert "bayesian" in bots["bots"]
        print(f"[OK] GET /bots: {bots['bots']}")

        req = urllib.request.Request("http://localhost:8768/rooms?bot_name=bayesian&user_id=clerk_user_test_123", method="POST")
        room = json.loads(urllib.request.urlopen(req).read())
        room_id = room["room_id"]
        print(f"[OK] POST /rooms with user_id: {room_id}")

        # Play a full game with stable user_id query param
        async with websockets.connect(f"ws://localhost:8768/ws/{room_id}?user_id=clerk_user_test_123") as ws:
            msg = await recv_human_turn(ws)
            assert msg["type"] == "game_state"
            assert msg["hand_size"] == 14
            phase = msg.get("phase", "play")
            print(f"[OK] WS connected: hand={msg['hand_size']} phase={phase}")

            plays = 0
            for _ in range(50):
                if msg.get("type") == "game_over":
                    result = "Human won" if msg.get("human_won") else "Bot won"
                    print(f"[OK] Game over after {plays} plays: {result}")
                    break

                if msg.get("type") != "game_state":
                    print(f"  Unexpected: {msg.get('type')}: {msg.get('message', '')}")
                    break

                if msg.get("type") == "error":
                    print(f"  Error: {msg.get('message')}")
                    break

                phase = msg.get("phase", "")

                if phase == "play":
                    hand = msg["hand"]
                    if not hand:
                        break
                    rank = hand[0]["rank"]
                    await ws.send(json.dumps({"action": "play", "cards": [0], "rank": rank}))
                    plays += 1
                elif phase == "call_or_pass":
                    if random.random() < 0.3:
                        await ws.send(json.dumps({"action": "call_bluff"}))
                    else:
                        await ws.send(json.dumps({"action": "pass"}))
                else:
                    print(f"  Unknown phase: {phase}")
                    break

                # Drain all messages until we get the next game_state for the human
                msg = await recv_human_turn(ws)
            else:
                print(f"[OK] Game ran {plays} plays without crashing")

        # Test new game
        async with websockets.connect(f"ws://localhost:8768/ws/{room_id}") as ws:
            msg = await recv_human_turn(ws)
            if msg["type"] == "game_state":
                await ws.send(json.dumps({"action": "new_game"}))
                msg2 = await recv_human_turn(ws)
                assert msg2["type"] == "game_state"
                assert msg2["hand_size"] == 14
                print("[OK] New game works")

        # Test HybridBot with live bot_adaptation streaming
        req_hybrid = urllib.request.Request(
            "http://localhost:8768/rooms?bot_name=hybrid&user_id=clerk_user_test_hybrid_456",
            method="POST"
        )
        room_hybrid = json.loads(urllib.request.urlopen(req_hybrid).read())
        hybrid_room_id = room_hybrid["room_id"]
        print(f"[OK] POST /rooms with hybrid bot: {hybrid_room_id}")

        async with websockets.connect(f"ws://localhost:8768/ws/{hybrid_room_id}?user_id=clerk_user_test_hybrid_456") as ws_h:
            msg_h = await recv_human_turn(ws_h)
            assert msg_h["type"] == "game_state"
            assert "bot_adaptation" in msg_h
            adapt = msg_h["bot_adaptation"]
            assert adapt is not None, "HybridBot must return bot_adaptation payload"
            assert 0.0 <= adapt["estimated_bluff_rate"] <= 1.0
            assert 0.0 <= adapt["estimated_call_frequency"] <= 1.0
            assert isinstance(adapt["actions_observed"], int)
            print(f"[OK] HybridBot adaptation payload verified: {adapt}")

            # Play 5 turns against HybridBot
            for _ in range(5):
                if msg_h.get("type") in ("game_over", "error"):
                    break
                phase = msg_h.get("phase", "")
                if phase == "play":
                    hand = msg_h["hand"]
                    if not hand:
                        break
                    rank = hand[0]["rank"]
                    await ws_h.send(json.dumps({"action": "play", "cards": [0], "rank": rank}))
                elif phase == "call_or_pass":
                    await ws_h.send(json.dumps({"action": "pass"}))
                msg_h = await recv_human_turn(ws_h)
            print("[OK] HybridBot WebSocket interaction verified")

        print("\n=== ALL TESTS PASSED ===")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_websocket():
    """Pytest-compatible sync entry point (the file has no async plugin).

    Runs the full integration flow (spawns uvicorn, plays a WS game).
    Standalone equivalent: `python3 test_server.py`.
    """
    assert asyncio.run(_websocket_impl()) is True


if __name__ == "__main__":
    success = asyncio.run(_websocket_impl())
    sys.exit(0 if success else 1)
