# Architecture

> **Last updated:** August 10, 2026
> **Scope:** System design, deployment, protocols, database, interfaces.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Why This Split](#2-why-this-split)
3. [Tech Stack](#3-tech-stack)
4. [Deployment Targets](#4-deployment-targets)
5. [WebSocket Protocol](#5-websocket-protocol)
6. [API Endpoints](#6-api-endpoints)
7. [Database Schema](#7-database-schema)
8. [Game Engine Interface](#8-game-engine-interface)
9. [Key Design Decisions](#9-key-design-decisions)

---

## 1. System Overview

```
┌─────────────────────────────────────────┐
│  TERMINAL (Training + Bot vs Bot)       │
│  - 200K self-play games → model weights │
│  - Round-robin tournaments → win rates  │
│  - Logs to JSONL files                  │
└──────────────────┬──────────────────────┘
                   ↓ (model weights)
┌─────────────────────────────────────────┐
│  WEB APP (Human Play + Data Collection) │
│  - Humans play via browser              │
│  - Bots serve pre-trained models        │
│  - All actions logged to PostgreSQL     │
└──────────────────┬──────────────────────┘
                   ↓ (periodic export)
┌─────────────────────────────────────────┐
│  UNIFIED DATASET (Paper Analysis)       │
│  - Terminal logs + Web logs combined    │
│  - Python notebooks for visualization  │
└─────────────────────────────────────────┘
```

---

## 2. Why This Split

- **Training in terminal** — fast, free, no network overhead. PPO self-play runs at thousands of games per second on CPU. No need for a web server during training.
- **Human data from web** — the only way to get real humans playing. Bots serve pre-trained models loaded from checkpoints.
- **Merge for paper analysis** — terminal logs (bot vs bot benchmarks) + web logs (human vs bot data) combine into one dataset for the paper.

---

## 3. Tech Stack

### Backend

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | Python + FastAPI | Async, WebSocket native |
| WebSocket | FastAPI native WebSocket | Real-time game events |
| Database driver | `asyncpg` | Neon serverless Postgres |
| Auth | `python-jose` (JWT) | httpOnly cookies |
| Server | Uvicorn | ASGI server |
| ORM | None (raw SQL) | Keeps it simple |

### Frontend

| Component | Technology | Notes |
|-----------|------------|-------|
| Framework | Next.js 14+ (App Router) | Server components where possible |
| Language | TypeScript | Type safety |
| Styling | Tailwind CSS | Utility-first |
| WebSocket | Native `WebSocket` API | No socket.io dependency |
| State | React hooks + context | No Redux needed |

### Database

| Component | Technology | Notes |
|-----------|------------|-------|
| Database | PostgreSQL on Neon | Serverless, scales to zero |
| Hosting | Neon free tier | Branching for dev |
| Connection | `asyncpg` over TCP | WebSocket not needed for DB |

### Authentication

- JWT stored in **httpOnly cookies** (not localStorage)
- Backend validates JWT on every WebSocket connection and API call
- No refresh tokens for MVP — short-lived JWTs are sufficient

### Game State

- **In-memory** within the FastAPI process (dict keyed by `session_id`)
- Game state lost on backend restart (acceptable for MVP)
- Action history persisted to DB after each action for analysis

---

## 4. Deployment Targets

| Component | Platform | Why | Limitations |
|-----------|----------|-----|-------------|
| Frontend | Vercel | Free, fast, Next.js native, preview deploys | — |
| Backend | Render | Free tier (750 hrs/month), WebSocket support, git push deploy | Spins down after 15 min idle (~60s cold start) |
| Database | Neon (serverless Postgres) | Free tier, branching, scales to zero | Pauses after inactivity |

**Alternatives considered:**
- Railway — no longer has a free tier ($5/mo minimum)
- Oracle Cloud — 2 OCPU + 12GB RAM always free, but requires DevOps setup (Nginx, SSL, Docker)

---

## 5. WebSocket Protocol

### Connection

```
ws://backend-host/ws/game/{session_id}
```

The client sends a JWT in the first message after connection for authentication. The server validates and either accepts or closes the connection.

### Server → Client Events

| Event | Payload | When |
|-------|---------|------|
| `game_state` | Full game state (hand, pile size, draw pile, turn, history) | After every action |
| `opponent_play` | `{ player_id, cards_count, claimed_rank, was_bluff }` | After opponent plays cards |
| `challenge_result` | `{ was_correct, cards_revealed, pile_taken_by }` | After a bluff call |
| `game_over` | `{ winner_id, reason, final_scores }` | When game ends |

### Client → Server Events

| Event | Payload | When |
|-------|---------|------|
| `play_cards` | `{ cards: [card_id...], claimed_rank: rank }` | Player plays cards |
| `call_bluff` | `{}` (no payload needed) | Player calls bluff |
| `pass_turn` | `{}` (no payload needed) | Player passes (draws 1 card) |

### Message Format

All messages are JSON:

```json
{
  "event": "play_cards",
  "data": {
    "cards": ["3H", "3S", "3D"],
    "claimed_rank": 3
  }
}
```

---

## 6. API Endpoints

### Authentication

| Method | Path | Body | Response | Notes |
|--------|------|------|----------|-------|
| POST | `/auth/register` | `{ username, password }` | `{ user_id, token }` | Creates user, sets cookie |
| POST | `/auth/login` | `{ username, password }` | `{ user_id, token }` | Validates credentials, sets cookie |
| POST | `/auth/logout` | — | `{ success }` | Clears cookie |
| GET | `/auth/me` | — | `{ user_id, username }` | Returns current user from cookie |

### Game

| Method | Path | Body | Response | Notes |
|--------|------|------|----------|-------|
| POST | `/game/create` | `{ config }` | `{ session_id }` | Create a new game room |
| POST | `/game/join` | `{ session_id }` | `{ session_id }` | Join existing room |
| GET | `/game/{id}/state` | — | `{ game_state }` | Current game state snapshot |

### Stats

| Method | Path | Response | Notes |
|--------|------|----------|-------|
| GET | `/stats/me` | `{ wins, losses, games_played }` | Personal stats |
| GET | `/stats/leaderboard` | `[{ user_id, wins, rating }]` | Top players |

---

## 7. Database Schema

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(30) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Game sessions
CREATE TABLE game_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player1_id UUID REFERENCES users(id),
    player2_id UUID REFERENCES users(id),
    config JSONB DEFAULT '{}',
    winner_id UUID,
    num_turns INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

-- Actions (every play in every game)
CREATE TABLE actions (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES game_sessions(id),
    player_id UUID REFERENCES users(id),
    turn_number INTEGER NOT NULL,
    cards_played JSONB NOT NULL,
    claimed_rank INTEGER NOT NULL,
    was_bluff BOOLEAN NOT NULL,
    bluff_called BOOLEAN DEFAULT FALSE,
    caller_was_right BOOLEAN,
    pile_size_before INTEGER,
    hand_size_before INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Opponent models (serialized Bayesian state)
CREATE TABLE opponent_models (
    user_id UUID REFERENCES users(id),
    opponent_id UUID REFERENCES users(id),
    model_data JSONB NOT NULL,
    games_played INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, opponent_id)
);

-- Indexes
CREATE INDEX idx_actions_session ON actions(session_id);
CREATE INDEX idx_actions_player ON actions(player_id);
CREATE INDEX idx_sessions_player1 ON game_sessions(player1_id);
CREATE INDEX idx_sessions_player2 ON game_sessions(player2_id);
```

### Table Purposes

| Table | What It Stores | Retention |
|-------|---------------|-----------|
| `users` | Auth credentials, profiles | Until deleted |
| `game_sessions` | Game metadata (players, result, config, duration) | 90 days |
| `actions` | Every card play, bluff, challenge — full game replay | 90 days |
| `opponent_models` | Serialized `OpponentModel` per user pair (JSONB) | Until deleted |

### Why These Indexes

- `idx_actions_session` — fast lookup of all actions in a game (for replay, analysis)
- `idx_actions_player` — fast lookup of a player's history (for opponent model loading)
- `idx_sessions_player1/player2` — fast lookup of a user's game history (for stats)

---

## 8. Game Engine Interface

All bots implement this interface so the game engine doesn't care which bot is playing:

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from cards import Card, Rank
from game import Action

class BotInterface(ABC):
    @abstractmethod
    def decide_play(self, hand: List[Card], game_state: dict) -> tuple[List[Card], Rank]:
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

### What `game_state` Contains

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

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **All bots implement BotInterface** | NN can slot in without changing game engine. Easy to swap bots for benchmarking. |
| **Game engine is pure Python, no framework dependencies** | Reusable in terminal (training) and web (FastAPI). No coupling. |
| **StateEncoder is separate from bot logic** | NN uses `StateEncoder` to convert game state to flat vector. Bot logic stays clean. |
| **Opponent model is per-user, persisted as JSONB** | Cross-game learning. Model gets smarter the more you play against the same person. |
| **Population prior Beta(3,7)** | New opponents start with a reasonable 30% bluff estimate instead of uniform Beta(1,1). Speeds up adaptation. |
| **Action history persisted to DB** | Enables full game replay for analysis. JSONB is flexible for different card representations. |
| **JWT in httpOnly cookies** | More secure than localStorage. No XSS token theft. |
| **In-memory game state** | Fast, simple, no serialization overhead. State lost on restart is acceptable for MVP. |
| **Render for backend** | Free tier with WebSocket support. Cold start (~60s) acceptable for a research project. |

---

## Appendix: Game Rules Summary

| Parameter | Value |
|-----------|-------|
| Players | 2 (human vs bot) |
| Decks | 1 standard 52-card deck |
| Cards per player | 14 (remaining 24 form the draw pile) |
| Rank progression | **FREE** — claim any rank each turn |
| Cards per turn | 1–4 cards, all claimed to be the same rank |
| Passing | Draw 1 card from the draw pile |
| Challenge | Call "Bluff!" on the most recent play only |
| Bluff caught | Bluffer takes the entire pile |
| Wrong call | Caller takes the entire pile |
| Win condition | First player to empty their hand wins |
| Round limit | 100 turns (draw if no winner) |

See `game-rules.md` for the complete rules specification.

---

*This document describes the current architecture. Update it when deployment targets, protocols, or schemas change.*
