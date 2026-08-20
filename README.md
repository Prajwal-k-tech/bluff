# Bluff Bot — Adaptive Opponent Modeling in 2-Player Bluff

## What Is This?
A Python bot that plays the card game Bluff (Cheat) and adapts to your play style. It detects bluffs, models your behavior, and adjusts its strategy over time.

## What's Already Built
- `cards.py` — Card, Deck, Hand classes
- `game.py` — Full Bluff game engine (2-player, challenges, card placement)
- `human.py` — CLI human player interface
- `main.py` — CLI game runner (human vs bot, multiple games)
- `server.py` — FastAPI WebSocket backend (web play)
- `bots/base.py` — BotInterface abstract base class
- `bots/random_bot.py` — Zero-intelligence baseline
- `bots/honest_bot.py` — Never bluffs, predictable baseline
- `bots/cardcount_bot.py` — Hypergeometric probability-based agent
- `bots/rule_bot.py` — Fixed heuristic strategy
- `bots/adaptive_bot.py` — Bayesian opponent model + card counting
- `test_bots.py` — Bot-vs-bot round-robin tournament
- `test_server.py` — WebSocket integration test
- `frontend/` — Next.js 16 web UI (Catppuccin Mocha theme)

## How to Run

```bash
# Play against the bot (CLI)
python main.py

# Play against a specific bot
python main.py --bot random
python main.py --bot honest
python main.py --bot cardcount
python main.py --bot rule
python main.py --bot adaptive    # default

# Play multiple games
python main.py --games 10

# Bot-vs-bot tournament
python test_bots.py --games 20

# Run server (web play)
uvicorn server:app --reload --port 8000

# Run frontend
cd frontend && npm run dev
```

## Project Phases
See `docs/timeline.md` for the full phase breakdown.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🟡 In progress | Game engine + easy bots + web scaffold |
| Phase 2 | ⬜ Not started | Bayesian opponent modeling + benchmarking |
| Phase 3 | ⬜ Not started | PPO self-play training + NN bot |
| Phase 4 | ⬜ Not started | Visualization + data collection |
| Phase 5 | ⬜ Not started | Paper evaluation + submission |

## Tech Stack
- Python 3.14 (game engine, bots, backend)
- FastAPI + WebSocket (real-time web play)
- Next.js 16 + TypeScript + Tailwind (frontend)
- Catppuccin Mocha Peach (design theme)
- Neon serverless Postgres (planned for data collection)

## Documentation
See `docs/README.md` for the full documentation index.
