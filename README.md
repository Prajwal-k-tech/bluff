# Bluff Bot — Adaptive Opponent Modeling in 2-Player Bluff

## What Is This?
A Python bot that plays the card game Bluff (Cheat) and adapts to your play style. It detects bluffs, models your behavior, and adjusts its strategy over time. Built for a research paper on personalized Bayesian opponent modeling with offensive bluffing strategy.

## What's Already Built
- `cards.py` — Card, Deck, Hand classes
- `game.py` — Full Bluff game engine (2-player, challenges, card placement)
- `human.py` — CLI human player interface
- `main.py` — CLI game runner (human vs bot, multiple games)
- `server.py` — FastAPI WebSocket backend (web play)
- `bots/base.py` — BotInterface + shared public-info pool model (pending claims)
- `bots/random_bot.py` — Zero-intelligence baseline
- `bots/honest_bot.py` — Never bluffs, predictable baseline
- `bots/cardcount_bot.py` — Hypergeometric probability-based agent (bluffs mathematically)
- `bots/bayesian_bot.py` — Bayesian opponent model + card counting + offensive bluffing
- `bots/pure_nn_bot.py` — PPO-trained neural network bot (loads `nn/checkpoints/final.pt`)
- `bots/prob.py` — Shared Hypergeometric probability utilities
- `nn/` — PPO self-play training pipeline (BluffNet actor-critic, GAE, opponent pool, entropy annealing, `--device auto`)
- `analysis/logger.py` — JSONL game logging (unified data-pipeline format)
- `analysis/report.py` — Bluff analytics: per-bot bluff/call stats, bluff-by-hand-size, matchup outcomes
- `db/schema.sql` — Neon Postgres schema (web data source)
- `test_bots.py` — Bot-vs-bot round-robin tournament (5 bots)
- `test_engine.py` — Engine unit tests (15 tests)
- `test_server.py` — WebSocket integration test
- `worksplit.md` + `AGENT_CHAT.md` — two-agent (Buffy × Muse) coordination board
- `frontend/` — Next.js 16 web UI (Catppuccin Mocha theme, Clerk auth)

## How to Run

```bash
# Play against the bot (CLI)
python main.py

# Play against a specific bot
python main.py --bot random
python main.py --bot honest
python main.py --bot cardcount
python main.py --bot bayesian    # default

# Play multiple games
python main.py --games 10

# Bot-vs-bot tournament (includes PureNN once nn/checkpoints/final.pt exists)
python test_bots.py --games 20

# PPO training (see docs/neural-network.md)
python -m nn.training --episodes 40000 --output nn/checkpoints/v4.pt

# Analytics from JSONL logs
python analysis/report.py --input data/terminal/training_v4.jsonl

# Engine unit tests
python test_engine.py

# Run server (web play)
uvicorn server:app --reload --port 8000

# Run frontend
cd frontend && npm run dev
```

## Project Phases
See `docs/timeline.md` for the full phase breakdown and `worksplit.md` for the live task board.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🟡 Partial | Game engine + 5 bots + web scaffold (gameplay works, deployment pending) |
| Phase 2 | 🟡 Partial | JSONL pipeline + DB schema done; Neon not connected, deployment pending |
| Phase 3 | 🔄 Active | PPO pipeline implemented (2 ADR'd bug-fix rounds); v4 run validating; ablation flags shipped |
| Phase 4 | ⬜ Not started | Hybrid bot + experiments |
| Phase 5 | ⬜ Not started | Paper draft + submission |

## Tech Stack
- Python 3.14 (game engine, bots, backend)
- FastAPI + WebSocket (real-time web play)
- PyTorch 2.14 CPU (PPO training; `--device auto` ready for CUDA)
- Next.js 16 + TypeScript + Tailwind (frontend)
- Catppuccin Mocha Peach (design theme)
- Clerk (authentication)
- Neon serverless Postgres (schema ready, not connected)

## Documentation
See `docs/README.md` for the full documentation index. Key docs:
- `docs/game-rules.md` — LOCKED rules spec
- `docs/decisions.md` — ADRs (including the 2026-09-10 PPO pipeline fixes)
- `docs/neural-network.md` — PPO architecture as implemented
- `docs/handoff.md` — current state for the next session/agent
- `worksplit.md` — two-agent task board (Buffy × Muse)
