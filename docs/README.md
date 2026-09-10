# Bluff Bot — Project Documentation

> **Last updated:** August 21, 2026

**Bluff Bot** is an AI-powered Bluff (Cheat) card game featuring Bayesian opponent modeling that learns each player's behavioral patterns, PPO-trained neural network bots via self-play, a real-time web interface for human play, and a research data pipeline designed for a publishable paper on personalized opponent modeling in imperfect-information games.

---

## How to Use This Docs Directory

**Any AI model working on this project should read this README first.** Each doc covers one concern. Read the relevant doc before making changes.

| When you do this… | Update this doc |
|-------------------|-----------------|
| Make architectural decisions | `decisions.md` |
| Add or change bot modes | `bot-modes.md` |
| Land benchmark numbers (any source) | `benchmarks.md` |
| Modify game rules | `game-rules.md` |
| Change NN architecture or training | `neural-network.md` |
| Modify the data pipeline | `data-pipeline.md` |
| Phases change or milestones shift | `timeline.md` |

Always keep this README's file listing current when adding or removing docs.

---

## File Listing

| File | Purpose |
|------|---------|
| `README.md` | **This file.** Master index, project summary, directory structure |
| `architecture.md` | System architecture: deployment targets, WebSocket protocol, API endpoints, DB schema, BotInterface |
| `game-rules.md` | Complete game rules, passing mechanics, information model |
| `bot-modes.md` | All bot implementations: Random, Honest, CardCount, Bayesian, PureNNBot (5 bots) + planned HybridBot |
| `neural-network.md` | PPO architecture, state encoding, action space, training loop, hyperparameters |
| `data-pipeline.md` | What gets logged, telemetry schema, privacy, how data is used for the paper |
| `decisions.md` | Architectural decisions log (ADRs) |
| `timeline.md` | Phase breakdown, milestones, success criteria |
| `research-papers.md` | Curated reading list for bluffing AI, PPO, Bayesian methods |
| `benchmarks.md` | Results tables for the paper (baselines, checkpoints, methods notes) |
| `paper-outline.md` | Living paper outline: claims → evidence map, experiment matrix, skeleton |
| `handoff.md` | Session handoff context for AI agents |
| *(repo root)* `worksplit.md` + `AGENT_CHAT.md` | Two-agent coordination board (Buffy × Muse): task ownership + message log |

---

## Quick Start

To understand this project in 5 minutes, read:

1. **`README.md`** — you are here
2. **`architecture.md`** — system overview, deployment, tech stack
3. **`game-rules.md`** — how the game works
4. **`bot-modes.md`** — what bots exist and why

---

## Project Status

**Phase 3 active.** Engine (5 bots incl. PureNNBot), web app, JSONL pipeline,
DB schema, and the full PPO training stack exist and work. Three documented
PPO failure modes were found and fixed (docs/decisions.md ADRs 2026-09-10);
v5 is the opponent-conditioned run testing research claim #2. Deployment
(Render + Vercel + Neon) still pending.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | 🟡 Partial | Game engine + 5 bots + web scaffold (gameplay works, deployment pending) |
| Phase 2 | 🟡 Partial | JSONL pipeline + DB schema done; Neon not connected, deployment pending |
| Phase 3 | 🔄 Active | PPO implemented; v5 opponent-conditioning run in flight; ablations queued |
| Phase 4 | ⬜ Not started | Hybrid bot + experiments |
| Phase 5 | ⬜ Not started | Paper draft (outline exists: `paper-outline.md`) + submission |

---

## Project Directory Structure

```
Bluff/
├── docs/                # This documentation
├── cards.py             # Card, Deck, Hand classes
├── game.py              # Game engine (state, actions, turns)
├── server.py            # FastAPI WebSocket backend
├── human.py             # CLI human player interface
├── main.py              # CLI game runner
├── test_bots.py         # Bot-vs-bot round-robin tournament
├── test_server.py       # WebSocket integration test
├── bots/                # Bot implementations
│   ├── __init__.py
│   ├── base.py          # BotInterface abstract class
│   ├── random_bot.py    # Zero-intelligence baseline
│   ├── honest_bot.py    # Never bluffs
│   ├── cardcount_bot.py # Hypergeometric probability (bluffs mathematically)
│   ├── bayesian_bot.py  # Bayesian opponent model + card counting + offensive bluffing
│   └── prob.py          # Shared Hypergeometric probability utilities
├── frontend/            # Next.js 16 web UI
│   ├── src/app/
│   │   ├── page.tsx     # Landing page (Clerk auth)
│   │   ├── game/page.tsx # Game page (card UI)
│   │   ├── layout.tsx   # Root layout
│   │   └── globals.css  # Catppuccin Mocha theme
│   └── package.json
├── nn/                  # (Phase 3) Neural network (planned)
└── analysis/            # (Phase 5) Paper analysis (planned)
```

---

*This document is the entry point for all AI and human contributors. Keep it current.*
