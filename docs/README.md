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
| `bot-modes.md` | All bot implementations: Random, Honest, CardCount, Bayesian, PureNNBot, **HybridBot** (6 bots) |
| `neural-network.md` | PPO architecture, state encoding, action space, training loop, hyperparameters |
| `data-pipeline.md` | What gets logged, telemetry schema, privacy, human study protocol |
| `decisions.md` | Architectural decisions log (ADRs) — includes HonestBot Paradox, S3 partitioning, Thompson Sampling |
| `timeline.md` | Phase breakdown, milestones, success criteria |
| `research-papers.md` | Curated reading list for bluffing AI, PPO, Bayesian methods |
| `benchmarks.md` | Results tables: tournament standings, conditioning curves, ablation study (Sections 1–6) |
| `paper-outline.md` | Living paper outline: claims → evidence map |
| `paper-draft.md` | Full conference paper draft (Markdown) |
| `paper/main.tex` | IEEEtran LaTeX manuscript (ready for pdflatex compilation) |
| `paper/references.bib` | BibTeX references (12 citations: Brown, Moravčík, Bitan, Yeung, Yoshihara, etc.) |
| `literature-synthesis.md` | Deep synthesis of prior work on bluffing AI and imperfect-information games |
| `research-synthesis.md` | Research claims synthesis and experimental evidence mapping |
| `s3-design.md` | Cross-session S3 persistence design: Neon schema, CardCounter vs OpponentModel partitioning |
| `dataset-request-template.md` | Email template for requesting the Bitan & Kraus (2018) human participant dataset |
| `handoff.md` | Session handoff context for AI agents |
| *(repo root)* `worksplit.md` + `AGENT_CHAT.md` | Multi-agent coordination board (Buffy × Muse × Antigravity × Tess): task ownership + message log |

---

## Quick Start

To understand this project in 5 minutes, read:

1. **`README.md`** — you are here
2. **`architecture.md`** — system overview, deployment, tech stack
3. **`game-rules.md`** — how the game works
4. **`bot-modes.md`** — what bots exist and why

---

## Project Status

**Phase 5 complete.** All phases delivered. HybridBot (PPO + Bayesian + CardCount + Thompson Sampling) achieves tournament-best +156 net score across 1,500 games. S3 cross-session persistence live. Frontend complete with game-over overlay, AI Mental Model card, and draw-aware logging. LaTeX conference paper packaged. Deployment configs authored (`render.yaml`); user must provision Render/Vercel/Neon secrets.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Done | Game engine + 6 bots (Random, Honest, CardCount, Bayesian, PureNN, Hybrid) + full Next.js web app with game-over overlay, AI Mental Model, and S3 memory badge |
| Phase 2 | ✅ Done | Neon Postgres schema + `db/pg.py` pipeline + `test_s3_persistence.py` 7/7 green; Clerk identity binding + device UUID fallback |
| Phase 3 | ✅ Done | PPO v7 (100k eps, `final.pt`) — 5 failure modes found and fixed (ADRs). 83.8% bluff success rate, 23.5% desperation spike at ≥15 cards |
| Phase 4 | ✅ Done | HybridBot: 3-way fusion (PPO policy + Bayesian opponent model + hypergeometric CardCounter). Thompson Sampling. 1,500-game round-robin #1 (+156 net). HonestBot Paradox solved (29-1) |
| Phase 5 | ✅ Done | E2 ablation confirms Claim #2 (33.8% vs 20.7% dynamic range, p<0.001). LaTeX paper `docs/paper/main.tex`. Human adaptation study (6.8× persona separation, 0 losses/150 games). Deployment ready (`render.yaml`) |

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
