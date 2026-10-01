# Bluff

Bluff is a two-player, imperfect-information card game with experimental agents and a small web interface. The repository combines a Python game engine, several rule-based and learned agent prototypes, tournament tooling, and a FastAPI/WebSocket backend with a Next.js frontend.

## What is in this repository

- **Game engine:** card, hand, turn, challenge, draw, and terminal-state logic in `cards.py` and `game.py`.
- **Agent prototypes:** random, honest, card-counting, Bayesian, neural-policy, and hybrid bots under `bots/`.
- **Training and analysis:** PPO training code in `nn/` and game logging and summary scripts in `analysis/`.
- **Tournament runner:** `test_bots.py` runs bot matchups and can write a game log.
- **Web interface:** a FastAPI/WebSocket server and a Next.js client in `server.py` and `frontend/`.

## Status and scope

This is an active project and research prototype. The public branch contains implementation and experiment tooling, but does not establish that an agent is stronger than human players, solves the full game, or reliably adapts to real people. No peer-reviewed paper is included in this repository snapshot. Results should be accompanied by the code, configuration, and artifacts used to produce them.

## Run a local game

With Python and the dependencies in `requirements.txt` installed:

```bash
python main.py
```

Choose a specific baseline with:

```bash
python main.py --bot bayesian
```

Run a small bot tournament with:

```bash
python test_bots.py --games 20
```

The tournament runner is an experiment script; this command is not a benchmark or evidence of agent strength. See [`docs/`](docs/) for the rules, agent descriptions, training notes, and deployment setup.

## Repository map

| Path | Contents |
|---|---|
| `cards.py`, `game.py` | Cards and game state transitions |
| `bots/` | Agent implementations |
| `nn/` | Neural policy and training code |
| `analysis/` | Game logs and evaluation summaries |
| `server.py`, `frontend/` | Web application |
| `docs/` | Project documentation |
