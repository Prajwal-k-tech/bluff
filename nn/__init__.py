"""nn — PPO self-play training + BluffNet neural network (Phase 3).

Per docs/neural-network.md:
- state_encoder.py: game state → ~50-dim vector (incl. Bayesian features)
- model.py: BluffNet actor-critic (50→256→256, actor 54 actions, critic 1)
- training.py: PPO loop, GAE, opponent pool, checkpoints

Usage:
    python -m nn.training --episodes 2000 --output nn/checkpoints/final.pt
"""
