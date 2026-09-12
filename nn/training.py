"""PPO self-play training for BluffNet (Phase 3).

Implements docs/neural-network.md:
- League opponent sampling (ADR 2026-09-10 "league roster"): 40% scripted
  archetypes (10% Random / 15% Honest / 15% CardCount / 15% Bayesian) +
  45% snapshot pool + 15% mirror — Charlesworth-style scripted floor
- Reward shaping (ADR 2026-09-10 "potential-based hand shaping"): win ±1,
  draw −0.5, hand-size-delta potential 0.03/card, flat time cost 0.01/step
- GAE(λ=0.95), PPO clip 0.2, 4 epochs/update, entropy annealing 0.05→0.01
- Optional R-NaD path (--use-rnad): population baseline, KL regularization,
  exploitability metric. Disabled by default — PPO path is byte-identical.

Usage:
    python -m nn.training --episodes 2000 --output nn/checkpoints/final.pt
    python -m nn.training --episodes 200000 --ent-coef 0.01 --log data/terminal/training.jsonl
    python -m nn.training --episodes 5000 --use-rnad --kl-coef 0.01
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cards import Card, Rank
from game import GameState
from bots.base import BotInterface, build_game_state
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot

from bots.base import pending_claims_from_actions, pool_by_rank
from nn.model import (BluffNet, build_legal_actions, decode_action_into,
                      legal_call_and_pass, CALL_ACTION, PASS_ACTION)
from nn.state_encoder import (StateEncoder, opponent_signals_from_actions,
                              POPULATION_PRIOR, hybrid_claim_features)

# LOCKED rule (game-rules.md §1): 100 turns then draw. The old 200 let
# degenerate policies stall twice as long per game.
MAX_TURNS = 100

# Reward shaping constants (ADR 2026-09-10 "potential-based hand shaping")
WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = -0.5      # draws are jointly irrational in a shedding game
SHED_REWARD = 0.01      # per card shed (kept from docs/neural-network.md)
HAND_POTENTIAL = 0.03   # per card of hand-size DECREASE (potential-based)
TIME_PENALTY = 0.01     # per transition — flat opportunity cost (γ<1-like);
                        # total over a full game ≈ 1.0, comparable to win/loss.
                        # (Do NOT scale by turn index: compounding swamps the
                        # terminal signal and rewards losing-fast over winning-slow.)
BLUFF_CAUGHT_PENALTY = 0.3   # × pile size taken
BLUFF_SUCCESS_BONUS = 0.1    # × cards dumped


# ---------------------------------------------------------------------------
# Policy players — uniform wrapper over rule-based bots and BluffNet policies
# ---------------------------------------------------------------------------

class NNPlayer:
    """Wraps a BluffNet + encoder as a playable policy (used for agent + pool)."""

    def __init__(self, net: BluffNet, encoder: StateEncoder = None,
                 trace_signals: bool = False):
        self.net = net
        self.encoder = encoder or StateEncoder()
        # Optional per-decision record of the observed opponent signals
        # (analysis/conditioning_curve.py claim-#2 evidence). Disabled by
        # default — zero overhead in training/eval.
        self.trace_signals = trace_signals
        self.signal_trace: List[dict] = []

    def _context(self, game: GameState, seat: int) -> dict:
        """Encoder context from the agent's own (fair) perspective."""
        hand = game.get_hand(seat)
        other = 1 - seat
        can_call, can_pass = legal_call_and_pass(game, seat)
        actions = game.actions
        n = len(actions)
        # Self-tracking features (docs/neural-network.md "Bayesian integration")
        my_plays = [a for a in actions if a.player == seat]
        bluffs = [a for a in my_plays if a.was_bluff]
        my_bluff_rate = len(bluffs) / max(1, len(my_plays))
        caught = sum(1 for a in bluffs if a.bluff_called and a.caller_was_right)
        my_bluff_success = 1.0 - caught / max(1, len(bluffs))
        # Pool model over PUBLIC information only (docs/bot-modes.md):
        # pool[r] = 4 − own copies of r − unresolved pile claims of r.
        # Uncalled plays stay hidden (game-rules.md §5) — the old version
        # treated uncalled honest plays as revealed, an info leak.
        pending = pending_claims_from_actions(actions)
        remaining = pool_by_rank(hand.cards, pending)
        # v5 opponent conditioning: observed public signals instead of a
        # constant prior. This is what lets the policy distinguish Honest
        # (0% bluff → never call) from Random (bluffs often → call a lot).
        # (v4's constant 0.3 collapsed the respond policy to always-call,
        # 0% vs Honest — see ADR 2026-09-10.)
        signals = opponent_signals_from_actions(actions, seat)
        signals["my_bluff_rate"] = my_bluff_rate
        signals["my_bluff_success_rate"] = my_bluff_success
        ctx = {
            "opponent_hand_size": game.get_hand(other).size(),
            "pile_size": game.get_pile_size(),
            "draw_pile_size": len(game.draw_pile),
            "turn_number": game.turn_count,
            "can_pass": can_pass,
            "last_action": actions[-1] if actions else None,
            "cards_remaining": remaining,
            **signals,
        }
        if self.encoder.hybrid:
            # v8 hybrid (ADR): structured Bayesian evidence from the shared
            # pool math — the exact features that make Honest/CardCount sane
            # callers, injected so the respond head doesn't have to re-derive
            # hypergeometric comparisons end-to-end (155k-ep failed lesson).
            ctx.update(hybrid_claim_features(
                hand, actions, game.get_hand(other).size()))
        if self.trace_signals:
            self.signal_trace.append({
                "opp_call_rate": signals["opponent_call_rate"],
                "opp_revealed_bluff": signals["opponent_bluff_revealed"],
                "my_bluff_rate": my_bluff_rate,
                "turn": game.turn_count,
            })
        return ctx

    def decide_play(self, game: GameState, seat: int,
                    deterministic: bool = False):
        hand = game.get_hand(seat).cards
        state = self.encoder.encode(hand, self._context(game, seat))
        mask = build_legal_actions(hand, can_call=False, can_pass=False)
        idx = self.net.act(state, mask, deterministic=deterministic)
        cards, rank = decode_action_into(idx, hand)
        return idx, state, mask, cards, rank

    def decide_respond(self, game: GameState, seat: int,
                       deterministic: bool = False):
        """Returns (idx, state, mask, call: bool) or None if stuck.

        The mask is respond_only ({call, pass}): play actions are never
        offered here, so the policy can't waste probability mass on them and
        the stored log-prob matches the executed behavior. (The v1 bug:
        an unmasked 54-action respond mask made the agent sample a junk
        "play" index ~96% of the time, which the harness coerced into a
        call — 88% of v1 training games were draws.)
        """
        can_call, can_pass = legal_call_and_pass(game, seat)
        if not can_call:
            return None
        hand = game.get_hand(seat).cards
        state = self.encoder.encode(hand, self._context(game, seat))
        mask = build_legal_actions(hand, can_call=True, can_pass=can_pass,
                                   respond_only=True)
        idx = self.net.act(state, mask, deterministic=deterministic)
        if idx == PASS_ACTION and can_pass:
            return idx, state, mask, False
        # CALL (defensive fallback covers pass-sampled-while-illegal)
        return idx, state, mask, True


class RulePlayer:
    """Wraps a BotInterface bot to the same interface as NNPlayer."""

    def __init__(self, bot: BotInterface, name: str):
        self.bot = bot
        self.name = name

    def decide_play(self, game: GameState, seat: int, deterministic=False):
        hand = game.get_hand(seat)
        other = 1 - seat
        cards, rank = self.bot.decide_play(
            hand.cards,
            build_game_state(
                hand_size=hand.size(),
                opponent_hand_size=game.get_hand(other).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=game.turn_count,
                last_action=game.actions[-1] if game.actions else None,
                cards_played=game.get_cards_played(),
                hand=hand.cards,
                actions=game.actions,
            ),
        )
        return None, None, None, cards, rank

    def decide_respond(self, game: GameState, seat: int, deterministic=False):
        can_call, can_pass = legal_call_and_pass(game, seat)
        if not can_call:
            return None
        last = game.actions[-1]
        other = 1 - seat
        call = self.bot.decide_call(
            last,
            build_game_state(
                hand_size=game.get_hand(seat).size(),
                opponent_hand_size=game.get_hand(other).size(),
                pile_size=game.get_pile_size(),
                draw_pile_size=len(game.draw_pile),
                turn_number=game.turn_count,
                last_action=last,
                cards_played=game.get_cards_played(),
                hand=game.get_hand(seat).cards,
                actions=game.actions,
            ),
        )
        if not call and not can_pass:
            call = True  # forced call when draw pile empty (game-rules.md §2)
        return None, None, None, call


# ---------------------------------------------------------------------------
# Self-play game with reward tracking
# ---------------------------------------------------------------------------

class Transition:
    __slots__ = ("state", "action", "mask", "log_prob", "value", "reward",
                 "done")

    def __init__(self, state, action, mask, log_prob, value):
        self.state = state
        self.action = action
        self.mask = mask          # legal-action mask at decision time
        self.log_prob = log_prob
        self.value = value
        self.reward = 0.0
        self.done = False


def _obs(net: BluffNet, state: torch.Tensor, mask: torch.Tensor):
    """Masked forward pass: (masked_log_probs, value).

    Log-probs are computed over LEGAL actions only (illegal → -inf before
    softmax), so the stored behavior log-prob matches the sampling
    distribution and PPO's importance ratio is well-defined. Device-safe:
    inputs are moved to the net's device, outputs returned on CPU as floats.
    """
    device = next(net.parameters()).device
    logits, value = net(state.unsqueeze(0).to(device))
    logits = logits.squeeze(0)
    mask = mask.to(device)
    masked = logits.clone()
    masked[~mask] = float("-inf")
    log_probs = F.log_softmax(masked, dim=-1).cpu()
    return log_probs, value.squeeze().item()


def play_training_game(agent: NNPlayer, opponent, agent_seat: int,
                       max_turns: int = MAX_TURNS,
                       deterministic: bool = False) -> Tuple[List[Transition], int, dict]:
    """Play one self-play game.

    Returns (agent transitions, winner or -1, stats dict with turns /
    agent_plays / agent_bluffs for eval metrics).
    """
    game = GameState(num_players=2)
    game.deal(14)
    players = {agent_seat: agent, 1 - agent_seat: opponent}

    transitions: List[Transition] = []
    # Index of the agent's most recent play transition (for bluff shaping)
    pending_play: Optional[int] = None
    pending_qty = 0
    pending_pile_before = 0

    # Potential-based hand shaping: each transition rewards hand-size DECREASE
    # since the previous agent transition (Ng et al. 1999 — preserves the
    # optimal policy while densifying the signal). This makes pile-takes
    # (wrong calls, caught bluffs) immediately negative, which kills the
    # v2 "always-call + 1-card-honest" stalemate basin where draws produced
    # zero learning signal.
    agent_hand_prev = game.get_hand(agent_seat).size()

    def apply_hand_potential(t: Transition) -> None:
        nonlocal agent_hand_prev
        cur = game.get_hand(agent_seat).size()
        t.reward += HAND_POTENTIAL * (agent_hand_prev - cur)
        t.reward -= TIME_PENALTY
        agent_hand_prev = cur

    stats = {"turns": 0, "agent_plays": 0, "agent_bluffs": 0}

    def shape_bluff_outcome(t_idx: int, called: bool, caller_right: bool):
        t = transitions[t_idx]
        if called and caller_right:
            # Bluff caught: bluffer takes the pile (explicit penalty kept —
            # larger than the hand-potential cost, deters risky bluffs)
            t.reward += -BLUFF_CAUGHT_PENALTY * (pending_pile_before + pending_qty)
        elif not called and game.actions[-1].was_bluff:
            # Successful bluff (went through)
            t.reward += BLUFF_SUCCESS_BONUS * pending_qty

    turn = 0
    while not game.game_over and turn < max_turns:
        current = game.current_player
        player = players[current]

        # ---- play phase ----
        if current == agent_seat:
            idx, state, mask, cards, rank = agent.decide_play(game, agent_seat,
                                                              deterministic)
            log_probs, value = _obs(agent.net, state, mask)
            success, _ = game.play_cards(current, cards, rank)
            if not success:
                turn += 1
                continue
            t = Transition(state, idx, mask, float(log_probs[idx].item()),
                           value)
            t.reward += SHED_REWARD * len(cards)
            apply_hand_potential(t)
            transitions.append(t)
            pending_play = len(transitions) - 1
            pending_qty = len(cards)
            pending_pile_before = game.actions[-1].pile_size_before
            stats["agent_plays"] += 1
            if game.actions[-1].was_bluff:
                stats["agent_bluffs"] += 1
        else:
            _, _, _, cards, rank = player.decide_play(game, current)
            success, _ = game.play_cards(current, cards, rank)
            if not success:
                turn += 1
                continue

        if game.game_over:
            break

        # ---- respond phase (call or pass) ----
        responder = 1 - current
        resp = players[responder].decide_respond(game, responder)
        if resp is None:
            turn += 1
            continue
        idx_r, state_r, mask_r, call = resp

        if responder == agent_seat:
            log_probs, value = _obs(agent.net, state_r, mask_r)
            t = Transition(state_r, idx_r, mask_r, float(log_probs[idx_r].item()),
                           value)
            apply_hand_potential(t)
            transitions.append(t)
            # Respond-phase metrics (conditioning_curve.py claim-#2 evidence)
            stats["agent_responds"] = stats.get("agent_responds", 0) + 1
            stats["agent_calls"] = stats.get("agent_calls", 0) + int(call)
            # Snapshot for immediate pile-transfer credit below (v6.1: the
            # v5/v6 respond head was signal-deaf because a wrong call's
            # hand-delta landed on the NEXT transition — GAE had to ferry the
            # signal back through V(s'), so the respond head never got a
            # crisp gradient while the play head did. Charging the transfer
            # at the decision that caused it fixes the asymmetry.)
            resp_my_hand_before = game.get_hand(agent_seat).size()
            resp_opp_hand_before = game.get_hand(1 - agent_seat).size()
        else:
            resp_my_hand_before = None

        if call:
            success, _, action = game.call_bluff(responder)
            if success and action:
                # Rule-based bots that keep state (BayesianBot) observe reveals
                for p in players.values():
                    bot = getattr(p, "bot", None)
                    if bot is not None:
                        bot.observe_action(action, game.get_hand(1 - action.player).size())
                if pending_play is not None and current == agent_seat:
                    shape_bluff_outcome(pending_play, True,
                                        action.caller_was_right)
            if responder == agent_seat and resp_my_hand_before is not None:
                # v6.1 immediate pile-transfer credit at the respond decision:
                # wrong call → we take the pile (my hand grows → negative);
                # right call → bluffer takes it (their hand grows → positive).
                # Sum telescopes consistently with Φ = c·(opp_hand − my_hand).
                my_now = game.get_hand(agent_seat).size()
                opp_now = game.get_hand(1 - agent_seat).size()
                transitions[-1].reward += HAND_POTENTIAL * (resp_my_hand_before - my_now)
                transitions[-1].reward += HAND_POTENTIAL * (opp_now - resp_opp_hand_before)
                # Re-baseline the play-side potential so the transfer is not
                # double-counted by the next transition's generic delta.
                agent_hand_prev = my_now
        else:
            game.pass_turn(passer=responder)
            if pending_play is not None and current == agent_seat:
                shape_bluff_outcome(pending_play, False, False)

        pending_play = None
        turn += 1

    stats["turns"] = turn
    # End-state metric for benchmarks (shed-race margin: how many cards the
    # opponent had left when the game ended — 0 means they shed out).
    stats["opp_hand_end"] = game.get_hand(1 - agent_seat).size()
    stats["agent_hand_end"] = game.get_hand(agent_seat).size()

    # ---- terminal reward ----
    winner = game.winner if game.game_over else -1
    if transitions:
        # Flush any remaining hand-size delta (e.g. pile taken on the final
        # call has no subsequent transition to carry it)
        apply_hand_potential(transitions[-1])
        if winner == agent_seat:
            transitions[-1].reward += WIN_REWARD
        elif winner == 1 - agent_seat:
            transitions[-1].reward += LOSS_REWARD
        else:
            # Draws get a NEGATIVE terminal reward: the v2 mirror-match
            # collapse showed that zero-reward draws create a gradient-free
            # stalemate basin (5954/5955 mirror games drawn).
            transitions[-1].reward += DRAW_REWARD
        transitions[-1].done = True
    return transitions, winner, stats


# ---------------------------------------------------------------------------
# PPO core
# ---------------------------------------------------------------------------

def compute_gae(transitions: List[Transition], gamma: float, lam: float):
    """GAE advantages + discounted returns. done flags break bootstrapping."""
    n = len(transitions)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(n)):
        tr = transitions[t]
        if tr.done:
            gae = 0.0
            next_value = 0.0
        else:
            nxt = transitions[t + 1] if t + 1 < n else None
            next_value = nxt.value if nxt is not None else 0.0
        delta = tr.reward + gamma * next_value * (0 if tr.done else 1) - tr.value
        gae = delta + gamma * lam * (0 if tr.done else gae)
        advantages[t] = gae
        returns[t] = gae + tr.value
    return advantages, returns


def compute_gae_rnad(transitions: List[Transition], v_bar: torch.Tensor,
                     gamma: float, lam: float):
    """GAE with population baseline V-bar instead of critic value.

    delta = r + gamma * Vbar_next * (1-done) - Vbar
    A     = delta + gamma * lam * (1-done) * A_next   (reversed)
    returns = A + Vbar

    v_bar: (T,) CPU tensor of per-transition population baseline values.
    """
    n = len(transitions)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    vbar_next = 0.0
    for t in reversed(range(n)):
        tr = transitions[t]
        if tr.done:
            gae = 0.0
            vbar_next = 0.0
        else:
            vbar_next = v_bar[t + 1].item() if t + 1 < n else 0.0
        vbar_t = v_bar[t].item()
        delta = tr.reward + gamma * vbar_next * (0 if tr.done else 1) - vbar_t
        gae = delta + gamma * lam * (0 if tr.done else gae)
        advantages[t] = gae
        returns[t] = gae + vbar_t
    return advantages, returns


def ppo_update(net: "BluffNet | BluffNetXL", optimizer: torch.optim.Optimizer,
               transitions: List[Transition], clip_eps: float = 0.2,
               epochs: int = 4, batch_size: int = 256, gamma: float = 0.99,
               lam: float = 0.95, ent_coef: float = 0.02, vf_coef: float = 0.5,
               max_grad_norm: float = 0.5) -> dict:
    advantages, returns = compute_gae(transitions, gamma, lam)
    device = next(net.parameters()).device
    states = torch.stack([t.state for t in transitions]).to(device)
    actions = torch.tensor([t.action for t in transitions],
                           dtype=torch.long, device=device)
    masks = torch.stack([t.mask for t in transitions]).to(device)
    old_logp = torch.tensor([t.log_prob for t in transitions], device=device)
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)
    # Advantage normalization (standard PPO)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(transitions)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            logits, values = net(states[mb])
            # Masked log-softmax — must match the sampling distribution
            # (illegal actions at -inf) or the PPO ratio is ill-defined.
            masked = logits.clone()
            masked[~masks[mb]] = float("-inf")
            logp_all = F.log_softmax(masked, dim=-1)
            new_logp = logp_all.gather(1, actions[mb].unsqueeze(1)).squeeze(1)
            # Entropy over legal actions only. exp(-inf)=0 and 0*-inf=NaN,
            # so zero out illegal entries' log-probs before the product.
            probs = logp_all.exp()
            entropy = -(probs * logp_all.masked_fill(~masks[mb], 0.0)).sum(-1).mean()

            ratio = (new_logp - old_logp[mb]).exp()
            mb_adv = adv[mb]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), ret[mb])

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            for k, v in (("policy_loss", policy_loss.item()),
                         ("value_loss", value_loss.item()),
                         ("entropy", entropy.item())):
                metrics[k] += v
            updates += 1

    return {k: v / max(1, updates) for k, v in metrics.items()}


def rnad_update(net, optimizer, transitions, pool: OpponentPool,
                clip_eps=0.2, epochs=4, batch_size=256, gamma=0.99,
                lam=0.95, ent_coef=0.02, vf_coef=0.5, kl_coef=0.01,
                v_bar_ema: Optional[float] = None,
                max_grad_norm: float = 0.5) -> dict:
    """R-NaD style update: PPO + population baseline GAE + KL regularization.

    Structurally mirrors ppo_update (same minibatch loop, masking, clipping)
    but uses:
      1. V-bar (population baseline) instead of critic value for GAE.
      2. KL(softmax(logits) || softmax(pool_centroid_logits)) as a regularizer.

    Returns (metrics_dict, updated_v_bar_ema).
    """
    device = next(net.parameters()).device

    # --- V-bar computation (population baseline) ---
    states_dev = torch.stack([t.state for t in transitions]).to(device)
    with torch.no_grad():
        v_bar_raw = pool.population_baseline(states_dev, net)  # (T,) CPU

    # EMA smoothing of V-bar across updates (Oracle risk #1 mitigation)
    alpha = 0.1  # EMA weight for current observation
    if v_bar_ema is None:
        v_bar_ema = v_bar_raw.mean().item()
    else:
        v_bar_ema = alpha * v_bar_raw.mean().item() + (1 - alpha) * v_bar_ema

    # --- GAE with V-bar baseline ---
    advantages, returns = compute_gae_rnad(transitions, v_bar_raw, gamma, lam)

    # --- Centroid logits for KL ---
    centroid_logits = pool.centroid_logits(states_dev)  # (action_dim,) CPU
    has_kl = centroid_logits is not None and kl_coef > 0

    actions = torch.tensor([t.action for t in transitions],
                           dtype=torch.long, device=device)
    masks = torch.stack([t.mask for t in transitions]).to(device)
    old_logp = torch.tensor([t.log_prob for t in transitions], device=device)
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    if has_kl:
        ref_logits = centroid_logits.to(device)

    n = len(transitions)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
    updates = 0

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            logits, values = net(states_dev[mb])
            masked = logits.clone()
            masked[~masks[mb]] = float("-inf")
            logp_all = F.log_softmax(masked, dim=-1)
            new_logp = logp_all.gather(1, actions[mb].unsqueeze(1)).squeeze(1)

            probs = logp_all.exp()
            entropy = -(probs * logp_all.masked_fill(~masks[mb], 0.0)).sum(-1).mean()

            ratio = (new_logp - old_logp[mb]).exp()
            mb_adv = adv[mb]
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(-1), ret[mb])

            # KL(π || π_pool): KL(softmax(logits) || softmax(ref_logits))
            # averaged over minibatch — no grad on ref.
            # Masked to legal actions only to avoid 0*(-inf)=NaN on illegal.
            kl_term = torch.tensor(0.0, device=device)
            if has_kl:
                pi = probs  # (mb_size, action_dim), already softmax
                ref_prob = F.softmax(ref_logits, dim=-1)  # (action_dim,)
                legal = masks[mb]  # (mb_size, action_dim) bool
                # log pi over legal actions (illegal entries zeroed out)
                log_pi = logp_all.masked_fill(~legal, 0.0)
                # log ref over legal actions (illegal entries zeroed out)
                log_ref = ref_prob.log().unsqueeze(0).expand_as(pi).masked_fill(~legal, 0.0)
                # KL = sum_legal( pi * (log pi - log ref) )
                kl_per_action = pi * (log_pi - log_ref)
                kl_term = kl_per_action.sum(-1).mean()  # scalar

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + kl_coef * kl_term

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            for k, v in (("policy_loss", policy_loss.item()),
                         ("value_loss", value_loss.item()),
                         ("entropy", entropy.item()),
                         ("kl", kl_term.item())):
                metrics[k] += v
            updates += 1

    avg_metrics = {k: v / max(1, updates) for k, v in metrics.items()}
    return avg_metrics, v_bar_ema

class OpponentPool:
    """Snapshot pool per docs/neural-network.md (pool_size=30)."""

    def __init__(self, pool_size: int = 30):
        self.pool_size = pool_size
        self.snaps: List[NNPlayer] = []

    def add(self, net: BluffNet):
        snap = NNPlayer(copy.deepcopy(net))
        snap.net.eval()
        self.snaps.append(snap)
        if len(self.snaps) > self.pool_size:
            self.snaps.pop(0)

    def population_baseline(self, states: torch.Tensor,
                            current_net: "BluffNet | BluffNetXL") -> torch.Tensor:
        """Mean critic-head value across all pool snapshots + current net.

        Returns a (B,) tensor on CPU. Pool snapshots live on CPU — they are
        moved to the current net's device temporarily during the forward pass
        and then freed.  Simple, correct, no caching.
        """
        device = next(current_net.parameters()).device
        B = states.shape[0]
        # Collect current-net values (already on device)
        with torch.no_grad():
            _, vals = current_net(states.to(device))
            acc = vals.squeeze(-1).cpu()  # (B,)
        count = 1
        # Add each pool snapshot
        for snap in self.snaps:
            snap_dev = next(snap.net.parameters()).device
            with torch.no_grad():
                _, vals = snap.net(states.to(snap_dev))
                acc = acc + vals.squeeze(-1).cpu()
            count += 1
        return acc / count

    def centroid_logits(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        """Weighted-mean actor logits across pool snapshots (no current net).

        Args:
            states: (B, state_dim) tensor — used to run forward passes.
        Returns:
            (action_dim,) CPU tensor — the population centroid KL reference.
            Uniform weight over snapshots; current net excluded.
        """
        if not self.snaps:
            return None
        acc = None
        for snap in self.snaps:
            sdev = next(snap.net.parameters()).device
            with torch.no_grad():
                logits, _ = snap.net(states.to(sdev))
                l_cpu = logits.mean(dim=0).cpu()  # (action_dim,)
                if acc is None:
                    acc = l_cpu
                else:
                    acc = acc + l_cpu
        return acc / len(self.snaps)

    def sample(self, latest: NNPlayer):
        """Opponent sampling with a scripted-opponent floor (40%).

        v3 lesson: with only 10% rule bots the policy overfits to NN-like
        grindy opponents — optimal vs Random/mirrors, 0% vs Honest (which
        punishes wrong calls by shedding 4/turn while you re-take the pile).
        Charlesworth fixes exactly this with scripted-opponent curriculum:
        keep a permanent floor of simple, DIFFERENT opponents so the policy
        must stay robust to strategic archetypes, not just to itself.
        """
        # v6 mix (ADR 2026-09-10 "league roster"): 40% scripted archetypes
        # (Charlesworth league floor) + 45% NN pool + 15% mirror. v4/v5 lesson:
        # with only Random+Honest scripted, the policy overfits to two
        # archetypes and loses to card-counters; the benchmark sweeps all four
        # baselines, so train against all four.
        r = random.random()
        if r < 0.10:
            return RulePlayer(RandomBot(), "Random")
        if r < 0.25:
            return RulePlayer(HonestBot(), "Honest")
        if r < 0.40:
            return RulePlayer(CardCountBot(), "CardCount")
        if r < 0.55:
            return RulePlayer(BayesianBot(), "Bayesian")
        if r < 0.70 or not self.snaps:
            return latest
        return random.choice(self.snaps)

def evaluate(net: BluffNet, opponent: RulePlayer, num_games: int,
             encoder: StateEncoder) -> tuple:
    """Deterministic-policy evaluation against a fixed opponent.

    Returns (win_rate, bluff_rate). Bluff rate is a first-class paper metric
    (Dewey 2025 / Ahle 2022 track learned bluff frequencies over training;
    Yeung 2008 gives the theoretical equilibrium they should approach).
    """
    agent = NNPlayer(net, encoder)
    wins = 0
    plays = bluffs = 0
    for i in range(num_games):
        _, winner, stats = play_training_game(agent, opponent, agent_seat=i % 2,
                                              deterministic=True)
        if winner == i % 2:
            wins += 1
        plays += stats["agent_plays"]
        bluffs += stats["agent_bluffs"]
    return wins / max(1, num_games), bluffs / max(1, plays)


def compute_exploitability(net: BluffNet, pool: OpponentPool,
                           encoder: StateEncoder,
                           num_games: int = 50) -> float:
    """Max over pool members of (member-vs-net winrate) - 0.5.

    Lightweight proxy for exploitability: the worst-case win-rate advantage
    any pool member has against the current net.  At Nash equilibrium this
    is 0.  We only evaluate NN pool snapshots (scripted bots are always
    available but this metric tracks convergence of the learned population).
    """
    if not pool.snaps:
        return 0.0
    max_wr = 0.0
    for snap_player in pool.snaps:
        # snap_player.net plays as agent_seat=0; net plays as opponent
        # We want: how well does snap beat net?  So snap is agent (seat 0),
        # net is opponent (seat 1).
        def _make_opponent(n=net, e=encoder):
            """Closure wrapping the current net as a RulePlayer-like opponent."""
            class _NNOpponent:
                def __init__(self, net, enc):
                    self.nn_player = NNPlayer(net, enc)
                def decide_play(self, game, seat, deterministic=False):
                    return self.nn_player.decide_play(game, seat, deterministic)
                def decide_respond(self, game, seat, deterministic=False):
                    return self.nn_player.decide_respond(game, seat, deterministic)
            return _NNOpponent(n, e)

        opp = _make_opponent()
        wins = 0
        for i in range(num_games):
            # snap_player is agent (seat 0), net-based opp is seat 1
            _, winner, _ = play_training_game(
                snap_player, opp, agent_seat=0, deterministic=True)
            if winner == 0:
                wins += 1
        wr = wins / max(1, num_games)
        max_wr = max(max_wr, wr)
    return max_wr - 0.5


def save_checkpoint(net: BluffNet, path: str,
                    state_dim: Optional[int] = None,
                    action_dim: int = 54):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if state_dim is None:
        # Infer input dim from the net itself (v7=39, v8 hybrid=42)
        state_dim = net.shared[0].in_features
    # Always save CPU tensors so checkpoints load on CPU-only machines
    # (web server, CI, other agents' laptops).
    torch.save({
        "state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "state_dim": state_dim,
        "action_dim": action_dim,
    }, path)


def load_checkpoint(path: str) -> BluffNet:
    ckpt = torch.load(path, weights_only=True)
    net = BluffNet(state_dim=ckpt["state_dim"], action_dim=ckpt["action_dim"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(episodes: int = 2000, output: str = "nn/checkpoints/final.pt",
          batch_size: int = 2048, minibatch: int = 256, epochs: int = 4,
          lr: float = 3e-4, clip_eps: float = 0.2, gamma: float = 0.99,
          lam: float = 0.95, ent_coef: float = 0.05,
          ent_coef_end: float = 0.01,
          eval_interval: int = 500, eval_games: int = 40,
          pool_size: int = 30, seed: int = 0, log_jsonl: Optional[str] = None,
          device: str = "auto", ablate_opponent_features: bool = False,
          init_from: Optional[str] = None, hybrid: bool = False,
          use_rnad: bool = False, kl_coef: float = 0.01,
          kl_coef_end: float = 0.001):
    """PPO training with linear entropy annealing (ADR 2026-08-11: 0.05→0.01).

    ent_coef is the START value; it anneals linearly to ent_coef_end across
    the run. High early entropy explores bluffing; the low tail lets the
    policy converge (Patwa 2026, SplendorRL anneal schedules).

    use_rnad: when True, switches the update to R-NaD (population baseline
    GAE + KL regularization). When False (default), the code path is
    byte-identical to pure PPO.

    kl_coef is the START value for KL regularization; it anneals linearly
    to kl_coef_end across the run (same pattern as ent_coef).

    device: "auto" uses CUDA when available. NOTE (ADR 2026-09-10): the
    BluffNet backbone is tiny (~100k params) and rollout collection is
    single-state, so GPU mainly accelerates the PPO update phase; the pure-
    Python game loop dominates runtime. Measure — don't assume speedup.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # Claim-#2 ablation control: replace the 4 opponent-conditioning
    # features with constant population priors (v4 marginals, zero opponent
    # information) — the fair v4-comparable control run.
    encoder = StateEncoder(use_bayesian_features=not ablate_opponent_features,
                           hybrid_features=hybrid)
    state_dim = encoder.state_dim
    if init_from:
        # Warm start (ADR 2026-09-10 "training resilience"): begin from prior
        # weights instead of scratch — used for killed-run auto-resume
        # (v7_latest) and curriculum-style runs. Three training runs (v4/v6/
        # v6.1) died silently when the spawning agent session ended; resume
        # capability is the cheap fix. Dim mismatch (39→42 hybrid) = fresh
        # net: warm-start is only valid within an architecture.
        src = load_checkpoint(init_from)
        if src.shared[0].in_features == state_dim:
            net = src.to(dev)
            net.train()
            print(f"Warm-started weights from {init_from}", flush=True)
        else:
            print(f"Warm-start skipped: {init_from} is "
                  f"{src.shared[0].in_features}-dim, run needs {state_dim} "
                  f"(hybrid={hybrid}). Training from scratch.", flush=True)
            net = BluffNet(state_dim=state_dim).to(dev)
    else:
        net = BluffNet(state_dim=state_dim).to(dev)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    pool = OpponentPool(pool_size)
    pool.add(net)

    tee = None
    if log_jsonl:
        from analysis.logger import TeeLogger
        tee = TeeLogger(log_jsonl, source="terminal")

    buffer: List[Transition] = []
    latest = NNPlayer(net, encoder)
    episode = 0
    wins_recent, games_recent = 0, 0
    best_eval = -1.0
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    last_metrics = None
    v_bar_ema: Optional[float] = None  # EMA of V-bar across updates (R-NaD)

    mode = "R-NaD" if use_rnad else "PPO"
    print(f"Training {mode}: episodes={episodes} batch={batch_size} "
          f"ent_coef={ent_coef} lr={lr}"
          + (f" kl_coef={kl_coef}→{kl_coef_end}" if use_rnad else ""))
    print(f"Output: {output}")

    while episode < episodes:
        episode += 1
        agent_seat = episode % 2  # alternate seats to avoid position bias
        opponent = pool.sample(latest)
        agent = NNPlayer(net, encoder)

        logger = tee.new_game("PureNN", getattr(opponent, "name", "snapshot")) \
            if tee else None

        transitions, winner, stats = play_training_game(agent, opponent,
                                                         agent_seat)
        buffer.extend(transitions)

        games_recent += 1
        if winner == agent_seat:
            wins_recent += 1

        if logger is not None:
            # Per-action logging during training is too slow; log summary only
            logger.log_game_end_simple(winner, agent_seat,
                                       num_turns=stats["turns"])

        # ---- update when buffer full ----
        if len(buffer) >= batch_size:
            ent_now = ent_coef + (ent_coef_end - ent_coef) * min(
                1.0, episode / max(1, episodes))
            if use_rnad:
                kl_now = kl_coef + (kl_coef_end - kl_coef) * min(
                    1.0, episode / max(1, episodes))
                metrics, v_bar_ema = rnad_update(
                    net, optimizer, buffer, pool, clip_eps, epochs,
                    minibatch, gamma, lam, ent_now, kl_coef=kl_now,
                    v_bar_ema=v_bar_ema)
            else:
                metrics = ppo_update(net, optimizer, buffer, clip_eps, epochs,
                                     minibatch, gamma, lam, ent_now)
            buffer = []
            pool.add(net)
            last_metrics = metrics

        # ---- eval + checkpoint ----
        if episode % eval_interval == 0 or episode == episodes:
            # v6: eval against the full archetype roster (train what you test)
            evals = {}
            for opp_name, opp_cls in (("Random", RandomBot), ("Honest", HonestBot),
                                      ("CardCount", CardCountBot),
                                      ("Bayesian", BayesianBot)):
                wr, bl = evaluate(net, RulePlayer(opp_cls(), opp_name),
                                  eval_games, encoder)
                evals[opp_name] = (wr, bl)
            metrics_str = (f" | pl={metrics['policy_loss']:.3f} "
                           f"vl={metrics['value_loss']:.3f} "
                           f"ent={metrics['entropy']:.3f}"
                           + (f" kl={metrics['kl']:.4f}"
                              if use_rnad and 'kl' in metrics else "")
                           ) if last_metrics else ""
            per_opp = " ".join(f"vs {n}={w:.0%}(b{b:.0%})"
                               for n, (w, b) in evals.items())
            # R-NaD exploitability metric (lightweight — only when enabled)
            expl_str = ""
            expl_val = 0.0
            if use_rnad:
                expl_val = compute_exploitability(net, pool, encoder,
                                                  num_games=min(eval_games, 50))
                expl_str = f" expl={expl_val:.3f}"
            print(f"[ep {episode:>6d}] {per_opp} | recent={wins_recent}/"
                  f"{games_recent} | pool={len(pool.snaps)}{metrics_str}{expl_str}",
                  flush=True)
            last_metrics = metrics if len(buffer) == 0 else last_metrics
            wins_recent, games_recent = 0, 0

            score = sum(w for w, _ in evals.values())
            # Crash-resilience + eval-curve artifacts (ADR 2026-09-10
            # "training resilience"):
            # - _latest.pt EVERY eval interval — a killed run never loses more
            #   than eval_interval episodes (best-only saving cost us v6.1).
            # - _eval.jsonl — machine-readable eval curve; stdout used to be
            #   the only record and died with the process.
            eval_record = {
                "record": "eval",
                "episode": episode,
                "opponents": {n: [round(w, 4), round(b, 4)]
                              for n, (w, b) in evals.items()},
                "best_combined": round(max(best_eval, score), 4),
                "timestamp": time.time(),
            }
            if use_rnad:
                eval_record["exploitability"] = round(expl_val, 4)
            save_checkpoint(net, output.replace(".pt", "_latest.pt"))
            with open(output.replace(".pt", "_eval.jsonl"), "a",
                      encoding="utf-8") as ef:
                ef.write(json.dumps(eval_record) + "\n")
            if score > best_eval:
                best_eval = score
                save_checkpoint(net, output)
                save_checkpoint(net, output.replace(".pt", "_best.pt"))

    save_checkpoint(net, output)
    print(f"Training complete. Final checkpoint: {output}")
    print(f"Best combined eval winrate: {best_eval:.2f}")


def main():
    parser = argparse.ArgumentParser(description="PPO training for BluffNet")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--output", type=str, default="nn/checkpoints/final.pt")
    parser.add_argument("--ent-coef", type=float, default=0.05,
                        help="Entropy bonus at start (anneals to --ent-coef-end)")
    parser.add_argument("--ent-coef-end", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        help='"auto" | "cpu" | "cuda"')
    parser.add_argument("--log", type=str, default=None,
                        help="Optional JSONL path for game summaries")
    parser.add_argument("--ablate-opponent-features", action="store_true",
                        help="Claim-#2 control: replace the 4 opponent-"
                             "conditioning features with constant population "
                             "priors (v4 marginals, zero opponent info)")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Warm-start from a checkpoint (auto-resume / "
                             "curriculum). Weights only; episode count "
                             "restarts at 0.")
    parser.add_argument("--hybrid", action="store_true",
                        help="v8: append 3 Bayesian pool-math features "
                             "(39→42 dims) — the HybridBot thesis experiment "
                             "(research claim #2).")
    parser.add_argument("--use-rnad", action="store_true", default=False,
                        help="Enable R-NaD update (population baseline GAE + "
                             "KL regularization). Default off — pure PPO path "
                             "is byte-identical when disabled.")
    parser.add_argument("--kl-coef", type=float, default=0.01,
                        help="KL regularization coefficient at start "
                             "(anneals to --kl-coef-end, R-NaD only)")
    parser.add_argument("--kl-coef-end", type=float, default=0.001,
                        help="KL regularization coefficient at end "
                             "(R-NaD only)")
    args = parser.parse_args()

    train(episodes=args.episodes, output=args.output, ent_coef=args.ent_coef,
          ent_coef_end=args.ent_coef_end,
          lr=args.lr, eval_interval=args.eval_interval,
          eval_games=args.eval_games, batch_size=args.batch_size,
          seed=args.seed, log_jsonl=args.log, device=args.device,
          ablate_opponent_features=args.ablate_opponent_features,
          init_from=args.init_from, hybrid=args.hybrid,
          use_rnad=args.use_rnad, kl_coef=args.kl_coef,
          kl_coef_end=args.kl_coef_end)


if __name__ == "__main__":
    main()
