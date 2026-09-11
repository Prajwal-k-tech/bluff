"""
BluffNet-XL PPO Reinforcement Learning Fine-Tuning.
Directly implements user mandate:
  "I guess we should do reinforcement learning where we reward good moves that get
   rid of many cards, punish getting caught bluffing etc. Train it logically and
   well gang dont do random stuff, try to take inspiration / reasoning from
   academic papers see if they have good data sets"

Architecture & Method:
1. Warm-start initialization from converged supervised BluffNet-XL (1.35M weights).
2. Generalized Advantage Estimation (GAE-Lambda: gamma=0.99, lambda=0.95).
3. Fine-grained game-theoretic reward shaping:
   - Multi-card clearance: +0.50 * k cards shed
   - Successful undetected bluff: +1.00 * k bonus
   - Caught bluff penalty: -1.00 * (pile + k)
   - Correct challenge: +0.50 * pile
   - Erroneous challenge: -0.80 * pile
   - Terminal win: +10.0 / loss: -10.0 / draw-lock: -2.0
4. League opponent sampling (20 synthetic personas + 4 canonical baselines + historical self snapshots).
5. Conservative PPO fine-tuning (lr=3e-5, clip=0.20, ent_coef=0.01) on RTX 3050 GPU.
6. Checkpoint output to nn/checkpoints/bluffnet_xl_ppo.pt.
"""

import os
import sys
import copy
import time
import math
import random
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import GameState, Action
from bots.base import pending_claims_from_actions, pool_by_rank
from bots.random_bot import RandomBot
from bots.honest_bot import HonestBot
from bots.cardcount_bot import CardCountBot
from bots.bayesian_bot import BayesianBot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot
from nn.model import (
    BluffNetXL,
    ACTION_DIM,
    CALL_ACTION,
    PASS_ACTION,
    action_index,
    decode_action_into,
    build_legal_actions,
    legal_call_and_pass,
)
from nn.state_encoder import StateEncoder, STATE_DIM, opponent_signals_from_actions


# ---------------------------------------------------------------------------
# Transition & Trajectory Storage
# ---------------------------------------------------------------------------

class RLTransition:
    def __init__(
        self,
        state: torch.Tensor,
        action: int,
        mask: torch.Tensor,
        log_prob: float,
        value: float,
    ):
        self.state = state
        self.action = action
        self.mask = mask
        self.log_prob = log_prob
        self.value = value
        self.reward: float = 0.0


def compute_gae(
    transitions: List[RLTransition],
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation (GAE-Lambda) and returns."""
    advantages = [0.0] * len(transitions)
    returns = [0.0] * len(transitions)
    last_adv = 0.0

    for t in reversed(range(len(transitions))):
        if t == len(transitions) - 1:
            next_val = last_value
        else:
            next_val = transitions[t + 1].value

        delta = transitions[t].reward + gamma * next_val - transitions[t].value
        advantages[t] = delta + gamma * lam * last_adv
        last_adv = advantages[t]
        returns[t] = advantages[t] + transitions[t].value

    return advantages, returns


# ---------------------------------------------------------------------------
# RL Player Wrapper for BluffNetXL
# ---------------------------------------------------------------------------

class XLRLPlayer:
    def __init__(self, net: BluffNetXL, device: torch.device):
        self.net = net
        self.device = device
        self.encoder = StateEncoder()

    def _context(self, game: GameState, seat: int) -> dict:
        hand = game.get_hand(seat)
        other = 1 - seat
        can_call, can_pass = legal_call_and_pass(game, seat)
        actions = game.actions

        my_plays = [a for a in actions if a.player == seat]
        bluffs = [a for a in my_plays if a.was_bluff]
        my_bluff_rate = len(bluffs) / max(1, len(my_plays))
        caught = sum(1 for a in bluffs if a.bluff_called and a.caller_was_right)
        my_bluff_success = 1.0 - caught / max(1, len(bluffs))

        pending = pending_claims_from_actions(actions)
        remaining = pool_by_rank(hand.cards, pending)
        signals = opponent_signals_from_actions(actions, seat)
        signals["my_bluff_rate"] = my_bluff_rate
        signals["my_bluff_success_rate"] = my_bluff_success

        return {
            "opponent_hand_size": game.get_hand(other).size(),
            "pile_size": game.get_pile_size(),
            "draw_pile_size": len(game.draw_pile),
            "turn_number": game.turn_count,
            "can_pass": can_pass,
            "last_action": actions[-1] if actions else None,
            "cards_remaining": remaining,
            **signals,
        }

    def act_play(self, game: GameState, seat: int) -> Tuple[int, torch.Tensor, torch.Tensor, float, float, List[Card], Rank]:
        hand = game.get_hand(seat).cards
        ctx = self._context(game, seat)
        state = self.encoder.encode(hand, ctx).to(self.device)
        mask = build_legal_actions(hand, can_call=False, can_pass=False, respond_only=False).to(self.device)

        with torch.no_grad():
            logits, value = self.net(state.unsqueeze(0))
            logits = logits.squeeze(0)
            val_item = value.squeeze().item()
            masked_logits = logits.clone()
            masked_logits[~mask] = float("-inf")
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        cards, rank = decode_action_into(action, hand)
        return action, state.cpu(), mask.cpu(), log_prob, val_item, cards, rank

    def act_respond(self, game: GameState, seat: int) -> Tuple[int, torch.Tensor, torch.Tensor, float, float, bool]:
        hand = game.get_hand(seat).cards
        ctx = self._context(game, seat)
        can_call, can_pass = legal_call_and_pass(game, seat)
        state = self.encoder.encode(hand, ctx).to(self.device)
        mask = build_legal_actions(hand, can_call=can_call, can_pass=can_pass, respond_only=True).to(self.device)

        with torch.no_grad():
            logits, value = self.net(state.unsqueeze(0))
            logits = logits.squeeze(0)
            val_item = value.squeeze().item()
            masked_logits = logits.clone()
            masked_logits[~mask] = float("-inf")
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()

        should_call = (action == CALL_ACTION)
        return action, state.cpu(), mask.cpu(), log_prob, val_item, should_call


# ---------------------------------------------------------------------------
# Episode Rollout with Reward Shaping
# ---------------------------------------------------------------------------

def play_rl_episode(
    agent: XLRLPlayer,
    opponent,
    agent_seat: int,
    max_turns: int = 100,
) -> Tuple[List[RLTransition], int, dict]:
    """
    Executes one rollout episode with shaped reward signals:
    - Multi-card clearance reward: +0.5 * k
    - Successful bluff bonus: +1.0 * k
    - Caught bluff penalty: -1.0 * (pile + k)
    - Correct call reward: +0.5 * pile
    - Erroneous call penalty: -0.8 * pile
    - Terminal outcome: +10.0 (win) / -10.0 (loss) / -2.0 (draw-lock)
    """
    game = GameState(num_players=2)
    game.deal(14)
    if hasattr(opponent, "reset"):
        opponent.reset()

    transitions: List[RLTransition] = []
    pending_play_idx: Optional[int] = None
    pending_cards_count = 0
    pending_pile_before = 0

    stats = {"turns": 0, "agent_bluffs": 0, "agent_calls": 0, "bluffs_caught": 0, "bluffs_won": 0}

    turn = 0
    while not game.game_over and turn < max_turns:
        curr = game.current_player
        other = 1 - curr

        if curr == agent_seat:
            # Agent's play phase
            act, s, m, lp, val, cards, rank = agent.act_play(game, curr)
            if not cards:
                turn += 1
                continue

            success, _ = game.play_cards(curr, cards, rank)
            if not success:
                turn += 1
                continue

            trans = RLTransition(s, act, m, lp, val)
            # Reward multi-card clearance (+0.50 * k)
            k = len(cards)
            trans.reward += 0.50 * k
            transitions.append(trans)
            pending_play_idx = len(transitions) - 1
            pending_cards_count = k
            pending_pile_before = game.actions[-1].pile_size_before

            if game.actions[-1].was_bluff:
                stats["agent_bluffs"] += 1

            if game.game_over:
                break

            # Opponent responds
            can_pass = len(game.draw_pile) > 0
            if hasattr(opponent, "decide_call"):
                opp_call = opponent.decide_call(
                    game.actions[-1],
                    {
                        "hand_size": game.get_hand(other).size(),
                        "opponent_hand_size": game.get_hand(curr).size(),
                        "pile_size": game.get_pile_size(),
                        "draw_pile_size": len(game.draw_pile),
                        "turn_number": turn,
                        "last_action": game.actions[-1],
                        "cards_played": game.get_cards_played(),
                        "hand": game.get_hand(other).cards,
                        "actions": game.actions,
                    },
                )
            else:
                opp_call = False

            if not can_pass:
                opp_call = True

            if opp_call:
                succ, _, act_obj = game.call_bluff(other)
                if succ and act_obj:
                    if hasattr(opponent, "observe_action"):
                        opponent.observe_action(act_obj, game.get_hand(curr).size())
                    if act_obj.caller_was_right:
                        # Agent's bluff was caught! Penalty = -1.0 * (pile + k)
                        if pending_play_idx is not None:
                            transitions[pending_play_idx].reward -= 1.0 * (pending_pile_before + pending_cards_count)
                            stats["bluffs_caught"] += 1
                    else:
                        # Agent played honest, opponent called wrongly!
                        if pending_play_idx is not None:
                            transitions[pending_play_idx].reward += 0.50 * pending_pile_before
            else:
                if game.can_pass():
                    game.pass_turn(passer=other)
                    # Successful bluff passed unchallenged!
                    if game.actions[-1].was_bluff and pending_play_idx is not None:
                        transitions[pending_play_idx].reward += 1.00 * pending_cards_count
                        stats["bluffs_won"] += 1

        else:
            # Opponent's play phase
            if hasattr(opponent, "decide_play"):
                opp_cards, opp_rank = opponent.decide_play(
                    game.get_hand(curr).cards,
                    {
                        "hand_size": game.get_hand(curr).size(),
                        "opponent_hand_size": game.get_hand(other).size(),
                        "pile_size": game.get_pile_size(),
                        "draw_pile_size": len(game.draw_pile),
                        "turn_number": turn,
                        "last_action": game.actions[-1] if game.actions else None,
                        "cards_played": game.get_cards_played(),
                        "actions": game.actions,
                    },
                )
            else:
                opp_cards, opp_rank = [game.get_hand(curr).cards[0]], game.get_hand(curr).cards[0].rank

            if not opp_cards:
                turn += 1
                continue

            success, _ = game.play_cards(curr, opp_cards, opp_rank)
            if not success or game.game_over:
                break

            # Agent responds (CALL or PASS)
            act, s, m, lp, val, should_call = agent.act_respond(game, other)
            trans = RLTransition(s, act, m, lp, val)
            stats["agent_calls"] += int(should_call)

            pile_at_call = game.get_pile_size()
            if should_call:
                succ, _, act_obj = game.call_bluff(other)
                if succ and act_obj:
                    if act_obj.caller_was_right:
                        # Correct challenge: agent caught opponent bluffing! (+0.5 * pile)
                        trans.reward += 0.50 * max(1, pile_at_call)
                    else:
                        # Wrong challenge: agent took pile! (-0.8 * pile)
                        trans.reward -= 0.80 * max(1, pile_at_call)
            else:
                if game.can_pass():
                    game.pass_turn(passer=other)
                else:
                    # Forced call
                    succ, _, act_obj = game.call_bluff(other)
                    if succ and act_obj:
                        if act_obj.caller_was_right:
                            trans.reward += 0.50 * max(1, pile_at_call)
                        else:
                            trans.reward -= 0.80 * max(1, pile_at_call)

            transitions.append(trans)

        turn += 1

    # Terminal Outcome Rewards
    stats["turns"] = turn
    winner = game.winner
    if winner == agent_seat:
        outcome = 1
        term_rew = 10.0
    elif winner == (1 - agent_seat):
        outcome = -1
        term_rew = -10.0
    else:
        outcome = 0
        term_rew = -2.0  # draw-lock penalty to discourage stalemates

    if transitions:
        transitions[-1].reward += term_rew

    return transitions, outcome, stats


# ---------------------------------------------------------------------------
# PPO Policy Optimization Routine
# ---------------------------------------------------------------------------

def ppo_update_xl(
    net: BluffNetXL,
    optimizer: torch.optim.Optimizer,
    transitions: List[RLTransition],
    advantages: List[float],
    returns: List[float],
    batch_size: int = 256,
    epochs: int = 4,
    clip_eps: float = 0.20,
    vf_coef: float = 0.50,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    net.train()
    states = torch.stack([t.state for t in transitions]).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    masks = torch.stack([t.mask for t in transitions]).to(device)
    old_logp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32, device=device)
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)

    # Advantage normalization
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(transitions)
    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            mb = perm[start:start + batch_size]
            logits, values = net(states[mb])

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

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            updates += 1

    return {k: v / max(1, updates) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Opponent Roster Sampling
# ---------------------------------------------------------------------------

def sample_league_opponent(iteration: int):
    """Sample opponent with balanced curriculum: 60% synthetic personas, 40% baselines."""
    r = random.random()
    if r < 0.60:
        proto = random.choice(SYNTHETIC_POPULATION)
        return PersonaBot(
            proto.name,
            proto.true_bluff_rate,
            proto.true_call_rate,
            proto.bluff_size_pref,
            proto.honest_dump_multi,
        )
    elif r < 0.70:
        return HonestBot()
    elif r < 0.80:
        return CardCountBot()
    elif r < 0.90:
        return BayesianBot()
    else:
        return RandomBot()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train_ppo_bluffnet_xl(
    checkpoint_in: str = "nn/checkpoints/bluffnet_xl_ultimate_v2.pt",
    checkpoint_out: str = "nn/checkpoints/bluffnet_xl_ppo.pt",
    total_episodes: int = 10000,
    rollout_batch: int = 500,
    lr: float = 3e-5,
    device_name: str = "auto",
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 80)
    print("  BLUFFNET-XL PPO REINFORCEMENT LEARNING FINE-TUNING")
    print(f"  Warm-Start Checkpoint: {checkpoint_in}")
    print(f"  Target Checkpoint:     {checkpoint_out}")
    print(f"  Hardware:              {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Total Episodes:        {total_episodes:,} (Rollout batch: {rollout_batch})")
    print("=" * 80)

    # Initialize model from supervised checkpoint
    net = BluffNetXL(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=512)
    if os.path.exists(checkpoint_in):
        d = torch.load(checkpoint_in, map_location="cpu", weights_only=True)
        net.load_state_dict(d["model_state_dict"])
        print(f"[PPO] Successfully warm-started from {checkpoint_in} (Val Acc: {d.get('val_acc', 0)*100:.2f}%)")
    else:
        print(f"[PPO] Warning: {checkpoint_in} not found, starting fresh.")

    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    agent = XLRLPlayer(net, device)

    episodes_done = 0
    t0 = time.time()
    best_wr = 0.0

    print(f"\n{'Iter':>5} | {'Episodes':>9} | {'Wins':>6} | {'Losses':>6} | {'Draws':>6} | {'Win Rate':>8} | {'Policy Loss':>11} | {'Val Loss':>9} | {'Time':>6}")
    print("-" * 85)

    iteration = 0
    best_net_score = -float("inf")
    while episodes_done < total_episodes:
        iteration += 1
        batch_transitions: List[RLTransition] = []
        batch_advantages: List[float] = []
        batch_returns: List[float] = []

        w, l, dr = 0, 0, 0
        t_iter_start = time.time()

        for ep in range(rollout_batch):
            opp = sample_league_opponent(iteration)
            seat = ep % 2
            trans, outcome, _ = play_rl_episode(agent, opp, seat)

            if outcome == 1:
                w += 1
            elif outcome == -1:
                l += 1
            else:
                dr += 1

            if trans:
                advs, rets = compute_gae(trans, last_value=0.0)
                batch_transitions.extend(trans)
                batch_advantages.extend(advs)
                batch_returns.extend(rets)

        episodes_done += rollout_batch
        wr = w / rollout_batch
        net_score = w - l

        # PPO update on harvested transitions
        metrics = ppo_update_xl(
            net,
            optimizer,
            batch_transitions,
            batch_advantages,
            batch_returns,
            batch_size=256,
            epochs=4,
            device=device,
        )

        iter_time = time.time() - t_iter_start
        star = ""
        if net_score > best_net_score:
            best_net_score = net_score
            star = " ★"
            os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "architecture": "BluffNetXL",
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": 512,
                "win_rate": wr,
                "net_score": net_score,
                "episodes": episodes_done,
                "iteration": iteration,
            }, checkpoint_out)

        print(f"{iteration:5d} | {episodes_done:9,d} | {w:6d} | {l:6d} | {dr:6d} | {wr*100:7.1f}% | {metrics['policy_loss']:11.4f} | {metrics['value_loss']:9.4f} | {iter_time:5.1f}s{star}")

    total_time = time.time() - t0
    print("-" * 85)
    print(f"PPO Training Complete in {total_time:.1f}s ({episodes_done/total_time:.1f} episodes/s).")
    print(f"Best League Win Rate: {best_wr*100:.2f}%")

    # Always save final checkpoint
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
        "architecture": "BluffNetXL",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dim": 512,
        "win_rate": wr,
        "episodes": episodes_done,
        "iteration": iteration,
    }, checkpoint_out)
    print(f"Final checkpoint saved to {checkpoint_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--rollout-batch", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--checkpoint-in", type=str, default="nn/checkpoints/bluffnet_xl_ultimate_v2.pt")
    parser.add_argument("--checkpoint-out", type=str, default="nn/checkpoints/bluffnet_xl_ppo.pt")
    args = parser.parse_args()

    train_ppo_bluffnet_xl(
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        total_episodes=args.episodes,
        rollout_batch=args.rollout_batch,
        lr=args.lr,
    )
