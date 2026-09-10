"""
Synthetic Population League Generator & Policy Distillation (Phase 2).

Generates large-scale synthetic gameplay trajectories against the 20-persona population:
1. Simulates games across all 20 personas to produce 50,000 diverse state-action transitions.
2. Trains a specialized BluffNet checkpoint (nn/checkpoints/synthetic_league.pt) on this population.
3. Evaluates the policy against standard competitive baselines.
"""

import sys
import os
import time
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank, Deck
from game import GameState, Action
from nn.model import (
    BluffNet,
    ACTION_DIM,
    CALL_ACTION,
    PASS_ACTION,
    action_index,
    build_legal_actions,
)
from nn.state_encoder import StateEncoder, STATE_DIM
from nn.training import load_checkpoint, save_checkpoint
from test_bots import build_game_state, play_bot_vs_bot
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot


def generate_synthetic_league_data(
    target_samples: int = 50000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates 50k transitions across all 20 personas."""
    print(f"Simulating synthetic league matches to gather {target_samples} transitions...")
    encoder = StateEncoder()
    states = []
    actions = []
    masks = []

    t0 = time.time()
    samples_collected = 0
    game_count = 0

    while samples_collected < target_samples:
        game_count += 1
        persona = SYNTHETIC_POPULATION[game_count % len(SYNTHETIC_POPULATION)]

        game = GameState(num_players=2)
        game.deal(14)
        deck = game.deck

        for turn in range(1, 80):
            if game.game_over:
                break
            cp = game.current_player
            opp = 1 - cp

            active_hand = game.get_hand(cp).cards
            opp_hand = game.get_hand(opp).cards

            if not active_hand:
                break

            context = {
                "opponent_hand_size": len(opp_hand),
                "pile_size": game.get_pile_size(),
                "draw_pile_size": len(game.draw_pile),
                "turn_number": turn,
                "can_pass": len(game.draw_pile) > 0,
                "last_action": game.actions[-1] if game.actions else None,
                "cards_remaining": {r: 4 for r in Rank},
                "opponent_call_rate": persona.true_call_rate,
                "opponent_bluff_revealed": persona.true_bluff_rate,
                "my_bluff_rate": 0.20,
                "my_bluff_success_rate": 0.60,
            }

            s_vec = encoder.encode(active_hand, context)
            mask = build_legal_actions(active_hand, can_call=False, can_pass=False, respond_only=False)

            p_cards, p_rank = persona.decide_play(active_hand, context)
            if not p_cards:
                continue

            q = min(4, max(1, len(p_cards)))
            a_idx = action_index(p_rank, q)

            states.append(s_vec)
            actions.append(a_idx)
            masks.append(mask)
            samples_collected += 1

            game.play_cards(cp, p_cards, p_rank)
            last_act = game.actions[-1]

            # Opponent responds
            can_pass = len(game.draw_pile) > 0
            resp_context = {
                "opponent_hand_size": len(active_hand),
                "pile_size": game.get_pile_size(),
                "draw_pile_size": len(game.draw_pile),
                "turn_number": turn,
                "can_pass": can_pass,
                "last_action": last_act,
                "cards_remaining": {r: 4 for r in Rank},
                "opponent_call_rate": 0.40,
                "opponent_bluff_revealed": 0.20,
                "my_bluff_rate": 0.20,
                "my_bluff_success_rate": 0.60,
            }
            resp_s_vec = encoder.encode(opp_hand, resp_context)
            resp_mask = build_legal_actions(opp_hand, can_call=True, can_pass=can_pass, respond_only=True)

            should_call = persona.decide_call(last_act, resp_context)
            resp_a_idx = CALL_ACTION if (should_call or not can_pass) else PASS_ACTION

            states.append(resp_s_vec)
            actions.append(resp_a_idx)
            masks.append(resp_mask)
            samples_collected += 1

            if should_call or not can_pass:
                game.call_bluff(opp)
            else:
                game.pass_turn(opp)

            if samples_collected >= target_samples:
                break

    duration = time.time() - t0
    print(f"Generated {len(states)} samples from {game_count} simulated games in {duration:.1f}s.")

    return torch.stack(states), torch.tensor(actions, dtype=torch.long), torch.stack(masks)


def train_synthetic_league(
    target_samples: int = 30000,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    output_path: str = "nn/checkpoints/synthetic_league.pt",
):
    print("=================================================================")
    print("      Synthetic League Distillation & Pretraining               ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    states, actions, masks = generate_synthetic_league_data(target_samples)

    dataset = TensorDataset(states, actions, masks)
    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = BluffNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    t0 = time.time()

    print(f"\nTraining BluffNet on {n_train} transitions ({n_val} validation)...")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for b_states, b_actions, b_masks in train_loader:
            b_states, b_actions, b_masks = b_states.to(device), b_actions.to(device), b_masks.to(device)
            logits, _ = model.forward(b_states)
            masked_logits = logits.clone()
            masked_logits[~b_masks] = float("-inf")
            log_probs = F.log_softmax(masked_logits, dim=-1)

            loss = F.nll_loss(log_probs, b_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = log_probs.argmax(dim=-1)
            correct += (preds == b_actions).sum().item()
            total += b_states.size(0)
            total_loss += loss.item() * b_states.size(0)

        tr_loss = total_loss / total
        tr_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for b_states, b_actions, b_masks in val_loader:
                b_states, b_actions, b_masks = b_states.to(device), b_actions.to(device), b_masks.to(device)
                logits, _ = model.forward(b_states)
                masked_logits = logits.clone()
                masked_logits[~b_masks] = float("-inf")
                log_probs = F.log_softmax(masked_logits, dim=-1)

                loss = F.nll_loss(log_probs, b_actions)
                preds = log_probs.argmax(dim=-1)
                v_correct += (preds == b_actions).sum().item()
                v_total += b_states.size(0)
                v_loss += loss.item() * b_states.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_checkpoint(model, output_path)

        star = " *" if is_best else ""
        print(f"Epoch {ep:2d}/{epochs} | Train Loss: {tr_loss:.4f} (Acc: {tr_acc*100:4.1f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:4.1f}%){star}")

    duration = time.time() - t0
    print(f"\nTraining completed in {duration:.1f}s. Checkpoint saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", default="nn/checkpoints/synthetic_league.pt")
    args = parser.parse_args()

    train_synthetic_league(
        target_samples=args.samples,
        epochs=args.epochs,
        output_path=args.output,
    )
