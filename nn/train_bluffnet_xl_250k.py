"""
BluffNet-XL Ultimate V2: 250,000 Transitions on RTX 3050 GPU (40 Epochs).
Directly satisfies user mandate:
  "Keep working, find something to work on finish and complete the project use bigger data sets, run more epochs make it the ultimate model"

Features:
- 250,000 diverse state-action-mask transitions harvested across all 20 synthetic personas + AcademicBeast self-play.
- Full 54-action discrete space coverage.
- Cosine Annealing learning rate schedule (1e-3 -> 1e-6) over 40 epochs.
- AdamW optimizer with weight decay 1e-4 and gradient clipping (max_norm 1.0).
- Automatic GPU hardware acceleration on NVIDIA GeForce RTX 3050 6GB Laptop GPU.
- Checkpoint output to nn/checkpoints/bluffnet_xl_ultimate_v2.pt.
"""

import os
import sys
import time
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath("."))

from cards import Card, Rank
from game import GameState, Action
from bots.base import pending_claims_from_actions, pool_by_rank
from bots.academic_beast_bot import AcademicBeastBot
from nn.model import (
    BluffNetXL,
    ACTION_DIM,
    CALL_ACTION,
    PASS_ACTION,
    action_index,
    build_legal_actions,
)
from nn.state_encoder import StateEncoder, STATE_DIM
from experiments.synthetic_population_eval import SYNTHETIC_POPULATION, PersonaBot


def generate_league_data(
    target_samples: int = 250000,
    cache_path: str = "data/synthetic_league_250k.pt",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if os.path.exists(cache_path):
        print(f"[BluffNet-XL 250k] Loading cached dataset from {cache_path}...")
        data = torch.load(cache_path, weights_only=True)
        return data["states"], data["actions"], data["masks"]

    print(f"[BluffNet-XL 250k] Harvesting {target_samples:,} diverse transitions across 20 synthetic personas + AcademicBeast...")
    encoder = StateEncoder()
    states = []
    actions = []
    masks = []

    samples = 0
    game_idx = 0
    t0 = time.time()

    # Pre-instantiate an AcademicBeastBot for grandmaster demonstrations
    grandmaster = AcademicBeastBot(use_nn=False)  # pure game-theoretic expert

    while samples < target_samples:
        game_idx += 1
        persona = SYNTHETIC_POPULATION[game_idx % len(SYNTHETIC_POPULATION)]

        game = GameState(num_players=2)
        game.deal(14)
        grandmaster.reset()

        use_gm = (game_idx % 3 == 0)  # 33% grandmaster demonstration games

        for turn in range(1, 100):
            if game.game_over:
                break
            cp = game.current_player
            opp = 1 - cp

            active_hand = game.get_hand(cp).cards
            opp_hand = game.get_hand(opp).cards

            if not active_hand:
                break

            pending = pending_claims_from_actions(game.actions)
            pool = pool_by_rank(active_hand, pending)

            context = {
                "opponent_hand_size": len(opp_hand),
                "pile_size": game.get_pile_size(),
                "draw_pile_size": len(game.draw_pile),
                "turn_number": turn,
                "can_pass": len(game.draw_pile) > 0,
                "last_action": game.actions[-1] if game.actions else None,
                "cards_remaining": pool,
                "opponent_call_rate": persona.true_call_rate,
                "opponent_bluff_revealed": persona.true_bluff_rate,
                "my_bluff_rate": 0.20,
                "my_bluff_success_rate": 0.60,
            }

            s_vec = encoder.encode(active_hand, context)
            mask = build_legal_actions(active_hand, can_call=False, can_pass=False, respond_only=False)

            if use_gm and cp == 0:
                p_cards, p_rank = grandmaster.decide_play(active_hand, context)
            else:
                p_cards, p_rank = persona.decide_play(active_hand, context)

            if not p_cards:
                break

            q = min(4, max(1, len(p_cards)))
            act_idx = action_index(p_rank, q)
            if mask[act_idx]:
                states.append(s_vec)
                actions.append(act_idx)
                masks.append(mask)
                samples += 1

            success, _ = game.play_cards(cp, p_cards, p_rank)
            if not success or game.game_over:
                break

            last_action = game.actions[-1]
            if not active_hand:
                break

            # Response phase (CALL or PASS)
            opp_can_pass = len(game.draw_pile) > 0
            resp_context = {
                "opponent_hand_size": len(game.get_hand(cp).cards),
                "pile_size": game.get_pile_size(),
                "draw_pile_size": len(game.draw_pile),
                "turn_number": turn,
                "can_pass": opp_can_pass,
                "last_action": last_action,
                "cards_remaining": pool_by_rank(opp_hand, pending_claims_from_actions(game.actions)),
                "opponent_call_rate": 0.35,
                "opponent_bluff_revealed": 0.20,
                "my_bluff_rate": 0.20,
                "my_bluff_success_rate": 0.60,
            }
            s_resp = encoder.encode(opp_hand, resp_context)
            resp_mask = build_legal_actions(opp_hand, can_call=True, can_pass=opp_can_pass, respond_only=True)

            if use_gm and opp == 0:
                should_call = grandmaster.decide_call(last_action, resp_context)
            else:
                should_call = persona.decide_call(last_action, resp_context)

            resp_act = CALL_ACTION if should_call else PASS_ACTION

            if resp_mask[resp_act]:
                states.append(s_resp)
                actions.append(resp_act)
                masks.append(resp_mask)
                samples += 1

            if should_call:
                succ, _, act = game.call_bluff(opp)
                if act:
                    grandmaster.observe_action(act, len(game.get_hand(cp).cards))
            else:
                if game.can_pass():
                    game.pass_turn(passer=opp)
                else:
                    succ, _, act = game.call_bluff(opp)
                    if act:
                        grandmaster.observe_action(act, len(game.get_hand(cp).cards))

            if samples >= target_samples:
                break

        if game_idx % 250 == 0 or samples >= target_samples:
            rate = samples / max(0.001, time.time() - t0)
            print(f"  [Harvest] {samples:,} / {target_samples:,} samples ({rate:.0f} samples/s) across {game_idx} matches...")

    states_t = torch.stack(states[:target_samples])
    actions_t = torch.tensor(actions[:target_samples], dtype=torch.long)
    masks_t = torch.stack(masks[:target_samples])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"states": states_t, "actions": actions_t, "masks": masks_t}, cache_path)
    print(f"[BluffNet-XL 250k] Dataset saved to {cache_path} ({states_t.size(0):,} samples in {time.time()-t0:.1f}s).")
    return states_t, actions_t, masks_t


def train_bluffnet_xl_250k(
    samples: int = 250000,
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    min_lr: float = 1e-6,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    output_path: str = "nn/checkpoints/bluffnet_xl_ultimate_v2.pt",
):
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("=" * 70)
    print(f"  BLUFFNET-XL ULTIMATE V2 TRAINING (250k TRANSITIONS, {epochs} EPOCHS)")
    print(f"  Hardware: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"  Target Checkpoint: {output_path}")
    print("=" * 70)

    states_t, actions_t, masks_t = generate_league_data(target_samples=samples)

    dataset = TensorDataset(states_t, actions_t, masks_t)
    val_size = int(0.10 * len(dataset))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(20260911)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda"), num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == "cuda"))

    model = BluffNetXL(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=512).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[BluffNet-XL] Parameters: {param_count:,} weights (hidden_dim=512, 2 residual blocks, LayerNorm, GELU)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>9} | {'Time':>6}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total_items = 0

        for b_states, b_actions, b_masks in train_loader:
            b_states = b_states.to(device)
            b_actions = b_actions.to(device)
            b_masks = b_masks.to(device)

            optimizer.zero_grad()
            logits, _ = model(b_states)

            # Mask invalid actions
            masked_logits = logits.clone()
            masked_logits[~b_masks] = float("-inf")

            loss = criterion(masked_logits, b_actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * b_states.size(0)
            preds = masked_logits.argmax(dim=-1)
            correct += (preds == b_actions).sum().item()
            total_items += b_states.size(0)

        scheduler.step()

        train_loss = total_loss / total_items
        train_acc = correct / total_items

        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_items = 0

        with torch.no_grad():
            for b_states, b_actions, b_masks in val_loader:
                b_states = b_states.to(device)
                b_actions = b_actions.to(device)
                b_masks = b_masks.to(device)

                logits, _ = model(b_states)
                masked_logits = logits.clone()
                masked_logits[~b_masks] = float("-inf")

                loss = criterion(masked_logits, b_actions)
                v_loss += loss.item() * b_states.size(0)
                preds = masked_logits.argmax(dim=-1)
                v_correct += (preds == b_actions).sum().item()
                v_items += b_states.size(0)

        val_loss = v_loss / v_items
        val_acc = v_correct / v_items
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - t_epoch_start

        star = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            star = " ★"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "architecture": "BluffNetXL",
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "hidden_dim": 512,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "samples": samples,
                "epochs": epochs,
                "best_epoch": epoch,
            }, output_path)

        print(f"{epoch:6d} | {train_loss:11.4f} | {train_acc*100:9.2f}% | {val_loss:10.4f} | {val_acc*100:8.2f}% | {current_lr:9.2e} | {epoch_time:5.1f}s{star}")

    total_time = time.time() - start_time
    print("-" * 75)
    print(f"Training Complete in {total_time:.1f}s ({total_time/epochs:.2f}s/epoch).")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Checkpoint saved to {output_path}")

    # Synchronize to default league checkpoint as well
    league_path = "nn/checkpoints/bluffnet_xl_league.pt"
    if os.path.exists(output_path):
        import shutil
        shutil.copyfile(output_path, league_path)
        print(f"Promoted to {league_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=250000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="nn/checkpoints/bluffnet_xl_ultimate_v2.pt")
    args = parser.parse_args()

    train_bluffnet_xl_250k(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_path=args.output,
    )
