"""
Offline Policy Fine-Tuning on Human Gameplay Telemetry (Phase 3).

Fine-tunes a trained BluffNet checkpoint on human telemetry transitions
using supervised policy distillation regularized by KL divergence against the reference model:
    L(theta) = CrossEntropy(pi_theta(s), a_human) + beta_KL * D_KL(pi_ref(s) || pi_theta(s))

Preserves baseline Nash-equilibrium card counting while adapting action distributions
to human deception and calling patterns.
"""

import sys
import os
import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

sys.path.insert(0, os.path.abspath("."))

from nn.model import BluffNet, ACTION_DIM
from nn.training import load_checkpoint, save_checkpoint


def train_epoch(
    model: BluffNet,
    ref_model: BluffNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_weight: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    ref_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct = 0
    total_samples = 0

    for states, actions, masks in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        masks = masks.to(device)

        batch_size = states.size(0)

        # Forward pass
        logits, _ = model.forward(states)
        # Mask illegal actions
        masked_logits = logits.clone()
        masked_logits[~masks] = float("-inf")
        log_probs = F.log_softmax(masked_logits, dim=-1)

        # Supervised cross entropy loss
        ce_loss = F.nll_loss(log_probs, actions)

        # Reference policy KL penalty
        with torch.no_grad():
            ref_logits, _ = ref_model.forward(states)
            ref_masked = ref_logits.clone()
            ref_masked[~masks] = float("-inf")
            ref_probs = F.softmax(ref_masked, dim=-1)

        # Masked KL divergence: sum only over legal actions
        # KL(pi_ref || pi_theta) = sum_{a in legal} ref_probs * (log(ref_probs) - log_probs)
        log_ref = torch.log(torch.clamp(ref_probs, min=1e-8))
        kl_pointwise = torch.where(
            masks & (ref_probs > 1e-7),
            ref_probs * (log_ref - log_probs),
            torch.zeros_like(log_probs)
        )
        kl_div = kl_pointwise.sum(dim=-1).mean()

        loss = ce_loss + kl_weight * kl_div

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = log_probs.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total_samples += batch_size

        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_kl += kl_div.item() * batch_size

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        correct / total_samples,
    )


def evaluate(
    model: BluffNet,
    ref_model: BluffNet,
    dataloader: DataLoader,
    kl_weight: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates the model on validation data."""
    model.eval()
    ref_model.eval()

    total_loss = 0.0
    total_ce = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for states, actions, masks in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            masks = masks.to(device)

            batch_size = states.size(0)

            logits, _ = model.forward(states)
            masked_logits = logits.clone()
            masked_logits[~masks] = float("-inf")
            log_probs = F.log_softmax(masked_logits, dim=-1)

            ce_loss = F.nll_loss(log_probs, actions)

            ref_logits, _ = ref_model.forward(states)
            ref_masked = ref_logits.clone()
            ref_masked[~masks] = float("-inf")
            ref_probs = F.softmax(ref_masked, dim=-1)

            log_ref = torch.log(torch.clamp(ref_probs, min=1e-8))
            kl_pointwise = torch.where(
                masks & (ref_probs > 1e-7),
                ref_probs * (log_ref - log_probs),
                torch.zeros_like(log_probs)
            )
            kl_div = kl_pointwise.sum(dim=-1).mean()

            loss = ce_loss + kl_weight * kl_div

            preds = log_probs.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total_samples += batch_size

            total_loss += loss.item() * batch_size
            total_ce += ce_loss.item() * batch_size

    return (
        total_loss / total_samples,
        total_ce / total_samples,
        correct / total_samples,
    )


def finetune_human_policy(
    dataset_path: str = "data/human_dataset.pt",
    checkpoint_path: str = "nn/checkpoints/final.pt",
    output_path: str = "nn/checkpoints/human_adapted.pt",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    kl_weight: float = 0.15,
):
    print("=================================================================")
    print("   Offline Human Telemetry Fine-Tuning & Policy Distillation    ")
    print("=================================================================")
    print(f"Dataset:         {dataset_path}")
    print(f"Base Checkpoint: {checkpoint_path}")
    print(f"Output Path:     {output_path}")
    print(f"KL Weight:       {kl_weight}")
    print(f"Epochs / Batch:  {epochs} / {batch_size}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run scripts/export_human_dataset.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device:  {device}")

    # Load dataset
    data = torch.load(dataset_path, weights_only=True)
    states = data["states"]
    actions = data["actions"]
    masks = data["masks"]

    full_dataset = TensorDataset(states, actions, masks)
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    print(f"Loaded {len(full_dataset)} total samples ({n_train} train, {n_val} val).")

    # Load base model and create frozen reference model
    if os.path.exists(checkpoint_path):
        model = load_checkpoint(checkpoint_path).to(device)
        ref_model = load_checkpoint(checkpoint_path).to(device)
    else:
        print(f"Notice: checkpoint {checkpoint_path} not found; initializing fresh BluffNet.")
        model = BluffNet().to(device)
        ref_model = BluffNet().to(device)

    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    t0 = time.time()

    print("\nStarting fine-tuning...")
    for ep in range(1, epochs + 1):
        tr_loss, tr_ce, tr_acc = train_epoch(model, ref_model, train_loader, optimizer, kl_weight, device)
        val_loss, val_ce, val_acc = evaluate(model, ref_model, val_loader, kl_weight, device)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_checkpoint(model, output_path)

        star = " *" if is_best else ""
        print(f"Epoch {ep:2d}/{epochs} | Train Loss: {tr_loss:.4f} (Acc: {tr_acc*100:4.1f}%) | "
              f"Val Loss: {val_loss:.4f} (Acc: {val_acc*100:4.1f}%){star}")

    total_time = time.time() - t0
    print("\n=================================================================")
    print(f"Fine-tuning completed in {total_time:.1f}s.")
    print(f"Best model saved to {output_path} (Val Loss: {best_val_loss:.4f})")
    print("=================================================================")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BluffNet on human telemetry.")
    parser.add_argument("--dataset", default="data/human_dataset.pt", help="Path to telemetry dataset")
    parser.add_argument("--checkpoint", default="nn/checkpoints/final.pt", help="Base model checkpoint")
    parser.add_argument("--output", default="nn/checkpoints/human_adapted.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--kl-weight", type=float, default=0.15, help="KL divergence regularization weight")
    args = parser.parse_args()

    finetune_human_policy(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kl_weight=args.kl_weight,
    )


if __name__ == "__main__":
    main()
