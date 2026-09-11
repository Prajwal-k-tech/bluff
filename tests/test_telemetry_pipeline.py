import os
import torch
import pytest
from scripts.export_human_dataset import generate_synthetic_telemetry
from nn.model import BluffNet, ACTION_DIM
from nn.state_encoder import STATE_DIM
from nn.finetune_human import train_epoch, evaluate
from torch.utils.data import TensorDataset, DataLoader


def test_synthetic_telemetry_generation(tmp_path):
    n_samples = 50
    states, actions, masks = generate_synthetic_telemetry(n_samples)
    assert len(states) == n_samples
    assert states.shape == (n_samples, STATE_DIM)
    assert actions.shape == (n_samples,)
    assert masks.shape == (n_samples, ACTION_DIM)

    # Save to temp path
    out_file = tmp_path / "test_telemetry.pt"
    torch.save({
        "states": states,
        "actions": actions,
        "masks": masks,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
    }, out_file)
    assert os.path.exists(out_file)


def test_fine_tuning_step(tmp_path):
    device = torch.device("cpu")
    model = BluffNet().to(device)
    ref_model = BluffNet().to(device)
    ref_model.load_state_dict(model.state_dict())

    states, actions, masks = generate_synthetic_telemetry(40)
    dataset = TensorDataset(states, actions, masks)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss, ce, kl = train_epoch(model, ref_model, loader, optimizer, kl_weight=0.1, device=device)

    assert loss > 0.0
    assert not torch.isnan(torch.tensor(loss))

    val_loss, val_ce, val_acc = evaluate(model, ref_model, loader, kl_weight=0.1, device=device)
    assert 0.0 <= val_acc <= 1.0

    out_pt = tmp_path / "finetuned.pt"
    torch.save({"model_state_dict": model.state_dict()}, out_pt)
    assert os.path.exists(out_pt)
