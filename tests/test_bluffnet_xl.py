"""Unit tests for BluffNetXL deep residual architecture."""

import os
import pytest
import torch
from nn.model import BluffNetXL, ACTION_DIM


class TestBluffNetXL:
    def test_architecture_parameters_and_shapes(self):
        net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM, hidden_dim=512)
        total_params = sum(p.numel() for p in net.parameters())
        assert total_params > 1_000_000, f'Expected >1M params, got {total_params}'

        batch = torch.randn(8, 39)
        logits, values = net.forward(batch)
        assert logits.shape == (8, ACTION_DIM)
        assert values.shape == (8, 1)

    def test_residual_connections_and_gradients(self):
        net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM, hidden_dim=512)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        batch = torch.randn(4, 39)
        logits, values = net.forward(batch)
        loss = logits.sum() + values.sum()
        loss.backward()

        # Check all parameters received valid, non-null, finite gradients
        for name, param in net.named_parameters():
            assert param.grad is not None, f'Gradient missing for {name}'
            assert not torch.isnan(param.grad).any(), f'NaN gradient in {name}'
            assert not torch.isinf(param.grad).any(), f'Inf gradient in {name}'

        optimizer.step()

    def test_masked_action_sampling_strictly_legal(self):
        net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM)
        state = torch.randn(39)
        # Create mask allowing only action 5 and action 12
        mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        mask[5] = True
        mask[12] = True

        sampled_actions = set()
        for _ in range(50):
            action = net.act(state, mask, deterministic=False)
            assert action in (5, 12), f'Sampled illegal action: {action}'
            sampled_actions.add(action)

        assert len(sampled_actions) >= 1

    def test_deterministic_argmax(self):
        net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM)
        state = torch.randn(39)
        mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        mask[7] = True
        mask[20] = True

        action = net.act(state, mask, deterministic=True)
        assert action in (7, 20)

    def test_checkpoint_serialization(self, tmp_path):
        net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM)
        chk_path = str(tmp_path / 'bluffnet_xl.pt')
        torch.save({'model_state_dict': net.state_dict(), 'hidden_dim': 512}, chk_path)

        loaded_net = BluffNetXL(state_dim=39, action_dim=ACTION_DIM)
        data = torch.load(chk_path, weights_only=True)
        loaded_net.load_state_dict(data['model_state_dict'])

        x = torch.randn(2, 39)
        with torch.no_grad():
            orig_logits, orig_val = net.forward(x)
            loaded_logits, loaded_val = loaded_net.forward(x)

        assert torch.allclose(orig_logits, loaded_logits, atol=1e-6)
        assert torch.allclose(orig_val, loaded_val, atol=1e-6)
