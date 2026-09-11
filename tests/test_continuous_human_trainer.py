"""Unit tests for the Continuous Human Telemetry Ingestion & Distillation Pipeline."""

import os
import json
import pytest
import torch

from scripts.continuous_human_trainer import (
    load_replay_buffer,
    save_replay_buffer,
    append_to_buffer,
    run_continuous_update,
    BUFFER_FILE,
    MANIFEST_FILE,
    OUTPUT_CHECKPOINT,
)


class TestContinuousHumanTrainer:
    def test_buffer_save_and_load(self, tmp_path):
        test_states = torch.randn(10, 39)
        test_actions = torch.randint(0, 54, (10,))
        test_masks = torch.ones(10, 54, dtype=torch.bool)

        # Use temporary file location
        test_buf = str(tmp_path / "test_buf.pt")
        torch.save({"states": test_states, "actions": test_actions, "masks": test_masks}, test_buf)

        loaded = torch.load(test_buf, weights_only=True)
        assert loaded["states"].shape == (10, 39)
        assert loaded["actions"].shape == (10,)
        assert loaded["masks"].shape == (10, 54)

    def test_append_to_buffer_growth(self):
        new_states = torch.randn(15, 39)
        new_actions = torch.randint(0, 54, (15,))
        new_masks = torch.ones(15, 54, dtype=torch.bool)

        initial_count = 0
        old_s, _, _ = load_replay_buffer()
        if old_s is not None:
            initial_count = old_s.size(0)

        total_after = append_to_buffer(new_states, new_actions, new_masks)
        assert total_after == initial_count + 15

    def test_end_to_end_continuous_adaptation(self):
        """Verify full continuous update cycle executes and updates manifest."""
        res = run_continuous_update(
            epochs=1,
            batch_size=32,
            min_new_samples=64,
        )

        assert "samples_ingested" in res
        assert res["samples_ingested"] >= 64
        assert "final_val_loss" in res
        assert os.path.exists(OUTPUT_CHECKPOINT)
        assert os.path.exists(MANIFEST_FILE)

        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
            assert len(manifest) >= 1
            latest = manifest[-1]
            assert "timestamp" in latest
            assert latest["checkpoint_promoted"] == OUTPUT_CHECKPOINT
