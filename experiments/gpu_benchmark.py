"""GPU vs CPU benchmark for the real PPO update + inference path (BluffNet).

Settles the GPU-deferral ADR (2026-09-10) with measurements instead of asserts.
Uses the ACTUAL ppo_update() from nn/training.py on a rollout-sized batch
(2048 transitions, 4 epochs — the real per-cycle cost), plus the per-decision
inference cost that dominates the rollout loop.

Run:  .venv-gpu/bin/python experiments/gpu_benchmark.py
"""
import sys, os, time, statistics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim

from nn.training import ppo_update, Transition, load_checkpoint
from nn.state_encoder import STATE_DIM
from nn.model import ACTION_DIM

BATCH = 2048
EPOCHS = 4
STATE_DIM = STATE_DIM
ACTION_DIM = ACTION_DIM


def make_batch(n: int, device: str):
    """Synthetic rollout matching the real shapes (states stay on CPU —
    ppo_update moves them to the net's device itself, as in training)."""
    trans = []
    g = torch.Generator().manual_seed(42)
    for _ in range(n):
        state = torch.randn(STATE_DIM, generator=g)
        mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
        mask[torch.randperm(ACTION_DIM, generator=g)[:9]] = True  # ~9 legal actions
        trans.append(Transition(
            state=state,
            action=int(torch.randint(0, ACTION_DIM, (1,), generator=g)),
            mask=mask,
            log_prob=float(torch.randn(1, generator=g)),
            value=float(torch.randn(1, generator=g)),
        ))
    return trans


def bench_device(device: str, batch, rounds: int = 5):
    net = load_checkpoint("nn/checkpoints/final.pt").to(device)
    opt = optim.Adam(net.parameters(), lr=3e-4)

    # warmup (incl. CUDA kernel compile)
    ppo_update(net, opt, batch, epochs=EPOCHS)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        ppo_update(net, opt, batch, epochs=EPOCHS)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # per-decision inference (rollout side): masked forward, single state
    state = torch.randn(1, STATE_DIM, device=device)
    mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool, device=device)
    mask[0, :9] = True
    with torch.no_grad():
        for _ in range(10):  # warmup
            net.forward(state)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(1000):
            net.forward(state)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) / 1000 * 1000

    return {"update_mean_s": statistics.mean(times[1:]),
            "update_min_s": min(times[1:]),
            "infer_ms": infer_ms}


def main():
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
    batch = make_batch(BATCH, "cpu")

    results = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA unavailable — skipping")
            continue
        r = bench_device(device, batch)
        results[device] = r
        print(f"\n[{device.upper()}] ppo_update({BATCH} transitions, {EPOCHS} epochs): "
              f"mean {r['update_mean_s']*1000:.1f} ms  min {r['update_min_s']*1000:.1f} ms")
        print(f"[{device.upper()}] per-decision inference: {r['infer_ms']:.3f} ms")

    if "cpu" in results and "cuda" in results:
        su = results["cpu"]["update_mean_s"] / results["cuda"]["update_mean_s"]
        si = results["cpu"]["infer_ms"] / results["cuda"]["infer_ms"]
        e2e = 1.0 / (0.9 + 0.1 / su)  # update ≈10% of the training cycle
        print(f"\nVERDICT: update speedup x{su:.2f} | inference speedup x{si:.2f}")
        print(f"  end-to-end training gain at this net size ≈ x{e2e:.3f} "
              f"(~{100*(e2e-1):.1f}%) — GPU inference is SLOWER per decision; "
              f"revisit only with >=512-hidden nets or batched vectorized envs")


if __name__ == "__main__":
    main()
