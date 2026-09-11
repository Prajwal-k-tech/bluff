"""Probe human_adapted.pt vs final.pt on a fixed set of scripted positions.

Collects per-decision logit statistics (call/pass/play distribution, entropy,
argmax agreement) to diagnose why human_adapted collapses vs Bayesian.
"""

import sys, os, random, json, hashlib
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn.model import BluffNet, ACTION_DIM, CALL_ACTION, PASS_ACTION, build_legal_actions
from nn.training import load_checkpoint
from nn.state_encoder import StateEncoder, STATE_DIM
from cards import Card, Rank, Suit, Deck

CHECKPOINT_A = "nn/checkpoints/final.pt"
CHECKPOINT_B = "nn/checkpoints/human_adapted.pt"
NUM_POSITIONS = 500
SEED = 42


def make_scripted_positions(n, seed):
    """Generate n scripted game positions with known legal masks."""
    rng = random.Random(seed)
    encoder = StateEncoder()
    positions = []

    for i in range(n):
        deck = Deck()
        deck.cards = list(deck.cards)
        rng.shuffle(deck.cards)

        h_size = rng.randint(3, 18)
        hand = deck.cards[:h_size]
        remaining_deck = deck.cards[h_size:]
        opp_size = rng.randint(3, 18)
        pile_s = rng.randint(0, 20)
        turn = rng.randint(1, 80)
        can_pass = turn < 24

        # Context with realistic opponent signals
        context = {
            "opponent_hand_size": opp_size,
            "pile_size": pile_s,
            "draw_pile_size": max(0, 24 - turn),
            "turn_number": turn,
            "can_pass": can_pass,
            "last_action": None,
            "cards_remaining": {r: max(0, 4 - sum(1 for c in hand if c.rank == r)) for r in Rank},
            "opponent_call_rate": rng.uniform(0.1, 0.6),
            "opponent_bluff_revealed": rng.uniform(0.1, 0.5),
            "my_bluff_rate": rng.uniform(0.05, 0.4),
            "my_bluff_success_rate": rng.uniform(0.3, 0.8),
        }
        state = encoder.encode(hand, context)

        # Generate masks for both play and respond phases
        play_mask = build_legal_actions(hand, can_call=False, can_pass=False, respond_only=False)
        respond_mask = build_legal_actions(hand, can_call=True, can_pass=can_pass, respond_only=True)

        # Alternate: even positions = play decisions, odd = respond decisions
        is_respond = (i % 2 == 1)
        mask = respond_mask if is_respond else play_mask

        positions.append({
            "state": state,
            "mask": mask,
            "is_respond": is_respond,
            "hand_size": h_size,
        })

    return positions


def probe_checkpoint(net, positions, name):
    """Collect logit/action statistics for each position."""
    net.eval()
    results = []

    for pos in positions:
        state = pos["state"].unsqueeze(0)
        mask = pos["mask"].unsqueeze(0)

        with torch.no_grad():
            logits, value = net(state)
            masked = logits.clone()
            masked[~mask] = float("-inf")
            log_probs = F.log_softmax(masked, dim=-1)
            probs = F.softmax(masked, dim=-1)

        # Stats
        call_prob = probs[0, CALL_ACTION].item()
        pass_prob = probs[0, PASS_ACTION].item()
        play_probs = probs[0].clone()
        play_probs[~mask[0]] = 0.0
        play_probs[CALL_ACTION] = 0.0
        play_probs[PASS_ACTION] = 0.0
        play_sum = play_probs.sum().item()

        # Entropy over legal actions
        legal_log_probs = log_probs[0].clone()
        legal_log_probs[~mask[0]] = 0.0
        entropy = -(probs[0][mask[0]] * legal_log_probs[mask[0]]).sum().item()

        # Argmax action
        argmax_idx = int(torch.argmax(probs[0]).item())
        argmax_is_call = argmax_idx == CALL_ACTION
        argmax_is_pass = argmax_idx == PASS_ACTION

        # Respond-specific: call rate when mask allows call+pass
        if pos["is_respond"]:
            respond_decision = "call" if call_prob > pass_prob else "pass"

        results.append({
            "call_prob": call_prob,
            "pass_prob": pass_prob,
            "play_prob_sum": play_sum,
            "entropy": entropy,
            "argmax_is_call": argmax_is_call,
            "argmax_is_pass": argmax_is_pass,
            "is_respond": pos["is_respond"],
            "hand_size": pos["hand_size"],
            "value": value.item(),
        })

    return results


def aggregate_stats(results, name):
    """Aggregate per-checkpoint statistics."""
    n = len(results)
    respond_results = [r for r in results if r["is_respond"]]
    play_results = [r for r in results if not r["is_respond"]]

    stats = {
        "name": name,
        "n_total": n,
        "n_respond": len(respond_results),
        "n_play": len(play_results),
        "mean_entropy": sum(r["entropy"] for r in results) / n,
        "mean_value": sum(r["value"] for r in results) / n,
        # Respond-phase stats
        "respond_mean_call_prob": sum(r["call_prob"] for r in respond_results) / max(1, len(respond_results)),
        "respond_mean_pass_prob": sum(r["pass_prob"] for r in respond_results) / max(1, len(respond_results)),
        "respond_argmax_call_rate": sum(r["argmax_is_call"] for r in respond_results) / max(1, len(respond_results)),
        "respond_argmax_pass_rate": sum(r["argmax_is_pass"] for r in respond_results) / max(1, len(respond_results)),
        "respond_mean_entropy": sum(r["entropy"] for r in respond_results) / max(1, len(respond_results)),
        # Play-phase stats (call/pass should be 0 or near 0)
        "play_mean_call_prob": sum(r["call_prob"] for r in play_results) / max(1, len(play_results)),
        "play_mean_pass_prob": sum(r["pass_prob"] for r in play_results) / max(1, len(play_results)),
        "play_argmax_call_rate": sum(r["argmax_is_call"] for r in play_results) / max(1, len(play_results)),
        "play_mean_entropy": sum(r["entropy"] for r in play_results) / max(1, len(play_results)),
    }

    # Call-rate shift: how often does the model choose call over pass in respond phase?
    # A global shift means human_adapted always calls
    respond_calls = sum(1 for r in respond_results if r["argmax_is_call"])
    respond_passes = sum(1 for r in respond_results if r["argmax_is_pass"])
    stats["respond_call_over_pass"] = respond_calls / max(1, respond_calls + respond_passes)

    # Temperature: use entropy as proxy. Lower entropy = more peaked = "colder"
    # Higher entropy = flatter = "hotter"
    stats["entropy_ratio"] = None  # computed after both

    # Per hand-size respond call rate
    hand_size_bins = {}
    for r in respond_results:
        hs = r["hand_size"]
        bucket = "small" if hs <= 7 else ("medium" if hs <= 12 else "large")
        if bucket not in hand_size_bins:
            hand_size_bins[bucket] = {"calls": 0, "total": 0, "call_prob_sum": 0.0}
        hand_size_bins[bucket]["total"] += 1
        hand_size_bins[bucket]["calls"] += int(r["argmax_is_call"])
        hand_size_bins[bucket]["call_prob_sum"] += r["call_prob"]
    for bucket, d in hand_size_bins.items():
        d["call_rate"] = d["calls"] / max(1, d["total"])
        d["mean_call_prob"] = d["call_prob_sum"] / max(1, d["total"])

    stats["respond_by_hand_size"] = hand_size_bins

    return stats


def main():
    print("=== Probe: human_adapted.pt vs final.pt ===")
    print(f"Positions: {NUM_POSITIONS}, Seed: {SEED}\n")

    # Load both checkpoints
    net_a = load_checkpoint(CHECKPOINT_A)
    net_b = load_checkpoint(CHECKPOINT_B)

    # Verify they're different
    hash_a = hashlib.md5(
        torch.cat([p.flatten() for p in net_a.parameters()]).detach().numpy().tobytes()
    ).hexdigest()
    hash_b = hashlib.md5(
        torch.cat([p.flatten() for p in net_b.parameters()]).detach().numpy().tobytes()
    ).hexdigest()
    print(f"final.pt param hash:        {hash_a}")
    print(f"human_adapted.pt param hash: {hash_b}")
    print(f"Identical: {hash_a == hash_b}\n")

    # Generate positions
    positions = make_scripted_positions(NUM_POSITIONS, SEED)

    # Probe both
    results_a = probe_checkpoint(net_a, positions, "final.pt")
    results_b = probe_checkpoint(net_b, positions, "human_adapted.pt")

    # Aggregate
    stats_a = aggregate_stats(results_a, "final.pt")
    stats_b = aggregate_stats(results_b, "human_adapted.pt")

    # Compute cross-stats
    stats_a["entropy_ratio"] = stats_a["mean_entropy"] / max(1e-8, stats_b["mean_entropy"])
    stats_b["entropy_ratio"] = stats_b["mean_entropy"] / max(1e-8, stats_a["mean_entropy"])

    # Logit difference analysis: same position, how do the logits differ?
    logit_diffs = []
    for i in range(NUM_POSITIONS):
        state = positions[i]["state"].unsqueeze(0)
        mask = positions[i]["mask"].unsqueeze(0)
        with torch.no_grad():
            logits_a, _ = net_a(state)
            logits_b, _ = net_b(state)
        # Masked logit difference for legal actions
        masked_a = logits_a[0].clone()
        masked_a[~mask[0]] = 0.0
        masked_b = logits_b[0].clone()
        masked_b[~mask[0]] = 0.0
        diff = (masked_b - masked_a).abs().mean().item()
        logit_diffs.append(diff)

    mean_logit_diff = sum(logit_diffs) / len(logit_diffs)

    # Print comparison
    print("=" * 72)
    print(f"{'Metric':<42} {'final.pt':>12} {'human_adapted':>12}")
    print("=" * 72)
    print(f"{'Mean entropy (all)':<42} {stats_a['mean_entropy']:>12.4f} {stats_b['mean_entropy']:>12.4f}")
    print(f"{'Mean entropy (respond only)':<42} {stats_a['respond_mean_entropy']:>12.4f} {stats_b['respond_mean_entropy']:>12.4f}")
    print(f"{'Mean entropy (play only)':<42} {stats_a['play_mean_entropy']:>12.4f} {stats_b['play_mean_entropy']:>12.4f}")
    print(f"{'Mean value estimate':<42} {stats_a['mean_value']:>12.4f} {stats_b['mean_value']:>12.4f}")
    print("-" * 72)
    print(f"{'RESPOND PHASE:'}")
    print(f"{'  Argmax call rate':<42} {stats_a['respond_argmax_call_rate']:>11.1%} {stats_b['respond_argmax_call_rate']:>11.1%}")
    print(f"{'  Argmax pass rate':<42} {stats_a['respond_argmax_pass_rate']:>11.1%} {stats_b['respond_argmax_pass_rate']:>11.1%}")
    print(f"{'  Mean P(call)':<42} {stats_a['respond_mean_call_prob']:>12.4f} {stats_b['respond_mean_call_prob']:>12.4f}")
    print(f"{'  Mean P(pass)':<42} {stats_a['respond_mean_pass_prob']:>12.4f} {stats_b['respond_mean_pass_prob']:>12.4f}")
    print(f"{'  Call/(Call+Pass) ratio':<42} {stats_a['respond_call_over_pass']:>11.1%} {stats_b['respond_call_over_pass']:>11.1%}")
    print("-" * 72)
    print(f"{'PLAY PHASE:'}")
    print(f"{'  Argmax call rate (should be 0)':<42} {stats_a['play_argmax_call_rate']:>11.1%} {stats_b['play_argmax_call_rate']:>11.1%}")
    print(f"{'  Mean P(call)':<42} {stats_a['play_mean_call_prob']:>12.4f} {stats_b['play_mean_call_prob']:>12.4f}")
    print(f"{'  Mean P(pass)':<42} {stats_a['play_mean_pass_prob']:>12.4f} {stats_b['play_mean_pass_prob']:>12.4f}")
    print("-" * 72)
    print(f"{'Mean |logit diff| (same position)':<42} {'':>12} {mean_logit_diff:>12.4f}")
    print(f"{'Entropy ratio (A/B)':<42} {'':>12} {stats_a['entropy_ratio']:>12.4f}")
    print("=" * 72)

    # Hand-size breakdown for respond phase
    print("\nRespond-phase call rate by hand size:")
    print(f"  {'Hand Size':<16} {'final.pt':>12} {'human_adapted':>12}")
    print(f"  {'-'*40}")
    for bucket in ["small", "medium", "large"]:
        d_a = stats_a["respond_by_hand_size"].get(bucket, {})
        d_b = stats_b["respond_by_hand_size"].get(bucket, {})
        cr_a = d_a.get("call_rate", 0)
        cr_b = d_b.get("call_rate", 0)
        print(f"  {bucket:<16} {cr_a:>11.1%} {cr_b:>11.1%}")

    # Diagnosis
    print("\n" + "=" * 72)
    print("DIAGNOSIS")
    print("=" * 72)

    # Hypothesis 1: Global call-threshold shift
    call_shift = abs(stats_b["respond_argmax_call_rate"] - stats_a["respond_argmax_call_rate"])
    print(f"\n(i) Global call-threshold shift: delta={call_shift:.1%}")
    if call_shift > 0.15:
        if stats_b["respond_argmax_call_rate"] > stats_a["respond_argmax_call_rate"]:
            print(f"    SUPPORTED: human_adapted calls MORE ({stats_b['respond_argmax_call_rate']:.1%} vs {stats_a['respond_argmax_call_rate']:.1%})")
        else:
            print(f"    SUPPORTED: human_adapted calls LESS ({stats_b['respond_argmax_call_rate']:.1%} vs {stats_a['respond_argmax_call_rate']:.1%})")
    else:
        print(f"    NOT SUPPORTED: shift too small ({call_shift:.1%})")

    # Hypothesis 2: Temperature/flattening
    entropy_change = stats_b["mean_entropy"] - stats_a["mean_entropy"]
    print(f"\n(ii) Temperature/flattening change: delta_entropy={entropy_change:+.4f}")
    if entropy_change > 0.05:
        print(f"    SUPPORTED: human_adapted is MORE random (higher entropy)")
    elif entropy_change < -0.05:
        print(f"    SUPPORTED: human_adapted is MORE deterministic (lower entropy)")
    else:
        print(f"    NOT SUPPORTED: entropy change too small")

    # Hypothesis 3: Data artifact
    print(f"\n(iii) Data artifact: Dataset has {1923/10000:.1%} calls, {3012/10000:.1%} passes, {5065/10000:.1%} plays")
    print(f"    Training data call rate (of respond decisions): ~{1923/(1923+3012):.1%} (call/(call+pass))")
    print(f"    vs actual model respond behavior:")
    print(f"      final.pt:     {stats_a['respond_call_over_pass']:.1%} call")
    print(f"      human_adapted: {stats_b['respond_call_over_pass']:.1%} call")

    # Check entropy ratio for temperature
    if stats_b["mean_entropy"] < stats_a["mean_entropy"] * 0.8:
        print(f"\n    → human_adapted entropy is {(1 - stats_b['mean_entropy']/stats_a['mean_entropy'])*100:.0f}% lower = peaked = COLLAPSED policy")
    elif stats_b["mean_entropy"] > stats_a["mean_entropy"] * 1.2:
        print(f"\n    → human_adapted entropy is {(stats_b['mean_entropy']/stats_a['mean_entropy'] - 1)*100:.0f}% higher = flattened = less decisive")

    # Save full results
    out_path = "data/experiments/probe_human_adapted.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stats_a": stats_a, "stats_b": stats_b, "mean_logit_diff": mean_logit_diff}, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
