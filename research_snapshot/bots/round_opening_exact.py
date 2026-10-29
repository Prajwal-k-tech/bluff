"""Exact opening style marginals under uniform physical hand allocations.

This is a count-level likelihood conditioned on minimum known rank counts,
not a posterior over action history or an actual hidden hand.
"""

from collections.abc import Mapping
from math import comb, isfinite

from cards import Rank
from bots.round_opening_likelihood import STYLES


def exact_opening_style_stats(
    available_counts: Mapping[Rank, int], hand_size: int, rank: Rank,
    quantity: int, bluff_probability: float = 0.3, *,
    lower_bounds: Mapping[Rank, int] | None = None, max_states: int = 10000,
    mixed_takes_truthful_finish: bool = False,
    honest_tie_rule: str = "uniform",
) -> dict[str, dict[str, float | int | None]]:
    """Return exact joint claim/truth masses for all toy opening styles.

    Physical hands are uniformly distributed among allocations satisfying the
    supplied per-rank minimum counts. `allocation_count` is their weighted
    physical-subset count; `dp_states` counts unique memoized recursion states.
    """
    if honest_tie_rule not in ("uniform", "lowest", "highest"):
        raise ValueError("unsupported maximum-rank tie rule")
    def canonical(mapping, label):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{label} must be a mapping")
        if any(not isinstance(key, Rank) for key in mapping):
            raise ValueError(f"{label} keys must be canonical Rank values")
        values = {r: mapping.get(r, 0) for r in Rank}
        if any(type(v) is not int or not 0 <= v <= 4 for v in values.values()):
            raise ValueError(f"{label} values must be integers from 0 through 4")
        return values

    available = canonical(available_counts, "available_counts")
    lower = canonical(lower_bounds if lower_bounds is not None else {}, "lower_bounds")
    if type(hand_size) is not int or not 1 <= hand_size <= sum(available.values()):
        raise ValueError("hand_size must be an integer from 1 through available total")
    if any(lower[r] > available[r] for r in Rank) or sum(lower.values()) > hand_size:
        raise ValueError("inconsistent lower bounds")
    if not isinstance(rank, Rank):
        raise ValueError("rank must be a canonical Rank")
    if type(quantity) is not int or not 1 <= quantity <= min(4, hand_size):
        raise ValueError("quantity must be an integer from 1 through min(4, hand_size)")
    if (isinstance(bluff_probability, bool)
            or not isinstance(bluff_probability, (int, float))
            or not isfinite(bluff_probability)
            or not 0 <= bluff_probability <= 1):
        raise ValueError("bluff_probability must be finite and in [0, 1]")
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")
    if type(mixed_takes_truthful_finish) is not bool:
        raise ValueError("mixed_takes_truthful_finish must be a bool")

    # Memoized coefficient DP. Features count nonempty ranks, or ranks tied at
    # q when every other count is at most q.
    memo = {}
    visited = 0
    # Separate the claimed rank so features count only other ranks.
    others = tuple(r for r in Rank if r != rank)
    def other_dp(index, left, mode, q=0):
        nonlocal visited
        key = ("other", index, left, mode, q)
        if key in memo:
            return memo[key]
        visited += 1
        if visited > max_states:
            raise RuntimeError("exact opening DP state budget exhausted")
        if left < 0:
            result = (0,) * (len(others) + 1)
        elif index == len(others):
            result = (1,) + (0,) * len(others) if left == 0 else (0,) * (len(others) + 1)
        else:
            r = others[index]
            hi = min(available[r], q) if mode == "honest" else available[r]
            if mode == "honest" and ((honest_tie_rule == "lowest" and r < rank) or
                                     (honest_tie_rule == "highest" and r > rank)):
                hi = min(hi, q-1)
            accum = [0] * (len(others) + 1)
            for n in range(lower[r], min(hi, left) + 1):
                feature = int(n > 0) if mode == "general" else int(n == q)
                for f, weight in enumerate(other_dp(index + 1, left - n, mode, q)):
                    if weight:
                        accum[f + feature] += comb(available[r], n) * weight
            result = tuple(accum)
        memo[key] = result
        return result

    denominator = sum(comb(available[rank], n) * sum(other_dp(0, hand_size - n, "general"))
                      for n in range(lower[rank], min(available[rank], hand_size) + 1))
    if denominator == 0:
        raise ValueError("no feasible hand allocation satisfies the constraints")
    n_weights = {
        n: comb(available[rank], n) * sum(other_dp(0, hand_size - n, "general"))
        for n in range(lower[rank], min(available[rank], hand_size) + 1)
    }
    mixed_true = 0.0
    mixed_false = 0.0
    honest_true = 0.0
    limit = min(4, hand_size)
    for n, weight in n_weights.items():
        if not weight:
            continue
        g_weights = other_dp(0, hand_size - n, "general")
        # Mixed true mass is (1-b)/(K*n): choose one of K held ranks, then
        # uniformly choose among its n truthful packet sizes.
        if quantity <= n:
            for g, ways in enumerate(g_weights):
                if ways:
                    k = g + int(n > 0)
                    world_mass = comb(available[rank], n) * ways / denominator
                    if (mixed_takes_truthful_finish and hand_size <= 4
                            and n == hand_size and g == 0):
                        # A collapsed hand has a deterministic whole-hand
                        # truthful opening; replace its generic mixed mass.
                        if quantity == hand_size:
                            mixed_true += world_mass
                    else:
                        mixed_true += world_mass * (
                            (1 - bluff_probability) / (k * n) if k else 0)
        # Honest-max toy chooses a maximum-count rank uniformly across ties;
        # its uniform tie rule is a simplifying hypothesis, not game policy.
        if n == quantity:
            tied = other_dp(0, hand_size - n, "honest", quantity)
            honest_true += comb(available[rank], n) * sum(
                ways / (t + 1) if honest_tie_rule == "uniform" else ways
                for t, ways in enumerate(tied)
            ) / denominator

        # Mixed false mass divides b uniformly over physically feasible false
        # pairs: all 13*limit when K>=2, or 12*limit in a collapsed hand.
        for g, ways in enumerate(g_weights):
            if not ways:
                continue
            ranks_held = g + int(n > 0)
            if quantity > limit:
                continue
            if (mixed_takes_truthful_finish and hand_size <= 4
                    and ranks_held == 1):
                # The generic false-pair mass is overridden in every
                # one-rank small hand, regardless of which rank it holds.
                continue
            if ranks_held >= 2:
                false_share = 1 / (13 * limit)
            elif hand_size <= 4 and n > 0:
                false_share = 0.0
            else:
                false_share = 1 / (12 * limit)
            mixed_false += (comb(available[rank], n) * ways / denominator
                            * bluff_probability * false_share)

    # For random-uniform, subset truth probability conditional on n held copies.
    random_true = random_false = 0.0
    if quantity <= limit:
        for n, weight in n_weights.items():
            if not weight:
                continue
            mass = weight / denominator / (13 * limit)
            truth = comb(n, quantity) / comb(hand_size, quantity) if quantity <= n else 0
            random_true += mass * truth
            random_false += mass * (1 - truth)

    stats = {}
    for name, truth, bluff in (
        ("honest_max", honest_true, 0.0),
        ("random_uniform", random_true, random_false),
        ("mixed_uniform", mixed_true, mixed_false),
    ):
        likelihood = truth + bluff
        stats[name] = {
            "joint_truth": truth, "joint_bluff": bluff,
            "likelihood": likelihood,
            "conditional_bluff": bluff / likelihood if likelihood else None,
            "allocation_count": denominator, "dp_states": visited,
        }
    return stats
