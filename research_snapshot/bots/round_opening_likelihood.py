"""Toy opening-claim likelihoods for fixed-rank Bluff rounds.

These three explicit generative hypotheses are useful for likelihood
accounting, not claims about optimal play or any opponent's personality.
Input is a hypothetical pre-play hand count mapping; this module never reads
engine state or a hidden hand. Rank counts are the canonical 13 ``Rank`` keys.
"""

from collections.abc import Mapping
from math import comb, isfinite

from cards import Rank

Claim = tuple[Rank, int, bool]
STYLES = ("honest_max", "random_uniform", "mixed_uniform")


def _validated_counts(rank_counts: Mapping[Rank, int]) -> dict[Rank, int]:
    if not isinstance(rank_counts, Mapping):
        raise TypeError("rank_counts must be a mapping")
    if any(not isinstance(rank, Rank) for rank in rank_counts):
        raise ValueError("rank_counts keys must be canonical Rank values")
    counts = {rank: rank_counts.get(rank, 0) for rank in Rank}
    if any(type(count) is not int or not 0 <= count <= 4 for count in counts.values()):
        raise ValueError("rank counts must be integers from 0 through 4")
    if sum(counts.values()) == 0:
        raise ValueError("rank_counts must describe a nonempty hand")
    return counts


def opening_claim_distribution(
    rank_counts: Mapping[Rank, int], style: str, bluff_probability: float = 0.3,
    *, mixed_takes_truthful_finish: bool = False,
) -> dict[Claim, float]:
    """Return P(rank, quantity, truthful) under one toy opening hypothesis."""
    counts = _validated_counts(rank_counts)
    if style not in STYLES:
        raise ValueError(f"unknown style: {style!r}")
    if (isinstance(bluff_probability, bool)
            or not isinstance(bluff_probability, (int, float))
            or not isfinite(bluff_probability)
            or not 0 <= bluff_probability <= 1):
        raise ValueError("bluff_probability must be finite and in [0, 1]")
    if type(mixed_takes_truthful_finish) is not bool:
        raise ValueError("mixed_takes_truthful_finish must be a bool")

    hand_size = sum(counts.values())
    if (style == "mixed_uniform" and mixed_takes_truthful_finish
            and hand_size <= 4):
        held = [(rank, count) for rank, count in counts.items() if count]
        if len(held) == 1 and held[0][1] == hand_size:
            return {(held[0][0], hand_size, True): 1.0}
    limit = min(4, hand_size)
    result: dict[Claim, float] = {}

    def add(rank: Rank, quantity: int, truthful: bool, mass: float) -> None:
        if mass:
            key = (rank, quantity, truthful)
            result[key] = result.get(key, 0.0) + mass

    if style == "honest_max":
        largest = max(counts.values())
        tied = [rank for rank, count in counts.items() if count == largest]
        for rank in tied:
            add(rank, largest, True, 1.0 / len(tied))
    elif style == "random_uniform":
        denominator = 13 * limit
        for rank, count in counts.items():
            for quantity in range(1, limit + 1):
                subset_count = comb(hand_size, quantity)
                true_mass = comb(count, quantity) / subset_count if quantity <= count else 0.0
                pair_mass = 1.0 / denominator
                add(rank, quantity, True, pair_mass * true_mass)
                add(rank, quantity, False, pair_mass * (1.0 - true_mass))
    else:
        bluff_probability = float(bluff_probability)
        held_ranks = [(rank, n) for rank, n in counts.items() if n]
        false_pairs = [(rank, q) for rank, n in counts.items()
                       for q in range(1, limit + 1)
                       if comb(hand_size, q) > comb(n, q)]
        if held_ranks:
            rank_mass = (1.0 - bluff_probability) / len(held_ranks)
            for rank, count in held_ranks:
                for quantity in range(1, count + 1):
                    add(rank, quantity, True, rank_mass / count)
        if false_pairs:
            mass = bluff_probability / len(false_pairs)
            for rank, quantity in false_pairs:
                add(rank, quantity, False, mass)
    return result


def observed_claim_likelihood(distribution: Mapping[Claim, float], rank: Rank,
                              quantity: int) -> float:
    """Total likelihood of an observed rank/quantity, either truth value."""
    return sum(distribution.get((rank, quantity, truth), 0.0)
               for truth in (True, False))


def conditional_bluff_probability(distribution: Mapping[Claim, float], rank: Rank,
                                  quantity: int) -> float | None:
    """P(bluff | observed rank, quantity), or None for an impossible claim."""
    total = observed_claim_likelihood(distribution, rank, quantity)
    if total == 0:
        return None
    return distribution.get((rank, quantity, False), 0.0) / total
