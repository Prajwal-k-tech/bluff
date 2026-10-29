"""Exact continuation likelihoods over current physical-card constraints.

The proposal is uniform over disjoint physical actor-hand and earlier-pile
allocations satisfying hand lower bounds. It is not a complete action-history
posterior, an optimal policy, or a model of any real opponent.
"""

from collections.abc import Mapping
from math import comb, isfinite

from bots.round_continuation_likelihood import ContinuationEvent, STYLES
from cards import Rank


def exact_continuation_distribution(
    available_counts: Mapping[Rank, int], hand_size: int, pile_size: int,
    active_rank: Rank, previous_quantity: int, *,
    lower_bounds: Mapping[Rank, int] | None = None,
    other_certified_false: bool = False,
    honest_pile_truthful: bool = False,
    mixed_respects_certificates: bool = False,
    mixed_takes_truthful_finish: bool = False,
    bluff_probability: float = 0.3, call_probability: float = 0.35,
    max_states: int = 10000,
) -> dict[str, dict[str, object]]:
    """Integrate continuation events over uniform disjoint hand/pile deals.

    For each rank, ``n`` cards go to the actor and ``k`` to the earlier center
    pile with physical multiplicity ``C(a,n) C(a-n,k)``. Lower bounds constrain
    only ``n``. The returned per-style records contain the full event
    distribution, feasible physical allocation count, and memoized DP states.
    ``mixed_respects_certificates`` changes only mixed-style actions in each
    physical world, not its allocation law or the other styles. It is opt-in.
    """
    def counts(mapping, name):
        if not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping")
        if any(not isinstance(rank, Rank) for rank in mapping):
            raise ValueError(f"{name} keys must be canonical Rank values")
        values = {rank: mapping.get(rank, 0) for rank in Rank}
        if any(type(value) is not int or not 0 <= value <= 4
               for value in values.values()):
            raise ValueError(f"{name} values must be integers from 0 through 4")
        return values

    available = counts(available_counts, "available_counts")
    lower = counts(lower_bounds if lower_bounds is not None else {}, "lower_bounds")
    total = sum(available.values())
    if type(hand_size) is not int or hand_size < 1:
        raise ValueError("hand_size must be a positive integer")
    if type(pile_size) is not int or pile_size < 0:
        raise ValueError("pile_size must be a nonnegative integer")
    if hand_size + pile_size > total:
        raise ValueError("hand and pile sizes exceed available cards")
    if any(lower[r] > available[r] for r in Rank) or sum(lower.values()) > hand_size:
        raise ValueError("inconsistent hand lower bounds")
    if not isinstance(active_rank, Rank):
        raise ValueError("active_rank must be a canonical Rank value")
    if type(previous_quantity) is not int or not 1 <= previous_quantity <= 4:
        raise ValueError("previous_quantity must be an integer from 1 through 4")
    if type(other_certified_false) is not bool:
        raise ValueError("other_certified_false must be a bool")
    if type(honest_pile_truthful) is not bool:
        raise ValueError("honest_pile_truthful must be a bool")
    if type(mixed_respects_certificates) is not bool:
        raise ValueError("mixed_respects_certificates must be a bool")
    if type(mixed_takes_truthful_finish) is not bool:
        raise ValueError("mixed_takes_truthful_finish must be a bool")
    if (isinstance(bluff_probability, bool)
            or not isinstance(bluff_probability, (int, float))
            or not isfinite(bluff_probability) or not 0 <= bluff_probability <= 1):
        raise ValueError("bluff_probability must be finite and in [0, 1]")
    if (isinstance(call_probability, bool)
            or not isinstance(call_probability, (int, float))
            or not isfinite(call_probability) or not 0 <= call_probability <= 1):
        raise ValueError("call_probability must be finite and in [0, 1]")
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")

    ranks = tuple(rank for rank in Rank if rank != active_rank)
    memo = {}

    def other_allocations(index, left_hand, left_pile):
        key = (index, left_hand, left_pile)
        if key in memo:
            return memo[key]
        if len(memo) >= max_states:
            raise RuntimeError("exact continuation DP state budget exhausted")
        # Reserve before descending so the cap includes in-progress states.
        # The rank index strictly increases, so these placeholders cannot cycle.
        memo[key] = 0
        if left_hand < 0 or left_pile < 0:
            value = 0
        elif index == len(ranks):
            value = int(left_hand == 0 and left_pile == 0)
        else:
            rank = ranks[index]
            a = available[rank]
            value = 0
            for n in range(lower[rank], min(a, left_hand) + 1):
                hand_ways = comb(a, n)
                for k in range(min(a - n, left_pile) + 1):
                    value += (hand_ways * comb(a - n, k)
                              * other_allocations(index + 1,
                                                  left_hand - n,
                                                  left_pile - k))
        memo[key] = value
        return value

    active = active_rank
    denominator = 0
    marginal = []
    for n in range(lower[active], min(available[active], hand_size) + 1):
        for k in range(min(available[active] - n, pile_size) + 1):
            ways = (comb(available[active], n)
                    * comb(available[active] - n, k)
                    * other_allocations(0, hand_size - n, pile_size - k))
            if ways:
                denominator += ways
                marginal.append((n, k, ways))
    if not denominator:
        raise ValueError("no feasible physical allocation satisfies constraints")

    honest_marginal = ([(n, k, ways) for n, k, ways in marginal
                        if k == pile_size]
                       if honest_pile_truthful else marginal)
    honest_denominator = sum(ways for _, _, ways in honest_marginal)

    def events(style, n, k):
        result: dict[ContinuationEvent, float] = {}

        def add(event, mass):
            if mass:
                result[event] = result.get(event, 0.0) + mass

        call = ("call_bluff", 0, None)
        passed = ("pass", 0, None)
        if style == "honest_max":
            if other_certified_false or n + k + previous_quantity > 4:
                add(call, 1.0)
            elif n:
                add(("play", min(4, n), True), 1.0)
            else:
                add(passed, 1.0)
        elif style == "random_uniform":
            add(call, 1 / 3)
            add(passed, 1 / 3)
            limit = min(4, hand_size)
            for q in range(1, limit + 1):
                truth = comb(n, q) / comb(hand_size, q)
                pair_mass = 1 / (3 * limit)
                add(("play", q, True), pair_mass * truth)
                add(("play", q, False), pair_mass * (1 - truth))
        else:
            if (mixed_takes_truthful_finish and hand_size <= 4
                    and n == hand_size):
                # A truthful all-active-rank finish wins immediately and
                # therefore precedes calls, including certified-false calls.
                add(("play", hand_size, True), 1.0)
                return result
            if (mixed_respects_certificates
                    and (other_certified_false
                         or n + k + previous_quantity > 4)):
                add(call, 1.0)
                return result
            remaining = 1 - call_probability
            add(call, call_probability)
            if n:
                for q in range(1, n + 1):
                    add(("play", q, True), remaining * (1 - bluff_probability) / n)
            else:
                add(passed, remaining * (1 - bluff_probability))
            false_q = [q for q in range(1, min(4, hand_size) + 1)
                       if comb(hand_size, q) > comb(n, q)]
            if false_q:
                mass = remaining * bluff_probability / len(false_q)
                for q in false_q:
                    add(("play", q, False), mass)
            else:
                add(passed, remaining * bluff_probability)
        return result

    output = {}
    for style in STYLES:
        distribution: dict[ContinuationEvent, float] = {}
        style_marginal = honest_marginal if style == "honest_max" else marginal
        style_denominator = honest_denominator if style == "honest_max" else denominator
        for n, k, ways in style_marginal:
            for event, mass in events(style, n, k).items():
                distribution[event] = (distribution.get(event, 0.0)
                                       + ways * mass / style_denominator)
        output[style] = {
            "distribution": distribution,
            "allocation_count": style_denominator,
            "dp_states": len(memo),
        }
    return output
