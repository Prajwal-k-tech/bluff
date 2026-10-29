"""Toy likelihoods for a normal fixed-rank continuation decision.

This is an explicit count-based event model, not an optimal policy, human
model, or an implemented mixed-style game opponent. Inputs describe a
hypothetical current hand and known own pile cards; no hidden hand or engine
truth is read. The categories play/pass/call are all available in this normal
state, and rank selection is absent because the rank is fixed for the round.
An optional contradiction flag may be supplied only when proved from the
actor's legal view; the default relies only on current rank-copy counts.
"""

from collections.abc import Mapping
from math import comb, isfinite

from cards import Rank
from bots.round_opening_likelihood import STYLES, _validated_counts

ContinuationEvent = tuple[str, int, bool | None]


def continuation_distribution(
    rank_counts: Mapping[Rank, int],
    style: str,
    active_rank: Rank,
    previous_quantity: int,
    *,
    own_pile_rank_count: int = 0,
    other_certified_false: bool = False,
    mixed_respects_certificates: bool = False,
    mixed_takes_truthful_finish: bool = False,
    bluff_probability: float = 0.3,
    call_probability: float = 0.35,
) -> dict[ContinuationEvent, float]:
    """Return P(event) for a toy opponent continuing after a packet.

    Events are ``("play", quantity, truthful)``, ``("pass", 0, None)`` or
    ``("call_bluff", 0, None)``. ``random_uniform`` chooses each category
    uniformly, then a legal packet quantity uniformly and truth by physical
    subset counts. ``mixed_uniform`` uses independent call mass, then a
    truthful/bluff mixture over its supported quantities.

    With ``mixed_respects_certificates=True``, only the mixed style instead
    calls with certainty when its legal card knowledge proves a contradiction.
    This models a card-aware bluffer without assuming truthful play. The flag
    is an opt-in research hypothesis, not a change to Random or a claim about
    all human behavior.
    """
    counts = _validated_counts(rank_counts)
    if not isinstance(active_rank, Rank):
        raise ValueError("active_rank must be a canonical Rank value")
    if style not in STYLES:
        raise ValueError(f"unknown style: {style!r}")
    if type(previous_quantity) is not int or not 1 <= previous_quantity <= 4:
        raise ValueError("previous_quantity must be an integer from 1 through 4")
    if type(own_pile_rank_count) is not int or not 0 <= own_pile_rank_count <= 4:
        raise ValueError("own_pile_rank_count must be an integer from 0 through 4")
    if type(other_certified_false) is not bool:
        raise ValueError("other_certified_false must be a bool")
    if type(mixed_respects_certificates) is not bool:
        raise ValueError("mixed_respects_certificates must be a bool")
    if type(mixed_takes_truthful_finish) is not bool:
        raise ValueError("mixed_takes_truthful_finish must be a bool")
    if counts[active_rank] + own_pile_rank_count > 4:
        raise ValueError("active-rank hand and pile counts cannot exceed four")

    def probability(value: float, name: str) -> float:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return float(value)

    bluff_probability = probability(bluff_probability, "bluff_probability")
    call_probability = probability(call_probability, "call_probability")
    n, hand_size = counts[active_rank], sum(counts.values())
    max_quantity = min(4, hand_size)
    call = ("call_bluff", 0, None)
    passed = ("pass", 0, None)
    result: dict[ContinuationEvent, float] = {}

    def add(event: ContinuationEvent, mass: float) -> None:
        if mass:
            result[event] = result.get(event, 0.0) + mass

    if style == "honest_max":
        if (other_certified_false
                or n + own_pile_rank_count + previous_quantity > 4):
            add(call, 1.0)
        elif n:
            add(("play", min(4, n), True), 1.0)
        else:
            add(passed, 1.0)
    elif style == "random_uniform":
        add(call, 1 / 3)
        add(passed, 1 / 3)
        for quantity in range(1, max_quantity + 1):
            true_mass = comb(n, quantity) / comb(hand_size, quantity)
            pair_mass = 1 / (3 * max_quantity)
            add(("play", quantity, True), pair_mass * true_mass)
            add(("play", quantity, False), pair_mass * (1 - true_mass))
    else:
        if (mixed_takes_truthful_finish and hand_size <= 4
                and n == hand_size):
            add(("play", hand_size, True), 1.0)
            return result
        if (mixed_respects_certificates
                and (other_certified_false
                     or n + own_pile_rank_count + previous_quantity > 4)):
            add(call, 1.0)
            return result
        add(call, call_probability)
        remaining = 1 - call_probability
        if n:
            for quantity in range(1, n + 1):
                add(("play", quantity, True), remaining * (1 - bluff_probability) / n)
        else:
            add(passed, remaining * (1 - bluff_probability))

        false_quantities = [
            quantity for quantity in range(1, max_quantity + 1)
            if comb(hand_size, quantity) > comb(n, quantity)
        ]
        if false_quantities:
            mass = remaining * bluff_probability / len(false_quantities)
            for quantity in false_quantities:
                add(("play", quantity, False), mass)
        else:
            add(passed, remaining * bluff_probability)
    return result
