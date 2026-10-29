"""Research-only opening-claim forecast over current-constraint hand proposals."""

from collections import Counter
from math import isfinite
import random

from bots.round_determinization import sample_root_world
from bots.round_opening_likelihood import STYLES, opening_claim_distribution
from cards import Rank


METHOD = "approximate opening-only current-constraint model, not history posterior"


def _style_weights(weights):
    if weights is None:
        return {style: 1 / len(STYLES) for style in STYLES}
    if not isinstance(weights, dict) or set(weights) != set(STYLES):
        raise ValueError("style_weights must contain exactly the canonical styles")
    values = weights.values()
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not isfinite(value) or value < 0 for value in values):
        raise ValueError("style weights must be finite and nonnegative")
    total = sum(weights.values())
    if not isfinite(total) or total <= 0:
        raise ValueError("style weights must have positive finite sum")
    return {style: weights[style] / total for style in STYLES}


def _opening_inputs(hand, state):
    if not isinstance(state, dict):
        raise TypeError("state must be a redacted observation dict")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("invalid viewer")
    sizes = tuple(state.get(key) for key in
                  ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size"))
    if (any(type(size) is not int or size < 0 for size in sizes)
            or sum(sizes) != 52 or sizes[0] != len(hand)):
        raise ValueError("public card counts must be nonnegative integers summing to 52")
    actions = state.get("actions")
    pending = state.get("last_action")
    if not isinstance(actions, (tuple, list)) or pending is None:
        raise ValueError("an unresolved opening claim is required")
    round_number = state.get("round_number")
    if type(round_number) is not int or round_number < 1:
        raise ValueError("invalid round number")
    player = getattr(pending, "player", None)
    action_index = getattr(pending, "action_index", None)
    pending_round = getattr(pending, "round_number", None)
    claimed_rank = getattr(pending, "claimed_rank", None)
    if (type(player) is not int or player not in (0, 1)
            or type(action_index) is not int or action_index < 0
            or type(pending_round) is not int or pending_round < 1
            or not isinstance(claimed_rank, Rank)):
        raise ValueError("invalid pending claim metadata")
    current = [action for action in actions
               if getattr(action, "round_number", None) == round_number]
    if (len(current) != 1
            or getattr(current[-1], "action_index", None)
            != action_index
            or player == viewer
            or action_index != len(actions) - 1
            or pending_round != round_number
            or state.get("active_rank") != claimed_rank
            or not isinstance(state.get("active_rank"), Rank)):
        raise ValueError("only the latest opponent packet of the current round is supported")
    quantity = len(getattr(pending, "cards_played", ()))
    if not 1 <= quantity <= 4:
        raise ValueError("invalid pending packet quantity")
    if (state.get("pile_size") != quantity or state.get("own_pile_cards") != []
            or state.get("can_call_bluff") is not True
            or getattr(pending, "bluff_called", None) is not False
            or getattr(pending, "response_kind", None) is not None):
        raise ValueError("pending claim is resolved or not an opening response")

    # Reconstruct the legal view before the pending play. Do not condition the
    # proposal on which hidden physical cards happened to occupy its packet.
    prestate = dict(state)
    prestate["actions"] = list(actions[:-1])
    prestate["last_action"] = None
    prestate["opponent_hand_size"] = state["opponent_hand_size"] + quantity
    prestate["pile_size"] = 0
    prestate["active_rank"] = None
    prestate["own_pile_cards"] = []
    return claimed_rank, quantity, prestate


def forecast_opening_claim(hand, state, rng: random.Random, *, samples=32,
                           max_states=10000, style_weights=None,
                           bluff_probability=0.3,
                           mixed_takes_truthful_finish=False):
    """Average toy style likelihoods across unconstrained pre-claim proposals."""
    if type(mixed_takes_truthful_finish) is not bool:
        raise ValueError("mixed_takes_truthful_finish must be a bool")
    if type(samples) is not int or not 1 <= samples <= 128:
        raise ValueError("samples must be an integer from 1 through 128")
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")
    if not isinstance(rng, random.Random):
        raise TypeError("rng must be random.Random")
    weights = _style_weights(style_weights)
    rank, quantity, prestate = _opening_inputs(hand, state)
    # Validate the prior even if all proposal samples later fail.
    if (isinstance(bluff_probability, bool)
            or not isinstance(bluff_probability, (int, float))
            or not isfinite(bluff_probability)
            or not 0 <= bluff_probability <= 1):
        raise ValueError("bluff_probability must be finite and in [0, 1]")

    joint = {style: {"true": 0.0, "false": 0.0} for style in STYLES}
    for _ in range(samples):
        world = sample_root_world(hand, prestate, rng, pending_truth=None,
                                  max_states=max_states)
        counts = Counter(card.rank for card in world.opponent_hand)
        distributions = {
            style: opening_claim_distribution(
                counts, style, bluff_probability,
                mixed_takes_truthful_finish=mixed_takes_truthful_finish)
            for style in STYLES
        }
        for style, distribution in distributions.items():
            joint[style]["true"] += distribution.get((rank, quantity, True), 0.0) / samples
            joint[style]["false"] += distribution.get((rank, quantity, False), 0.0) / samples

    style_stats = {}
    style_likelihoods = {}
    for style in STYLES:
        true_mass, false_mass = joint[style]["true"], joint[style]["false"]
        likelihood = true_mass + false_mass
        style_likelihoods[style] = likelihood
        style_stats[style] = {
            "joint_truth": true_mass,
            "joint_bluff": false_mass,
            "likelihood": likelihood,
            "conditional_bluff": false_mass / likelihood if likelihood else None,
        }
    total_likelihood = sum(weights[s] * style_likelihoods[s] for s in STYLES)
    posterior = ({s: weights[s] * style_likelihoods[s] / total_likelihood
                  for s in STYLES} if total_likelihood else {s: None for s in STYLES})
    total_false = sum(weights[s] * joint[s]["false"] for s in STYLES)
    total_true = sum(weights[s] * joint[s]["true"] for s in STYLES)
    total = total_true + total_false
    result = {
        "style_stats": style_stats,
        "style_posterior": posterior,
        "probability_bluff": total_false / total if total else None,
        "total_likelihood": total_likelihood,
        "samples": samples,
        "pre_hand_size": prestate["opponent_hand_size"],
        "method": METHOD,
    }
    if mixed_takes_truthful_finish:
        result["mixed_takes_truthful_finish"] = True
    return result
