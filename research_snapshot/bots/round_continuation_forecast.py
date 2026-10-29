"""Research-only exact forecasts for a fixed-rank continuation response.

The hidden-hand proposal is uniform over disjoint actor hand and earlier
center-pile allocations under conservative current rank bounds. This is not a
complete history posterior, optimal play, or a calibrated opponent model.
"""

from collections import Counter
from copy import deepcopy

from bots.round_card_memory import opponent_rank_bounds
from bots.round_continuation_exact import exact_continuation_distribution
from bots.round_honest_hypothesis_memory import honest_count_hypothesis_bounds
from bots.round_opening_forecast import STYLES, _style_weights
from bots.round_shared_pickup_knowledge import opponent_knows_our_rank_bounds
from cards import Card, Rank, Suit


METHOD = (
    "exact toy-style continuation under current hand/pile constraints; "
    "not a full history posterior; truthful-style earlier pile conditioned; "
    "conservative shared off-rank proofs included"
)


def _physical_cards(cards, name):
    try:
        result = tuple(cards)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of physical Cards") from exc
    if any(not isinstance(card, Card) or not isinstance(card.rank, Rank)
           or not isinstance(card.suit, Suit) for card in result):
        raise ValueError(f"{name} must contain Cards with canonical Rank and Suit")
    identities = [(card.rank, card.suit) for card in result]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} must not contain duplicate physical cards")
    return result


def _validated_view(hand, state, *, pending_opponent):
    if not isinstance(state, dict):
        raise TypeError("state must be a redacted observation dict")
    own_hand = _physical_cards(hand, "hand")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("invalid viewer")
    if ("viewer" in state and
            (type(state["viewer"]) is not int or state["viewer"] != viewer)):
        raise ValueError("viewer and player_id disagree")
    own_pile = _physical_cards(state.get("own_pile_cards"), "own_pile_cards")
    own_ids = {(card.rank, card.suit) for card in own_hand}
    if own_ids.intersection((card.rank, card.suit) for card in own_pile):
        raise ValueError("hand and own pile cards must be disjoint")
    own_hand_counts = Counter(card.rank for card in own_hand)
    own_pile_counts = Counter(card.rank for card in own_pile)
    if any(own_hand_counts[r] + own_pile_counts[r] > 4 for r in Rank):
        raise ValueError("known hand and pile cards exceed physical rank copies")

    sizes = tuple(state.get(key) for key in
                  ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size"))
    if (any(type(size) is not int or size < 0 for size in sizes)
            or sum(sizes) != 52 or sizes[0] != len(own_hand)
            or sizes[2] < len(own_pile) or sizes[3] <= 0
            or sizes[1] <= 0 or sizes[0] <= 0):
        raise ValueError("public sizes must be valid and sum to 52 with live hands and draw")
    actions = state.get("actions")
    pending = state.get("last_action")
    if not isinstance(actions, (tuple, list)) or not actions or pending is None:
        raise ValueError("an unresolved current-round packet is required")
    round_number = state.get("round_number")
    if type(round_number) is not int or round_number < 1:
        raise ValueError("invalid round number")
    active_rank = state.get("active_rank")
    if not isinstance(active_rank, Rank):
        raise ValueError("active_rank must be a canonical Rank")

    for index, action in enumerate(actions):
        player = getattr(action, "player", None)
        action_index = getattr(action, "action_index", None)
        action_round = getattr(action, "round_number", None)
        claim_rank = getattr(action, "claimed_rank", None)
        if (type(player) is not int or player not in (0, 1)
                or type(action_index) is not int or action_index != index
                or type(action_round) is not int or not 1 <= action_round <= round_number
                or not isinstance(claim_rank, Rank)):
            raise ValueError("action history metadata is malformed")
    current_indices = [i for i, action in enumerate(actions)
                       if action.round_number == round_number]
    if not current_indices or current_indices != list(range(current_indices[0], len(actions))):
        raise ValueError("current-round actions must form the history suffix")
    current = [actions[i] for i in current_indices]
    if any(action.claimed_rank != active_rank for action in current):
        raise ValueError("rank must remain fixed during the current round")
    for left, right in zip(current, current[1:]):
        if right.player == left.player:
            raise ValueError("current-round play actors must alternate")
        response_player = getattr(left, "response_player", None)
        if (getattr(left, "response_kind", None) != "play"
                or type(response_player) is not int or response_player not in (0, 1)
                or getattr(left, "response_player", None) != right.player
                or getattr(left, "response_was_call", None) is not False
                or getattr(left, "response_was_forced", None) is not False
                or getattr(left, "bluff_called", None) is not False
                or getattr(left, "caller_was_right", None) is not False):
            raise ValueError("prior current-round packet closure is inconsistent")
        before = getattr(left, "response_hand_size_before", None)
        if before is not None and (type(before) is not int or before < 0):
            raise ValueError("response hand-size metadata is malformed")

    latest = current[-1]
    def packet_size(action):
        cards = getattr(action, "cards_played", None)
        if not isinstance(cards, (list, tuple)) or not 1 <= len(cards) <= 4:
            raise ValueError("packet quantity must be from one through four")
        return len(cards)

    for action in current:
        packet_size(action)
        if action.player == viewer:
            _physical_cards(action.cards_played, "own action cards")
    latest_size = packet_size(latest)
    for index, (left, right) in enumerate(zip(current, current[1:])):
        before = getattr(left, "response_hand_size_before", None)
        if before is not None:
            responder = right.player
            remaining = state["hand_size"] if responder == viewer else state["opponent_hand_size"]
            remaining += sum(packet_size(action) for action in current[index + 1:]
                             if action.player == responder)
            if type(before) is not int or before != remaining:
                raise ValueError("response hand-size metadata is inconsistent")
    pending_player = getattr(pending, "player", None)
    pending_index = getattr(pending, "action_index", None)
    pending_round = getattr(pending, "round_number", None)
    pending_rank = getattr(pending, "claimed_rank", None)
    if (type(pending_player) is not int or pending_player not in (0, 1)
            or type(pending_index) is not int or pending_index < 0
            or type(pending_round) is not int or pending_round < 1
            or not isinstance(pending_rank, Rank)):
        raise ValueError("last_action metadata is malformed")
    metadata_fields = ("player", "action_index", "round_number", "claimed_rank")
    if any(getattr(pending, key, None) != getattr(latest, key, None)
           for key in metadata_fields) or packet_size(pending) != latest_size:
        raise ValueError("last_action must match the latest history packet")
    if getattr(latest, "response_kind", None) is not None:
        raise ValueError("latest packet response is already closed")
    if (getattr(latest, "bluff_called", None) is not False
            or getattr(latest, "caller_was_right", None) is not False
            or getattr(latest, "response_was_call", None) is not None
            or getattr(latest, "response_was_forced", None) is not None
            or getattr(latest, "response_player", None) is not None
            or getattr(latest, "response_hand_size_before", None) is not None):
        raise ValueError("latest packet is resolved or has malformed response flags")
    last_response = getattr(pending, "response_kind", None)
    if (last_response is not None
            or getattr(pending, "bluff_called", None) is not False
            or getattr(pending, "caller_was_right", None) is not False
            or getattr(pending, "response_was_call", None) is not None
            or getattr(pending, "response_was_forced", None) is not None
            or getattr(pending, "response_player", None) is not None
            or getattr(pending, "response_hand_size_before", None) is not None):
        raise ValueError("last_action is resolved")
    for field in ("response_kind", "bluff_called", "response_was_call",
                  "response_was_forced", "response_player"):
        if getattr(pending, field, None) != getattr(latest, field, None):
            raise ValueError("last_action response metadata must match history")
    if state.get("pile_size") != sum(packet_size(a) for a in current):
        raise ValueError("pile size must match unresolved current-round packets")
    expected_own_pile = [card for action in current if action.player == viewer
                         for card in action.cards_played]
    if own_pile != tuple(expected_own_pile):
        raise ValueError("own_pile_cards must match visible current-round packets")
    if (state.get("can_call_bluff") is not (pending_opponent is True)
            or state.get("can_pass") is not (pending_opponent is True)
            or state.get("can_play") is not (pending_opponent is True)):
        raise ValueError("legal response flags do not match the expected actor")

    if pending_opponent:
        if len(current) < 2 or latest.player == viewer:
            raise ValueError("a non-opening opponent continuation is required")
        if current[-2].player != viewer:
            raise ValueError("opponent continuation must follow the observer's play")
        if state.get("pile_size") <= latest_size:
            raise ValueError("opponent continuation must include an earlier packet")
        before_play = getattr(latest, "player_hand_size_before", None)
        if before_play is not None and (type(before_play) is not int
                                        or before_play != state["opponent_hand_size"] + latest_size):
            raise ValueError("opponent pre-play hand-size metadata is inconsistent")
    else:
        if latest.player != viewer:
            raise ValueError("response forecast requires the observer's latest play")
        if state["opponent_hand_size"] < 1:
            raise ValueError("empty opponent hand means the final packet was resolved")
        if len(current) > 1 and current[-2].player == viewer:
            raise ValueError("observer's latest packet did not follow the opponent")
        if getattr(latest, "player_hand_size_before", None) is not None:
            before_play = latest.player_hand_size_before
            expected = state["hand_size"] + latest_size
            if type(before_play) is not int or before_play != expected:
                raise ValueError("observer pre-play hand-size metadata is inconsistent")

    for index, action in enumerate(current):
        before_play = getattr(action, "player_hand_size_before", None)
        if before_play is not None:
            current_size = (state["hand_size"] if action.player == viewer
                            else state["opponent_hand_size"])
            expected = current_size + sum(
                packet_size(later) for later in current[index:]
                if later.player == action.player)
            if type(before_play) is not int or before_play != expected:
                raise ValueError("pre-play hand-size metadata is inconsistent")

    available = {rank: 4 - own_hand_counts[rank] - own_pile_counts[rank]
                 for rank in Rank}
    if any(value < 0 for value in available.values()):
        raise ValueError("known cards exceed the physical deck")
    hidden_pile = state["pile_size"] - len(own_pile)
    if hidden_pile < 0 or state["opponent_hand_size"] + hidden_pile > sum(available.values()):
        raise ValueError("hidden hand and earlier pile exceed available cards")
    return own_hand, own_pile, state, actions, current, latest, available, hidden_pile


def _event_rows(distribution):
    return [
        {"kind": kind, "quantity": quantity, "truth": truth, "probability": mass}
        for (kind, quantity, truth), mass in sorted(
            distribution.items(), key=lambda item: (item[0][0], item[0][1],
                                                     -1 if item[0][2] is None
                                                     else int(item[0][2])))
    ]


def _forecast(hand, state, style_weights, bluff_probability, call_probability,
              max_states, *, pending_opponent, use_honest_memory=False,
              mixed_respects_certificates=False,
              mixed_takes_truthful_finish=False):
    _, _, state, actions, current, latest, available, hidden_pile = (
        _validated_view(hand, state, pending_opponent=pending_opponent))
    weights = _style_weights(style_weights)
    lower, _ = opponent_rank_bounds(actions, state["player_id"], None)
    proposals = None
    if use_honest_memory:
        hypothesis = honest_count_hypothesis_bounds(state)
        if not hypothesis["compatible"]:
            reasons = ", ".join(hypothesis["incompatibility_reasons"])
            raise ValueError(
                "honest_max conditional distribution unsupported: honest-count "
                f"hypothesis is incompatible ({reasons})")
        conditional_lower = dict(lower)
        for rank, count in hypothesis["conditional_bounds"].items():
            conditional_lower[rank] = max(conditional_lower.get(rank, 0), count)
        proposals = conditional_lower
    # Reconstruct knowledge BEFORE our latest packet. Subtracting its hidden
    # quantity first would erase exactly the occupied-slot contradiction we
    # need: known other-rank cards cannot all fit in our remaining hand.
    before_packet = dict(state, actions=list(actions[:-1]))
    before_packet.setdefault("private_pickups", [])
    known_before = opponent_knows_our_rank_bounds(before_packet)
    other_certified_false = sum(
        count for rank, count in known_before.items()
        if rank != state["active_rank"]
    ) > state["hand_size"]
    result = exact_continuation_distribution(
        available, state["opponent_hand_size"], hidden_pile,
        state["active_rank"], len(latest.cards_played), lower_bounds=lower,
        other_certified_false=other_certified_false,
        mixed_respects_certificates=mixed_respects_certificates,
        mixed_takes_truthful_finish=mixed_takes_truthful_finish,
        honest_pile_truthful=True,
        bluff_probability=bluff_probability, call_probability=call_probability,
        max_states=max_states)
    baseline_allocation_count = result["random_uniform"]["allocation_count"]
    baseline_dp_states = result["random_uniform"]["dp_states"]
    honest_memory_support = "not_conditioned"
    if proposals is not None and proposals != lower:
        # These extra minima are a behavioral hypothesis, NOT certified card
        # facts. Legal observations may contradict them without invalidating
        # the observation or the other styles. Preserve the strict certified
        # integration above, and remove only this unsupported hypothesis.
        infeasible = (sum(proposals.values()) > state["opponent_hand_size"]
                      or any(count > available[rank]
                             for rank, count in proposals.items()))
        if infeasible:
            result["honest_max"] = {
                "distribution": {}, "allocation_count": 0, "dp_states": 0}
            honest_memory_support = "infeasible_conditional_hand_bounds"
        else:
            conditional = exact_continuation_distribution(
                available, state["opponent_hand_size"], hidden_pile,
                state["active_rank"], len(latest.cards_played), lower_bounds=proposals,
                other_certified_false=other_certified_false,
                mixed_respects_certificates=mixed_respects_certificates,
                mixed_takes_truthful_finish=mixed_takes_truthful_finish,
                honest_pile_truthful=True,
                bluff_probability=bluff_probability, call_probability=call_probability,
                max_states=max_states)
            result["honest_max"] = conditional["honest_max"]
            honest_memory_support = "conditioned"
    requested_weights = dict(weights)
    support_mass = sum(weights[style] for style in STYLES
                       if result[style]["allocation_count"] > 0)
    if support_mass <= 0:
        raise ValueError("no requested style supports the observed earlier pile")
    weights = {style: (weights[style] / support_mass
                      if result[style]["allocation_count"] > 0 else 0.0)
               for style in STYLES}
    style_rows = {style: _event_rows(result[style]["distribution"])
                  for style in STYLES}
    mixture = {}
    for style in STYLES:
        for (kind, quantity, truth), mass in result[style]["distribution"].items():
            key = (kind, quantity, truth)
            mixture[key] = mixture.get(key, 0.0) + weights[style] * mass
    style_metadata = {
        style: {"allocation_count": result[style]["allocation_count"],
                "dp_states": result[style]["dp_states"]}
        for style in STYLES
    }
    result = {
        "style_distributions": style_rows,
        "distribution": _event_rows(mixture),
        "style_weights": weights,
        "requested_style_weights": requested_weights,
        "honest_pile_truthful": True,
        "pre_hand_size": state["opponent_hand_size"],
        "hidden_pile_size": hidden_pile,
        "allocation_count": baseline_allocation_count,
        "dp_states": baseline_dp_states,
        "allocation_count_scope": "original certified bounds across all styles",
        "style_metadata": style_metadata,
        "honest_memory_used": use_honest_memory,
        "honest_memory_support": honest_memory_support,
        "other_certified_false": other_certified_false,
        "mixed_respects_certificates": mixed_respects_certificates,
        "method": METHOD,
    }
    if mixed_takes_truthful_finish:
        result["mixed_takes_truthful_finish"] = True
    return result


def forecast_continuation_response(
    hand, state, style_weights=None, bluff_probability=0.3,
    call_probability=0.35, max_states=10000, *, use_honest_memory=False,
    mixed_respects_certificates=False,
    mixed_takes_truthful_finish=False,
):
    """Forecast play/pass/challenge after the observer's unresolved packet."""
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")
    if type(use_honest_memory) is not bool:
        raise TypeError("use_honest_memory must be a bool")
    if type(mixed_respects_certificates) is not bool:
        raise TypeError("mixed_respects_certificates must be a bool")
    if type(mixed_takes_truthful_finish) is not bool:
        raise TypeError("mixed_takes_truthful_finish must be a bool")
    return _forecast(hand, state, style_weights, bluff_probability,
                     call_probability, max_states, pending_opponent=False,
                     use_honest_memory=use_honest_memory,
                     mixed_respects_certificates=mixed_respects_certificates,
                     mixed_takes_truthful_finish=mixed_takes_truthful_finish)


def forecast_continuation_claim(
    hand, state, style_weights=None, bluff_probability=0.3,
    call_probability=0.35, max_states=10000, *, use_honest_memory=False,
    mixed_respects_certificates=False,
    mixed_takes_truthful_finish=False,
):
    """Forecast an unresolved opponent continuation claim by its size only."""
    if type(max_states) is not int or not 1 <= max_states <= 10000:
        raise ValueError("max_states must be an integer from 1 through 10000")
    if type(use_honest_memory) is not bool:
        raise TypeError("use_honest_memory must be a bool")
    if type(mixed_respects_certificates) is not bool:
        raise TypeError("mixed_respects_certificates must be a bool")
    if type(mixed_takes_truthful_finish) is not bool:
        raise TypeError("mixed_takes_truthful_finish must be a bool")
    view = _validated_view(hand, state, pending_opponent=True)
    _, _, state, actions, current, pending, _, _ = view
    quantity = len(pending.cards_played)
    prestate = dict(state)
    prestate["actions"] = deepcopy(list(actions[:-1]))
    previous = prestate["actions"][-1]
    previous.response_kind = None
    previous.response_player = None
    previous.response_was_call = None
    previous.response_was_forced = None
    previous.response_hand_size_before = None
    previous.bluff_called = False
    prestate["last_action"] = previous
    prestate["opponent_hand_size"] += quantity
    prestate["pile_size"] -= quantity
    prestate["can_play"] = False
    prestate["can_call_bluff"] = False
    prestate["can_pass"] = False
    prior = forecast_continuation_response(
        hand, prestate, style_weights, bluff_probability, call_probability,
        max_states, use_honest_memory=use_honest_memory,
        mixed_respects_certificates=mixed_respects_certificates,
        mixed_takes_truthful_finish=mixed_takes_truthful_finish)
    weights = prior["style_weights"]
    distributions = {
        style: {(row["kind"], row["quantity"], row["truth"]): row["probability"]
                for row in prior["style_distributions"][style]}
        for style in STYLES
    }
    style_stats = {}
    for style in STYLES:
        truth = distributions[style].get(("play", quantity, True), 0.0)
        bluff = distributions[style].get(("play", quantity, False), 0.0)
        likelihood = truth + bluff
        style_stats[style] = {
            "joint_truth": truth,
            "joint_bluff": bluff,
            "likelihood": likelihood,
            "conditional_bluff": bluff / likelihood if likelihood else None,
            "allocation_count": prior["style_metadata"][style]["allocation_count"],
            "dp_states": prior["style_metadata"][style]["dp_states"],
        }
    total_likelihood = sum(weights[s] * style_stats[s]["likelihood"] for s in STYLES)
    posterior = ({s: weights[s] * style_stats[s]["likelihood"] / total_likelihood
                 for s in STYLES} if total_likelihood else {s: None for s in STYLES})
    true_mass = sum(weights[s] * style_stats[s]["joint_truth"] for s in STYLES)
    false_mass = sum(weights[s] * style_stats[s]["joint_bluff"] for s in STYLES)
    total = true_mass + false_mass
    result = {
        "style_stats": style_stats,
        "style_posterior": posterior,
        "probability_bluff": false_mass / total if total else None,
        "total_likelihood": total_likelihood,
        "prior_response_forecast": prior,
        "pre_hand_size": prior["pre_hand_size"],
        "hidden_pile_size": prior["hidden_pile_size"],
        "mixed_respects_certificates": mixed_respects_certificates,
        "method": METHOD,
    }
    if mixed_takes_truthful_finish:
        result["mixed_takes_truthful_finish"] = True
    return result
