"""Research-only response forecast after a hypothetical legal own play."""

from copy import deepcopy
from math import isfinite

from bots.round_continuation_forecast import forecast_continuation_response
from cards import Card, Rank, Suit
from game import Action
from bots.round_recycled_history import allowed_recycled_indices


def _physical_cards(cards, name):
    try:
        result = tuple(cards)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc
    if any(not isinstance(card, Card) or not isinstance(card.rank, Rank)
           or not isinstance(card.suit, Suit) for card in result):
        raise ValueError(f"{name} must contain canonical physical Cards")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate physical cards")
    return result


def _stage(hand, state, packet, rank):
    if not isinstance(state, dict):
        raise TypeError("state must be a redacted observation dict")
    own_hand = _physical_cards(hand, "hand")
    play = _physical_cards(packet, "packet")
    if not 1 <= len(play) <= 4:
        raise ValueError("packet must contain one through four cards")
    if any(card not in own_hand for card in play):
        raise ValueError("packet cards must be owned")
    if not isinstance(rank, Rank):
        raise ValueError("rank must be a canonical Rank")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("invalid viewer")
    if ("viewer" in state and (type(state["viewer"]) is not int
                                or state["viewer"] != viewer)):
        raise ValueError("viewer and player_id disagree")
    sizes = tuple(state.get(k) for k in
                  ("hand_size", "opponent_hand_size", "pile_size", "draw_pile_size"))
    if (any(type(n) is not int or n < 0 for n in sizes)
            or sizes[0] != len(own_hand) or sum(sizes) != 52
            or sizes[0] < 1 or sizes[1] < 1 or sizes[3] < 1):
        raise ValueError("public counts must be valid and sum to 52")
    if state.get("can_play") is not True:
        raise ValueError("observation must be this player's legal play turn")
    active = state.get("active_rank")
    if active is not None and not isinstance(active, Rank):
        raise ValueError("active_rank must be a canonical Rank or None")
    if active is not None and rank != active:
        raise ValueError("claimed rank must match the fixed round rank")

    actions = state.get("actions")
    if not isinstance(actions, (tuple, list)):
        raise ValueError("actions must be a public action history")
    prior = deepcopy(list(actions))
    recycled_unknown = allowed_recycled_indices(state)
    last = state.get("last_action")
    round_number = state.get("round_number")
    if type(round_number) is not int or round_number < 1:
        raise ValueError("invalid round number")
    for index, action in enumerate(prior):
        action_round = getattr(action, "round_number", None)
        if (type(getattr(action, "player", None)) is not int
                or action.player not in (0, 1)
                or type(getattr(action, "action_index", None)) is not int
                or action.action_index != index
                or type(action_round) is not int or not 1 <= action_round <= round_number
                or not isinstance(getattr(action, "claimed_rank", None), Rank)
                or not 1 <= len(getattr(action, "cards_played", ())) <= 4):
            raise ValueError("public action history metadata is malformed")
        if action.player == viewer and index not in recycled_unknown:
            _physical_cards(action.cards_played, "own action cards")
    current = [action for action in prior
               if getattr(action, "round_number", None) == round_number]
    if current and [a.action_index for a in current] != list(
            range(current[0].action_index, len(prior))):
        raise ValueError("current-round actions must form the history suffix")
    if current and any(action.claimed_rank != active for action in current):
        raise ValueError("active rank must match the current-round history")
    for left, right in zip(current, current[1:]):
        if (left.player == right.player
                or getattr(left, "response_kind", None) != "play"
                or getattr(left, "response_player", None) != right.player
                or getattr(left, "response_was_call", None) is not False
                or getattr(left, "response_was_forced", None) is not False
                or getattr(left, "bluff_called", None) is not False):
            raise ValueError("prior current-round response metadata is inconsistent")
        before = getattr(left, "response_hand_size_before", None)
        responder_size = sizes[0] if right.player == viewer else sizes[1]
        right_index = current.index(right)
        remaining_plays = sum(len(action.cards_played)
                              for action in current[right_index:]
                              if action.player == right.player)
        if before is not None and (type(before) is not int
                                   or before != responder_size + remaining_plays):
            raise ValueError("prior response hand-size metadata is inconsistent")
    if last is not None:
        if not prior:
            raise ValueError("pending action is absent from history")
        if state.get("can_call_bluff") is not True or state.get("can_pass") is not True:
            raise ValueError("pending opponent claim must expose legal response choices")
        pending = prior[-1]
        if (type(getattr(last, "player", None)) is not int
                or getattr(last, "player", None) not in (0, 1)
                or type(getattr(last, "action_index", None)) is not int
                or type(getattr(last, "round_number", None)) is not int
                or not isinstance(getattr(last, "claimed_rank", None), Rank)):
            raise ValueError("detached latest action metadata is malformed")
        for field in ("player", "claimed_rank", "action_index", "round_number"):
            if getattr(last, field, None) != getattr(pending, field, None):
                raise ValueError("last_action does not match public history")
        if (pending.player == viewer or not current or current[-1] is not pending
                or pending.response_kind is not None
                or pending.response_was_call is not None
                or pending.response_was_forced is not None
                or pending.response_player is not None
                or getattr(pending, "response_hand_size_before", None) is not None
                or pending.bluff_called is not False
                or len(getattr(last, "cards_played", ()))
                != len(getattr(pending, "cards_played", ()))
                or len(getattr(pending, "cards_played", ())) not in range(1, 5)):
            raise ValueError("latest action must be an unresolved opponent claim")
        for field in ("response_kind", "response_was_call", "response_was_forced",
                      "response_player", "response_hand_size_before", "bluff_called"):
            if getattr(last, field, None) != getattr(pending, field, None):
                raise ValueError("detached latest action response metadata disagrees")
        pending.response_kind = "play"
        pending.response_player = viewer
        pending.response_was_call = False
        pending.response_was_forced = False
        pending.response_hand_size_before = len(own_hand)
    elif (active is not None or sizes[2] != 0 or current
          or state.get("own_pile_cards") not in ([], ())
          or state.get("can_pass") is not False
          or state.get("can_call_bluff") is not False):
        raise ValueError("new-round play must have no pending packet or active rank")

    own_pile = _physical_cards(state.get("own_pile_cards", ()), "own_pile_cards")
    if set(own_hand).intersection(own_pile):
        raise ValueError("own hand and own pile cards must be disjoint")
    if any(sum(card.rank == rank for card in own_hand + own_pile) > 4 for rank in Rank):
        raise ValueError("known cards exceed physical rank copies")
    if last is not None:
        current_sizes = [len(getattr(action, "cards_played", ())) for action in current]
        expected_own_pile = tuple(card for action in current if action.player == viewer
                                  for card in _physical_cards(action.cards_played, "own action"))
        if (sizes[2] != sum(current_sizes) or own_pile != expected_own_pile
                or sum(current_sizes) > 52):
            raise ValueError("current-round public pile metadata is inconsistent")
    elif prior and prior[-1].round_number == round_number:
        raise ValueError("current-round history requires an unresolved latest action")

    if len(play) == len(own_hand):
        return own_hand, play, viewer, None, bool(any(c.rank != rank for c in play))

    staged_actions = prior
    staged = dict(state)
    staged["actions"] = staged_actions
    staged["hand_size"] = len(own_hand) - len(play)
    staged["pile_size"] = sizes[2] + len(play)
    staged["own_pile_cards"] = list(state.get("own_pile_cards", ())) + list(play)
    staged["can_play"] = staged["can_pass"] = staged["can_call_bluff"] = False
    staged["active_rank"] = rank
    action = Action(viewer, list(play), rank,
                    any(card.rank != rank for card in play), False, False, sizes[2])
    action.action_index = len(staged_actions)
    action.round_number = state.get("round_number")
    action.response_kind = None
    action.response_player = None
    action.response_hand_size_before = None
    action.response_was_call = None
    action.response_was_forced = None
    staged_actions.append(action)
    staged["last_action"] = action
    remaining_hand = tuple(card for card in own_hand if card not in set(play))
    return remaining_hand, play, viewer, staged, action.was_bluff


def forecast_candidate_response(hand, state, packet, rank, **forecastkwargs):
    """Forecast the opponent's response after a hypothetical own packet.

    Full-hand packets are automatically challenged by the rules, so this
    returns that forced event without consulting the continuation model.
    """
    post_hand, _, _, staged, was_bluff = _stage(hand, state, packet, rank)
    if staged is None:
        return {"forced": True, "own_truthful": not was_bluff,
                "probability_call": 1.0, "response_forecast": None}
    result = forecast_continuation_response(post_hand, staged, **forecastkwargs)
    if not isinstance(result, dict) or not isinstance(result.get("distribution"), list):
        raise ValueError("response forecast has an unsupported result shape")
    call_mass = 0.0
    total = 0.0
    for event in result["distribution"]:
        if (not isinstance(event, dict)
                or event.get("kind") not in ("play", "pass", "call_bluff")
                or type(event.get("probability")) not in (int, float)
                or not isfinite(event["probability"]) or event["probability"] < 0):
            raise ValueError("response forecast contains a malformed event")
        total += event["probability"]
        if event["kind"] == "call_bluff":
            call_mass += event["probability"]
    if not isfinite(total) or abs(total - 1.0) > 1e-8:
        raise ValueError("response forecast event probabilities must sum to one")
    return {"forced": False, "own_truthful": not was_bluff,
            "probability_call": call_mass, "response_forecast": result}
