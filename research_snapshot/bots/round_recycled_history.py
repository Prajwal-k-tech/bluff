"""Explicit search-only support for identity-independent passed own packets."""
from cards import Card, Rank

FLAG = "_search_recycled_own_history"


def recycled_unknown_indices(state):
    """Validate and return unknown own packet indices, without completing them."""
    viewer, current = state.get("player_id"), state.get("round_number")
    actions = state.get("actions")
    if type(viewer) is not int or viewer not in (0, 1) or type(current) is not int or current < 1:
        raise ValueError("invalid projection viewer/round")
    if (("viewer" in state and (type(state["viewer"]) is not int or state["viewer"] != viewer))
            or not isinstance(actions, (list, tuple))):
        raise ValueError("invalid projection observation")
    groups, unknown, previous = {}, set(), 0
    for index, action in enumerate(actions):
        if (getattr(action,"action_index",None) != index or type(action.action_index) is not int
                or type(getattr(action,"player",None)) is not int or action.player not in (0, 1)
                or type(getattr(action,"round_number",None)) is not int
                or not 1 <= action.round_number <= current
                or not isinstance(getattr(action,"claimed_rank",None), Rank)):
            raise ValueError("malformed projection history")
        if action.round_number not in (previous, previous+1):
            raise ValueError("projection history must have consecutive ordered rounds")
        previous = action.round_number
        groups.setdefault(action.round_number, []).append(action)
        cards = getattr(action,"cards_played",None)
        if not isinstance(cards,(list,tuple)) or not 1 <= len(cards) <= 4:
            raise ValueError("invalid projection packet")
        if action.player == viewer and any(not isinstance(c,Card) for c in cards):
            if (any(c is not None for c in cards) or action.round_number >= current
                    or getattr(action,"was_bluff",None) is not None):
                raise ValueError("unknown own packet is not a fully redacted past packet")
            unknown.add(index)
    for number in {actions[index].round_number for index in unknown}:
        group = groups[number]
        if number+1 in groups and groups[number+1][0].player != group[-1].player:
            raise ValueError("passed-round next starter is inconsistent")
        if [a.action_index for a in group] != list(range(group[0].action_index,group[-1].action_index+1)):
            raise ValueError("passed round must be contiguous")
        for offset, action in enumerate(group):
            final = offset == len(group)-1
            if (action.claimed_rank != group[0].claimed_rank
                    or (offset and action.player == group[offset-1].player)
                    or getattr(action,"response_kind",None) != ("pass" if final else "play")
                    or type(getattr(action,"response_player",None)) is not int
                    or action.response_player != 1-action.player
                    or getattr(action,"response_was_call",None) is not False
                    or getattr(action,"response_was_forced",None) is not False
                    or getattr(action,"bluff_called",None) is not False
                    or getattr(action,"caller_was_right",None) is not False
                    or getattr(action,"cards_revealed",False) is not False):
                raise ValueError("unknown own history is not a closed unrevealed passed round")
    return frozenset(unknown)


def allowed_recycled_indices(state):
    flag = state.get(FLAG, False)
    if type(flag) is not bool:
        raise ValueError("search history flag must be boolean")
    return recycled_unknown_indices(state) if flag else frozenset()


def projected_recycled_view(state):
    projected = dict(state)
    projected[FLAG] = True
    recycled_unknown_indices(projected)
    return projected
