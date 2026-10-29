"""Conservative reconstruction of what an opponent knows about our hand.

This is a derived legal-knowledge view for reasoning about proof calls. It
does not condition on whether either player is honest or strategically sound.
"""

from copy import copy, deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
import pickle

from game import Action

from bots.round_card_memory import opponent_rank_bounds
from bots.round_private_pickup_evidence import extract_private_pickup_evidence


_shared_knowledge_cache = ContextVar("round_shared_knowledge_cache", default=None)


@contextmanager
def shared_knowledge_cache():
    """Cache derived bounds within one decision, restoring any outer context."""
    token = _shared_knowledge_cache.set({})
    try:
        yield
    finally:
        _shared_knowledge_cache.reset(token)


def _copy_actions(actions):
    # Only field replacement occurs below; rank-bound inference reads nested
    # values without mutating them. Preserve extension copy hooks otherwise.
    if all(type(action) is Action for action in actions):
        return [copy(action) for action in actions]
    return deepcopy(list(actions))


def _opponent_knows_our_rank_bounds_uncached(state):
    """Return ``{rank: minimum_count}`` the opponent can know in our hand.

    ``state`` must be our detached legal player view. Root-owned unchallenged
    packets are hidden in the simulated opponent view; collector-private
    pickup cards are restored only for the opponent's own old packets, whose
    identities the opponent necessarily knew when playing them.
    """
    if not isinstance(state, dict):
        raise TypeError("state must be a player-view dict")
    root = state.get("player_id")
    if type(root) is not int or root not in (0, 1):
        raise ValueError("player_id must be 0 or 1")
    if "viewer" in state and (type(state["viewer"]) is not int
                              or state["viewer"] != root):
        raise ValueError("viewer and player_id disagree")
    actions = state.get("actions")
    if not isinstance(actions, (list, tuple)):
        raise ValueError("actions must be a sequence")

    # This validates that the private channel belongs to root and that every
    # pickup is a complete, ordered, resolved-round record.
    extract_private_pickup_evidence(state)
    restored = {}
    for pickup in state.get("private_pickups", ()):
        for packet in pickup:
            restored[packet["action_index"]] = list(packet["cards"])

    metaactions = _copy_actions(actions)
    for i, action in enumerate(metaactions):
        if getattr(action, "action_index", None) != i:
            raise ValueError("actions must have contiguous action indices")
        if type(getattr(action, "player", None)) is not int or action.player not in (0, 1):
            raise ValueError("action player must be 0 or 1")
        cards = list(getattr(action, "cards_played", ()) or ())
        if (action.player == root and action.bluff_called is not True
                and getattr(action, "cards_revealed", False) is not True):
            # Never leak root's private past identity/truth into opponent
            # inference, including when the original legal view exposes it.
            action.cards_played = [None] * len(cards)
            action.was_bluff = None
        elif action.player == 1 - root and i in restored:
            # The opponent knows every card in its own packet. This partial
            # target-hand knowledge is sound even if other packets are absent.
            action.cards_played = restored[i]

    bounds, _ = opponent_rank_bounds(metaactions, viewer=1 - root)
    return {rank: count for rank, count in bounds.items() if count > 0}


def opponent_knows_our_rank_bounds(state):
    """Return rank bounds, reusing exact legal-view calculations per decision."""
    cache = _shared_knowledge_cache.get()
    if cache is None or not isinstance(state, dict):
        return _opponent_knows_our_rank_bounds_uncached(state)

    actions = state.get("actions")
    if not isinstance(actions, (list, tuple)) or any(type(action) is not Action for action in actions):
        return _opponent_knows_our_rank_bounds_uncached(state)

    dependencies = (
        state.get("player_id"),
        ("viewer" in state, state.get("viewer")),
        actions,
        state.get("private_pickups"),
    )
    try:
        key = pickle.dumps(dependencies, protocol=5)
    except Exception:
        return _opponent_knows_our_rank_bounds_uncached(state)
    if key in cache:
        return dict(cache[key])

    result = _opponent_knows_our_rank_bounds_uncached(state)
    cache[key] = dict(result)
    return dict(result)
