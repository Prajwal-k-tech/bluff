"""Conservative opponent rank bounds from player-visible round history.

No claim is treated as a truthful card identity. Bounds are deal-local and
reconstructed, not persisted as behavioral evidence. They are guaranteed
minimum counts, not a complete posterior over correlated hidden hands.
"""

from collections import Counter

from cards import Card, Rank


def opponent_rank_bounds(actions, viewer, pending=None):
    """Return current bounds and bounds just before the pending claim.

    Only own packets and legally revealed packets enter the known center.
    A hidden opponent packet can consume up to q known copies of EACH rank;
    weakening every rank separately is conservative, not an exact belief.
    """
    known = Counter()
    center = Counter()
    before_pending = Counter()
    previous_round = None
    pending_index = getattr(pending, "action_index", None)
    for action in actions or ():
        round_number = getattr(action, "round_number", None)
        if previous_round is not None and round_number != previous_round:
            center.clear()
        previous_round = round_number
        if (action is pending or (type(pending_index) is int
                and getattr(action, "action_index", None) == pending_index)):
            before_pending = known.copy()
        cards = list(getattr(action, "cards_played", ()) or ())
        revealed = getattr(action, "bluff_called", False) is True
        visible = action.player == viewer or revealed
        visible_counts = Counter(card.rank for card in cards
                                 if visible and isinstance(card, Card))
        if action.player != viewer:
            for rank in Rank:
                consumed = visible_counts[rank] if revealed else len(cards)
                known[rank] = max(0, known[rank] - consumed)
        center.update(visible_counts)
        if revealed:
            # Access the honesty label only after a legal challenge reveal.
            was_bluff = getattr(action, "was_bluff", None)
            if type(was_bluff) is not bool:
                raise ValueError("revealed packet must expose its outcome")
            loser = action.player if was_bluff else 1 - action.player
            if loser != viewer:
                known.update(center)
            center.clear()
        elif getattr(action, "response_kind", None) == "pass":
            # The recycled supply is unknown, but already-held opponent cards
            # remain held. A private draw adds no guaranteed known copies.
            center.clear()
    return dict(known), dict(before_pending)
