"""Conservative opponent rank bounds from player-visible round history.

No claim is treated as a truthful card identity. Bounds are deal-local and
reconstructed, not persisted as behavioral evidence. They are guaranteed
minimum counts, not a complete posterior over correlated hidden hands.
"""

from collections import Counter

from cards import Card, Rank

_RANK_SET = frozenset(Rank)

def opponent_hand_and_center_bounds(actions, viewer, round_number, *, current_bounds=None):
    """Guaranteed rank counts in opponent hand PLUS its current center packets.

    Spending an unknown card into the center weakens a held-card bound but not
    this combined-zone bound: the opponent still knows that card's identity.
    No cards enter/leave these combined zones within an unresolved round.
    Once a round resolves, use the ordinary transfer/recycle reconstruction.
    """
    actions = list(actions or ())
    current = (opponent_rank_bounds(actions, viewer)[0]
               if current_bounds is None else current_bounds)
    if type(round_number) is not int:
        return current
    start = next((i for i, action in enumerate(actions)
                  if getattr(action, "round_number", None) == round_number), None)
    if start is None:
        return current
    round_actions = actions[start:]
    if any(getattr(action, "bluff_called", False) is True
           or getattr(action, "response_kind", None) == "pass"
           for action in round_actions):
        return current
    at_start, _ = opponent_rank_bounds(actions[:start], viewer)
    return at_start


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
    seen_opponent = False
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
        challenged = getattr(action, "bluff_called", False) is True
        revealed = challenged or getattr(action, "cards_revealed", False) is True
        visible = action.player == viewer or revealed
        visible_counts = Counter(card.rank for card in cards
                                 if visible and isinstance(card, Card))
        if action.player != viewer:
            # Preserve the original zero-valued keys as well as positive bounds.
            # Most hidden plays cannot consume any known cards; avoid repeatedly
            # doing Counter lookups/max arithmetic for every zero rank.
            if not seen_opponent:
                for rank in Rank:
                    known.setdefault(rank,0)
                seen_opponent = True
            if revealed:
                for rank,consumed in visible_counts.items():
                    if rank in _RANK_SET:
                        known[rank] = max(0,known[rank]-consumed)
            else:
                for rank,count in known.items():
                    if count and rank in _RANK_SET:
                        known[rank] = max(0,count-len(cards))
        center.update(visible_counts)
        if challenged:
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
