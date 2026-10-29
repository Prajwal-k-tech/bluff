"""Extract factual opponent packet labels from collector-private pickups.

This is an offline/player-view evidence adapter. It does not rewrite public
actions or turn private pickup identities into public challenge labels.
"""

from cards import Card, Rank, Suit


def extract_visible_packet_evidence(state):
    """Opponent labels exposed publicly or in an authorized historical pickup.

    Card exposure is distinct from a challenge response. Earlier packets gain
    labels only once their round has actually resolved by a challenge.
    """
    private = extract_private_pickup_evidence(state)
    labels = {row["action_index"]: row for row in private}
    rounds = {}
    for action in state["actions"]:
        rounds.setdefault(action.round_number, []).append(action)
    for packets in rounds.values():
        latest = packets[-1]
        for action in packets:
            if action.player == state["player_id"]:
                continue
            if (getattr(action, "cards_revealed", False) is not True
                    and action.bluff_called is not True):
                continue
            if (latest.bluff_called is not True
                    or getattr(latest, "response_kind", None) != "challenge"
                    or getattr(latest, "response_was_call", None) is not True
                    or type(getattr(latest, "response_was_forced", None)) is not bool
                    or getattr(latest, "response_player", None) != 1-latest.player
                    or type(latest.was_bluff) is not bool
                    or latest.caller_was_right is not latest.was_bluff):
                raise ValueError("public packet reveal lacks a resolved round")
            cards = action.cards_played
            if (any(not isinstance(c, Card) or not isinstance(c.rank, Rank)
                    or not isinstance(c.suit, Suit) for c in cards)
                    or len(set(cards)) != len(cards)
                    or type(action.was_bluff) is not bool
                    or any(c.rank != action.claimed_rank for c in cards) != action.was_bluff):
                raise ValueError("public packet reveal has inconsistent cards or truth")
            labels[action.action_index] = {
                "action_index": action.action_index, "rank": action.claimed_rank,
                "quantity": len(cards), "was_bluff": action.was_bluff,
                "evidence_source": "public_reveal"}
    return [labels[index] for index in sorted(labels)]


def extract_private_pickup_evidence(state):
    """Return deduplicated opponent packet labels from a legal player view.

    Each result describes one opponent action whose exact cards became known
    because this viewer collected the pile. Malformed or leaked channels fail
    closed with ``ValueError``.
    """
    if not isinstance(state, dict):
        raise TypeError("state must be a player-view dict")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("player_id must be 0 or 1")
    if "viewer" in state and (type(state["viewer"]) is not int or
                               state["viewer"] != viewer):
        raise ValueError("viewer and player_id disagree")
    actions = state.get("actions")
    pickups = state.get("private_pickups")
    if not isinstance(actions, (list, tuple)) or not isinstance(pickups, (list, tuple)):
        raise ValueError("actions and private_pickups must be sequences")

    action_rounds = {}
    for i, action in enumerate(actions):
        idx = getattr(action, "action_index", None)
        player = getattr(action, "player", None)
        round_number = getattr(action, "round_number", None)
        cards = getattr(action, "cards_played", None)
        rank = getattr(action, "claimed_rank", None)
        if (type(idx) is not int or idx != i or type(player) is not int
                or player not in (0, 1) or type(round_number) is not int
                or round_number < 1 or not isinstance(rank, Rank)
                or not isinstance(cards, (list, tuple)) or not 1 <= len(cards) <= 4):
            raise ValueError("malformed public action history")
        action_rounds.setdefault(round_number, []).append((idx, action))

    output = []
    seen = set()
    for pickup in pickups:
        if not isinstance(pickup, (list, tuple)) or not pickup:
            raise ValueError("malformed private pickup")
        indices = []
        card_seen = set()
        rows = []
        for packet in pickup:
            if not isinstance(packet, dict) or set(packet) != {"action_index", "cards", "claimed_rank"}:
                raise ValueError("malformed private pickup packet")
            idx, cards, rank = packet["action_index"], packet["cards"], packet["claimed_rank"]
            if (type(idx) is not int or not isinstance(cards, (list, tuple))
                    or not 1 <= len(cards) <= 4 or not isinstance(rank, Rank)):
                raise ValueError("malformed private pickup packet")
            if any(not isinstance(c, Card) or not isinstance(c.rank, Rank)
                   or not isinstance(c.suit, Suit) for c in cards):
                raise ValueError("invalid private pickup card")
            if len(set(cards)) != len(cards) or any(c in card_seen for c in cards):
                raise ValueError("duplicate card in private pickup")
            card_seen.update(cards)
            indices.append(idx)
            rows.append((idx, list(cards), rank))
        if indices != sorted(set(indices)):
            raise ValueError("private pickup action indices must be unique and ordered")

        round_number = getattr(actions[indices[0]], "round_number", None) if indices and 0 <= indices[0] < len(actions) else None
        round_rows = action_rounds.get(round_number, [])
        if not round_rows or [i for i, _ in round_rows] != indices:
            raise ValueError("pickup must contain the complete ordered round packet history")
        latest = round_rows[-1][1]
        if (getattr(latest, "response_kind", None) != "challenge"
                or getattr(latest, "bluff_called", None) is not True
                or getattr(latest, "response_was_call", None) is not True
                or getattr(latest, "response_player", None) != 1 - latest.player
                or getattr(latest, "caller_was_right", None) != getattr(latest, "was_bluff", None)):
            raise ValueError("pickup round was not completely resolved by a challenge")
        collector = latest.player if latest.was_bluff else 1 - latest.player
        if collector != viewer:
            raise ValueError("private pickup channel is not authorized for this viewer")

        for (idx, exact_cards, exact_rank), (_, action) in zip(rows, round_rows):
            if (idx != action.action_index or exact_rank != action.claimed_rank
                    or len(exact_cards) != len(action.cards_played)):
                raise ValueError("private pickup does not match public packet metadata")
            # Own packets and the challenged latest packet are independently
            # visible in the legal action history, so require exact agreement.
            visible = (action.player == viewer or action is latest
                       or getattr(action, "cards_revealed", False) is True)
            public_cards = list(action.cards_played)
            if visible:
                if public_cards != exact_cards:
                    raise ValueError("private pickup disagrees with revealed packet")
                actual_bluff = any(card.rank != exact_rank for card in exact_cards)
                if type(getattr(action, "was_bluff", None)) is not bool or action.was_bluff != actual_bluff:
                    raise ValueError("revealed packet truth metadata disagrees with cards")
            elif (any(card is not None for card in public_cards)
                  or getattr(action, "was_bluff", None) is not None):
                raise ValueError("unrevealed opponent packet leaked through actions")
            if idx in seen:
                continue
            seen.add(idx)
            if action.player != viewer:
                output.append({
                    "action_index": idx,
                    "rank": exact_rank,
                    "quantity": len(exact_cards),
                    "was_bluff": any(card.rank != exact_rank for card in exact_cards),
                    "evidence_source": "private_pickup",
                })
    return output
