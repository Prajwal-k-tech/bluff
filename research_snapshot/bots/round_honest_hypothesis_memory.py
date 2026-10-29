"""Conditional rank bounds under an honest-count opponent hypothesis.

These are hypothesis-conditioned research signals, never certified facts.
"""

from collections import Counter

from cards import Card, Rank
from bots.round_recycled_history import allowed_recycled_indices


def honest_count_hypothesis_bounds(state):
    """Infer opponent rank lower bounds from one actor's legal round view.

    Assumes the opponent plays truthfully and challenges only proved false
    packets. A contradiction marks the hypothesis incompatible and suppresses
    all bounds. Hidden cards in a transferred pile are never counted.
    """
    if not isinstance(state, dict):
        raise TypeError("state must be a player-view dict")
    viewer = state.get("player_id")
    if type(viewer) is not int or viewer not in (0, 1):
        raise ValueError("player_id must be 0 or 1")
    if ("viewer" in state and
            (type(state["viewer"]) is not int or state["viewer"] != viewer)):
        raise ValueError("viewer and player_id disagree")
    actions = state.get("actions")
    if not isinstance(actions, (tuple, list)):
        raise ValueError("actions must be a public action history")

    target = 1 - viewer
    hand_lower = Counter()
    target_center = Counter()
    viewer_center = Counter()
    caller_knows_viewer_hand = Counter()
    reasons = []
    previous_round = 0
    recycled_unknown = allowed_recycled_indices(state)

    for index, action in enumerate(actions):
        player = getattr(action, "player", None)
        action_index = getattr(action, "action_index", None)
        round_number = getattr(action, "round_number", None)
        rank = getattr(action, "claimed_rank", None)
        cards = getattr(action, "cards_played", None)
        if (type(player) is not int or player not in (0, 1)
                or type(action_index) is not int or action_index != index
                or type(round_number) is not int or round_number < 1
                or round_number < previous_round or round_number > previous_round + 1
                or not isinstance(rank, Rank)
                or not isinstance(cards, (tuple, list))
                or not 1 <= len(cards) <= 4):
            raise ValueError("malformed public action history")
        if round_number != previous_round:
            target_center.clear()
            viewer_center.clear()
            previous_round = round_number
        caller_known_before_packet = caller_knows_viewer_hand.copy()

        called = getattr(action, "bluff_called", None)
        was_bluff = getattr(action, "was_bluff", None)
        response = getattr(action, "response_kind", None)
        forced = getattr(action, "response_was_forced", None)
        response_player = getattr(action, "response_player", None)
        claimant_before = getattr(action, "player_hand_size_before", None)
        caller_before = getattr(action, "response_hand_size_before", None)
        if (claimant_before is not None
                and (type(claimant_before) is not int or claimant_before < len(cards))):
            raise ValueError("invalid claimant pre-play hand size")
        if caller_before is not None and (type(caller_before) is not int or caller_before < 0):
            raise ValueError("invalid responder pre-response hand size")
        if type(called) is not bool or response not in (None, "play", "pass", "challenge"):
            raise ValueError("invalid response metadata")
        if response is None:
            if called or forced is not None or response_player is not None:
                raise ValueError("unresolved action has response metadata")
        elif (type(response_player) is not int or response_player != 1 - player
              or type(forced) is not bool
              or called != (response == "challenge")):
            raise ValueError("closed response metadata is inconsistent")
        response_was_call = getattr(action, "response_was_call", None)
        if response == "challenge" and response_was_call is not True:
            raise ValueError("challenge response metadata is inconsistent")
        if response in ("play", "pass") and response_was_call is not False:
            raise ValueError("non-challenge response metadata is inconsistent")

        revealed = called is True or getattr(action, "cards_revealed", False) is True
        if player == viewer:
            if index not in recycled_unknown:
                if any(not isinstance(card, Card) for card in cards):
                    raise ValueError("viewer packet cards must be visible")
                actual_bluff = any(card.rank != rank for card in cards)
                if type(was_bluff) is not bool or was_bluff != actual_bluff:
                    raise ValueError("viewer packet truth metadata disagrees with cards")
                viewer_center.update(card.rank for card in cards)
            # A challenge exposes the exact packet identities, so remove only
            # those ranks from the caller's known-card inventory. An unresolved
            # packet exposes only q cards; subtract q from every rank to keep
            # the inventory a conservative lower bound.
            # Earlier pile packets may be visible NOW only because a later
            # packet was challenged. Do not give the historical caller those
            # identities before its call; only this packet's own resolution
            # made its identity public at this point in the replay.
            consumed = (Counter(card.rank for card in cards)
                        if called is True else Counter({rank: len(cards) for rank in Rank}))
            for known_rank, count in consumed.items():
                caller_knows_viewer_hand[known_rank] = max(
                    0, caller_knows_viewer_hand[known_rank] - count)
        else:
            if revealed:
                if any(not isinstance(card, Card) for card in cards):
                    raise ValueError("revealed opponent packet must expose cards")
                actual_bluff = any(card.rank != rank for card in cards)
                if type(was_bluff) is not bool or was_bluff != actual_bluff:
                    raise ValueError("revealed opponent truth metadata disagrees with cards")
                if actual_bluff:
                    reasons.append("opponent_revealed_false_packet")
            elif any(card is not None for card in cards) or was_bluff is not None:
                raise ValueError("unrevealed opponent packet exposes hidden truth")

            # A false revealed opponent packet contradicts this hypothesis.
            # Do not make later bounds depend on pretending it was truthful.
            if not reasons:
                hand_lower[rank] = max(0, hand_lower[rank] - len(cards))
                target_center[rank] += len(cards)

        # A voluntary challenge by the target is informative only after the
        # viewer's packet is publicly revealed false. Forced finals are ignored.
        if (player == viewer and response == "challenge" and not forced):
            if revealed and was_bluff is False:
                reasons.append("opponent_voluntarily_challenged_truth")
            elif revealed and was_bluff is True:
                if (type(claimant_before) is int and claimant_before >= len(cards)
                        and type(caller_before) is int and caller_before >= 0):
                    claimant_remaining = claimant_before - len(cards)
                    other_rank_proof_possible = sum(
                        count for known_rank, count in caller_known_before_packet.items()
                        if known_rank != rank) > claimant_remaining
                    if not other_rank_proof_possible and not reasons:
                        threshold = max(0, 5 - len(cards) - target_center[rank])
                        hand_lower[rank] = max(hand_lower[rank], threshold)

        # The center resolves on any challenge or pass. On a challenge the
        # target may take the pile; count only our own publicly known cards.
        if response == "challenge":
            if player == target and response_player == viewer and was_bluff is False:
                # The viewer challenged a truthful target packet and took the
                # pile. Under the truth hypothesis, every target card in this
                # center is now known to be in the viewer's hand.
                caller_knows_viewer_hand.update(target_center)
            elif response_player == target:
                if was_bluff is True:
                    # The claimant takes the pile. The target knows its own
                    # hypothesized truthful packets in it and the revealed
                    # viewer packet now held by the claimant.
                    caller_knows_viewer_hand.update(target_center)
                    caller_knows_viewer_hand.update(card.rank for card in cards)
                # If the target's own false claim lost, compatibility is
                # already false; its unknown pickup cannot safely add bounds.
            target_center.clear()
            viewer_center.clear()
        elif response == "pass":
            target_center.clear()
            viewer_center.clear()

    compatible = not reasons
    return {
        "conditional_bounds": ({rank: count for rank, count in hand_lower.items() if count}
                                if compatible else {}),
        "compatible": compatible,
        "incompatibility_reasons": tuple(dict.fromkeys(reasons)),
        "limitations": (
            "bounds depend on truthful opponent plays and challenge-only-when-provable behavior",
            "an incompatible hypothesis yields no usable bounds",
            "pile-transfer bounds use only revealed cards and conditional opponent-truth assumptions",
            "rank inference is skipped if known off-rank cards could explain the challenge",
            "rank inference requires player_hand_size_before and response_hand_size_before metadata",
            "forced final challenges provide no behavioral evidence",
            "bounds are not general card-certainty proofs",
        ),
    }


__all__ = ["honest_count_hypothesis_bounds"]
