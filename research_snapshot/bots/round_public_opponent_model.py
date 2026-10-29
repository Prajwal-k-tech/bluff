"""Research reconstruction of adaptive actors from legal observed history.

Not a posterior over arbitrary humans, and not a reconstruction of private
arbitrary private learning or unknown prior-game memory. Own-packet pickup
labels can be inferred when the opponent publicly collects that round's pile.
Known inferred evidence may persist.
Callers supply legal observations,
never an opponent runtime profile or evaluator-owned hidden state.
"""
from bots.round_policy import RoundAdaptiveBot
from bots.round_policy import _context
from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
from cards import Card, Rank


def public_adaptive_from_history(actions, *, opponent_id, seed=None,
                                 prior_public_profile=None):
    """Replay public response/reveal evidence from the opposite perspective.

    Existing observe_action ignores unrevealed truth even when the observing
    player knows their own packet. Consequently replay does not falsely teach
    the opponent that private label. Deal-local action IDs deduplicate updates.
    Optional prior_public_profile must be the caller's previously reconstructed
    public evidence, not an actual opponent profile. It is the checkpoint BEFORE
    this deal; replaying this deal on its own final checkpoint would double count.
    """
    if type(opponent_id) is not int or opponent_id not in (0, 1):
        raise ValueError("two-player opponent_id required")
    actor = RoundAdaptiveBot(seed)
    if prior_public_profile is not None:
        actor.restore_opponent_profile(prior_public_profile)
    actor.player_id = opponent_id
    for action in actions:
        actor.observe_action(action, 0)  # Hand size is unused by this learner.
    return actor


def pickup_adaptive_from_history(actions, *, observer_id, seed=None,
                                 prior_inferred_profile=None):
    """Infer labels opponent learned on collecting OUR known earlier packets.

    Complete ordered legal history for observer_id required. Never accepts the
    opponent's pickup channel. Prior is caller's inferred PRE-deal checkpoint.
    This reconstructs PickupAdaptive counts, not Center's private-hand forecasts.
    """
    if type(observer_id) is not int or observer_id not in (0, 1):
        raise ValueError("two-player observer_id required")
    actor = RoundPickupAdaptiveBot(seed)
    if prior_inferred_profile is not None:
        actor.restore_opponent_profile(prior_inferred_profile)
    actor.player_id = 1-observer_id
    rounds = {}
    for index, action in enumerate(actions):
        if (type(action.action_index) is not int or action.action_index != index
            or type(action.player) is not int or action.player not in (0, 1)
            or type(action.round_number) is not int or action.round_number < 1):
            raise ValueError("complete ordered legal history required")
        rounds.setdefault(action.round_number, []).append(action)
        actor.observe_action(action, 0)
    for packets in rounds.values():
        latest = packets[-1]
        if latest.bluff_called is not True:
            continue
        if (latest.response_kind != "challenge" or latest.response_was_call is not True
            or type(latest.was_bluff) is not bool
            or type(latest.caller_was_right) is not bool
            or latest.caller_was_right != latest.was_bluff
            or latest.response_player != 1-latest.player):
            raise ValueError("inconsistent public challenge resolution")
        collector = latest.player if latest.was_bluff else 1-latest.player
        if collector != actor.player_id:
            continue
        for action in packets:
            key = ("action", action.action_index)
            if (action.player != observer_id or key in actor._resolved_action_objects
                or action.response_was_forced is not False):
                continue
            cards = action.cards_played
            if (not 1 <= len(cards) <= 4 or not isinstance(action.claimed_rank, Rank)
                or any(not isinstance(c, Card) for c in cards)):
                raise ValueError("observer's own exact packet required")
            bluff = any(card.rank != action.claimed_rank for card in cards)
            actor._add(actor._bluff_evidence,
                _context(len(cards), action.pile_size_before+len(cards)), bluff)
            actor._resolved_action_objects.add(key)
    return actor
