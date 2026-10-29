"""Sampled public-Adaptive next-action forecast after a legal hypothetical play.

Worlds satisfy the focal player's current card constraints; frequencies are
not a history-conditioned posterior or a calibrated prediction for arbitrary
opponents.
"""
from collections import Counter
from copy import deepcopy
import random

from bots.base import public_action_for_viewer
from bots.round_candidate_response import _stage
from bots.round_certified_opening_world import sample_certified_opening_world
from bots.round_determinization import sample_root_world
from bots.round_opening_policy_forecast import exact_first_deal_policy_forecast
from bots.round_public_opponent_model import public_adaptive_from_history
from bots.round_policy import RoundAdaptiveBot
from bots.round_rollout import apply_choice, materialize_root
from bots.round_observation import round_player_view

FAMILIES = ("play", "pass", "challenge")
_PLAIN_CHOOSE = RoundAdaptiveBot.choose_action
_PLAIN_OBSERVE = RoundAdaptiveBot.observe_action


def _legal_response_category(response, view, hand):
    """Validate one response without executing its discarded continuation."""
    if not isinstance(response, dict):
        raise ValueError('public-Adaptive actor returned malformed response')
    category = response.get('action', response.get('kind'))
    if category in ('call', 'call_bluff', 'challenge'):
        category = 'challenge'
        valid = view.get('can_call_bluff') is True
    elif category == 'pass':
        valid = view.get('can_pass') is True
    elif category == 'play':
        cards = response.get('cards')
        valid = (view.get('can_play') is True and isinstance(cards, list)
            and 1 <= len(cards) <= 4 and all(card in hand for card in cards)
            and len(set(cards)) == len(cards)
            and response.get('rank') == view.get('active_rank'))
    else:
        valid = False
    if not valid:
        raise ValueError('public-Adaptive actor returned illegal response')
    return category
_VIEW_FIELDS = {"viewer", "player_id", "active_rank", "opponent_hand_size",
                "hand_size", "own_hand", "pile_size", "draw_pile_size",
                "round_number", "turn_number", "private_pickups", "actions",
                "last_action", "own_pile_cards", "can_play", "can_call_bluff",
                "can_pass"}


def _eligible_certified_opening(hand, view, prior_public_profile=None):
    """Return whether this is a supported fresh Adaptive opening observation.

    This predicate is deliberately separate from input validation. The caller
    validates the legal view and every hypothetical choice first; unsupported
    opening regimes fall back to the generic proposal, while malformed inputs
    continue to raise through the existing validation path.
    """
    if (not isinstance(view, dict) or len(hand) != 14
            or len(view.get("actions", ())) != 1
            or view.get("round_number") != 1 or view.get("turn_number") != 1
            or view.get("private_pickups") or view.get("own_pile_cards")
            or view.get("draw_pile_size") != 24):
        return False
    pending = view.get("last_action")
    if pending is None:
        return False
    quantity = len(getattr(pending, "cards_played", ()))
    if (quantity < 2 or view.get("opponent_hand_size", 0) + quantity != 14
            or any(action.was_bluff is not None
                   or any(card is not None for card in action.cards_played)
                   for action in (pending, view["actions"][0]))):
        return False
    opening_model = RoundAdaptiveBot(0)
    if prior_public_profile is not None:
        opening_model.restore_opponent_profile(deepcopy(prior_public_profile))
        if opening_model.to_dict() != prior_public_profile:
            raise ValueError("noncanonical public checkpoint")
    if any(opening_model._call_rate(n, n) < .35 for n in (1, 2, 3, 4)):
        return False
    # The exact declared Math/Adaptive opening law distinguishes zero-support
    # observations from a supported certificate without catching sampler
    # ValueErrors (which could otherwise conceal malformed inputs).
    law = exact_first_deal_policy_forecast(hand, view)["rank_quantity"]["Math"]
    return law["likelihood"] > 0


def forecast_public_adaptive_response(hand, view, packet, rank, *, seed=0,
                                      samples=16, prior_public_profile=None):
    """Forecast the sampled public-Adaptive actor's reply to our hypothetical play.

    ``prior_public_profile`` is the caller's reconstructed public-evidence
    checkpoint from before this deal. The actor receives only its sampled own
    hand and a legal view. Hidden sampled packet identities are never passed to
    its historical learner.
    """
    if type(seed) is not int or type(samples) is not int or not 1 <= samples <= 32:
        raise ValueError("invalid seed or sample budget")
    if not isinstance(view, dict) or set(view) - _VIEW_FIELDS:
        raise ValueError("forecast requires a legal player observation")
    _, _, viewer, staged, _ = _stage(hand, view, packet, rank)
    if staged is None:
        return {"forced": True, "counts": {"play": 0, "pass": 0, "challenge": 1},
                "probabilities": {"play": 0.0, "pass": 0.0, "challenge": 1.0},
                "proposal_is_history_posterior": False,
                "scope": "rules-forced final-play challenge"}

    rng = random.Random(seed)
    ambient_rng = random.getstate()
    counts = Counter()
    try:
        for _ in range(samples):
            world = sample_root_world(hand, view, rng)
            game = materialize_root(hand, view, world)
            events = apply_choice(game, {"action": "play", "cards": list(packet),
                                         "rank": rank})
            actor_id = 1 - viewer
            actor = public_adaptive_from_history(
                view["actions"], opponent_id=actor_id,
                seed=rng.randrange(2**32),
                prior_public_profile=prior_public_profile)
            for event in events:
                actor.observe_action(public_action_for_viewer(event, actor_id),
                                     game.hands[viewer].size())
            actor_view = round_player_view(game, actor_id)
            choice = actor.choose_action(list(game.hands[actor_id].cards), actor_view)
            category = choice.get("action", choice.get("kind"))
            if category in ("call", "call_bluff", "challenge"):
                category = "challenge"
            if category not in FAMILIES:
                raise ValueError("public-Adaptive actor returned unsupported action")
            apply_choice(game, choice)
            counts[category] += 1
    finally:
        random.setstate(ambient_rng)

    probabilities = {family: (counts[family] + 0.5) / (samples + 1.5)
                     for family in FAMILIES}
    return {"forced": False, "samples": samples,
            "counts": {family: counts[family] for family in FAMILIES},
            "probabilities": probabilities,
            "proposal_is_history_posterior": False,
            "scope": "current-constraint sampled public-Adaptive responses"}


def forecast_public_adaptive_responses(hand, view, choices, *, seed=0, samples=16,
                                       prior_public_profile=None, reuse_free_lead=False,
                                       condition_opening=False):
    """Batch forecasts while sharing sampled root worlds across choices."""
    ambient_rng = random.getstate()
    try:
        if type(seed) is not int or type(samples) is not int or not 1 <= samples <= 32:
            raise ValueError("invalid seed or sample budget")
        if not isinstance(view, dict) or set(view) - _VIEW_FIELDS:
            raise ValueError("forecast requires a legal player observation")
        if type(reuse_free_lead) is not bool:
            raise ValueError('reuse_free_lead must be boolean')
        if type(condition_opening) is not bool:
            raise ValueError('condition_opening must be boolean')
        if not isinstance(choices, (list, tuple)) or not 1 <= len(choices) <= 128:
            raise ValueError("choices must be a nonempty list or tuple of at most 128 entries")

        staged_choices = []
        for choice in choices:
            if not isinstance(choice, (list, tuple)) or len(choice) != 2:
                raise ValueError("each choice must be a (packet, rank) pair")
            packet, rank = choice
            _, _, viewer, staged, _ = _stage(hand, view, packet, rank)
            staged_choices.append((packet, rank, viewer, staged))

        use_certified_opening = (condition_opening
            and _eligible_certified_opening(hand, view, prior_public_profile))

        forced_result = {
            "forced": True,
            "counts": {"play": 0, "pass": 0, "challenge": 1},
            "probabilities": {"play": 0.0, "pass": 0.0, "challenge": 1.0},
            "proposal_is_history_posterior": False,
            "scope": "rules-forced final-play challenge",
        }
        results = [deepcopy(forced_result) if staged is None else None
                   for _, _, _, staged in staged_choices]
        if all(result is not None for result in results):
            return results

        rng = random.Random(seed)
        actor_id = 1 - staged_choices[0][2]
        baseline_actor = public_adaptive_from_history(
            view["actions"], opponent_id=actor_id, seed=0,
            prior_public_profile=deepcopy(prior_public_profile))
        counts_by_choice = [Counter() if result is None else None for result in results]
        # Only optimize the exact plain public actor, not extensions or injected
        # instance hooks. Preserve independent count lists and deduplication.
        plain_actor = (type(baseline_actor) is RoundAdaptiveBot
            and set(vars(baseline_actor)) == {'rng','player_id','last_decision_telemetry',
                '_resolved_action_objects','_call_evidence','_bluff_evidence'}
            and not baseline_actor.last_decision_telemetry)
        baseline_profile = baseline_actor.to_dict() if plain_actor else None
        reuse_view = (reuse_free_lead and plain_actor and view.get('active_rank') is None
            and view.get('last_action') is None and view.get('pile_size') == 0
            and RoundAdaptiveBot.choose_action is _PLAIN_CHOOSE
            and RoundAdaptiveBot.observe_action is _PLAIN_OBSERVE)

        for _ in range(samples):
            world = (sample_certified_opening_world(
                         hand, view, rng, prior_public_profile=prior_public_profile)
                     if use_certified_opening else sample_root_world(hand, view, rng))
            base_game = materialize_root(hand, view, world)
            # materialize_root overwrites hands/draw/pile after GameState's
            # constructor. Its separate fresh deck is dead scratch state;
            # a normal dealt engine also has an empty deck. No Card sharing.
            base_game.deck.cards.clear()
            actor_seed = rng.randrange(2**32)
            templates = {}
            for index, (packet, rank, viewer, staged) in enumerate(staged_choices):
                if staged is None:
                    continue
                if plain_actor:
                    actor = RoundAdaptiveBot(actor_seed)
                    actor.restore_opponent_profile(baseline_profile)
                    actor.player_id = baseline_actor.player_id
                    actor._resolved_action_objects = deepcopy(baseline_actor._resolved_action_objects)
                else:
                    actor = deepcopy(baseline_actor)
                    actor.rng.seed(actor_seed)
                if reuse_view:
                    quantity = len(packet)
                    if quantity not in templates:
                        game = deepcopy(base_game)
                        apply_choice(game, {'action': 'play', 'cards': list(packet), 'rank': rank})
                        templates[quantity] = round_player_view(game, actor_id)
                    # Views, including mutable Cards and both detached public
                    # latest-packet copies, remain independent per candidate.
                    actor_view = deepcopy(templates[quantity])
                    actor_view['active_rank'] = rank
                    latest = actor_view['actions'][-1]
                    pending = actor_view['last_action']
                    assert latest.player == viewer and pending.player == viewer
                    assert latest.was_bluff is None and pending.was_bluff is None
                    assert all(card is None for card in latest.cards_played + pending.cards_played)
                    latest.claimed_rank = rank
                    pending.claimed_rank = rank
                    actor.observe_action(pending, actor_view['opponent_hand_size'])
                    actor_hand = list(actor_view['own_hand'])
                    response = actor.choose_action(actor_hand, actor_view)
                    category = _legal_response_category(response, actor_view, actor_hand)
                    counts_by_choice[index][category] += 1
                    continue
                game = deepcopy(base_game)
                events = apply_choice(game, {"action": "play", "cards": list(packet),
                                             "rank": rank})
                for event in events:
                    actor.observe_action(public_action_for_viewer(event, actor_id),
                                         game.hands[viewer].size())
                actor_view = round_player_view(game, actor_id)
                response = actor.choose_action(list(game.hands[actor_id].cards), actor_view)
                category = response.get("action", response.get("kind"))
                if category in ("call", "call_bluff", "challenge"):
                    category = "challenge"
                if category not in FAMILIES:
                    raise ValueError("public-Adaptive actor returned unsupported action")
                apply_choice(game, response)
                counts_by_choice[index][category] += 1

        for index, result in enumerate(results):
            if result is not None:
                continue
            counts = counts_by_choice[index]
            probabilities = {family: (counts[family] + 0.5) / (samples + 1.5)
                             for family in FAMILIES}
            results[index] = {
                "forced": False, "samples": samples,
                "counts": {family: counts[family] for family in FAMILIES},
                "probabilities": probabilities,
                "proposal_is_history_posterior": False,
                "scope": ("certified fresh-opening-conditioned sampled public-Adaptive responses"
                    if use_certified_opening else
                    "current-constraint sampled public-Adaptive responses"
                    + (" (certified opening unsupported; generic fallback)"
                       if condition_opening else "")),
            }
        return results
    finally:
        random.setstate(ambient_rng)
