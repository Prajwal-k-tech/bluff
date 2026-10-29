"""Research forecast of candidate responses using legal-view policy experts.

Proposal deals satisfy current card constraints, not a full history posterior.
Expert probabilities are conditional on those proposals and declared policies.
"""
from collections import Counter
from copy import deepcopy
import random

from bots.round_candidate_response import _stage, forecast_candidate_response
from bots.round_determinization import sample_root_world
from bots.round_rollout import materialize_root, apply_choice
from bots.round_observation import round_player_view
from bots.base import public_action_for_viewer
from bots.round_policy import RoundHonestBot, RoundMathBot, RoundRandomBot

EXPERTS = {"honest": RoundHonestBot, "math": RoundMathBot, "random": RoundRandomBot}
FAMILIES = ("play", "pass", "challenge")


def forecast_policy_responses(hand, view, packet, rank, *, seed=0, samples=16,
                              hypothesis_consistent=False):
    """Return smoothed expert family probabilities after a hypothetical play."""
    if type(seed) is not int or type(samples) is not int or not 1 <= samples <= 32:
        raise ValueError("invalid seed or sample budget")
    if set(view).intersection({"hands", "opponent_hand", "game", "target", "truth"}):
        raise ValueError("forecast requires a legal player observation")
    _, _, viewer, staged, _ = _stage(hand, view, packet, rank)
    if staged is None:
        return {"forced": True, "probabilities": {
            name: {"play": 0., "pass": 0., "challenge": 1.} for name in EXPERTS}}
    rng = random.Random(seed)
    ambient = random.getstate()
    counts = {name: Counter() for name in EXPERTS}
    constructors = {"math": RoundMathBot} if hypothesis_consistent else EXPERTS
    try:
        for _ in range(samples):
            world = sample_root_world(hand, view, rng)
            root = materialize_root(hand, view, world)
            apply_choice(root, {"action": "play", "cards": list(packet), "rank": rank})
            actor = 1-viewer
            for name, constructor in constructors.items():
                game = deepcopy(root)
                policy = constructor(rng.randrange(2**32))
                policy.reset()
                policy.player_id = actor
                # Rebuild only legally visible historical evidence. Unknown
                # older own packets remain unknown in the root materializer.
                for action in game.actions:
                    policy.observe_action(public_action_for_viewer(action, actor),
                                          game.hands[viewer].size())
                actor_view = round_player_view(game, actor)
                choice = policy.choose_action(list(game.hands[actor].cards), actor_view)
                category = choice.get("action", choice.get("kind"))
                category = "challenge" if category in ("call", "call_bluff", "challenge") else category
                if category not in FAMILIES:
                    raise ValueError("policy returned unsupported family")
                # Engine acceptance is the action legality check.
                apply_choice(game, choice)
                counts[name][category] += 1
    finally:
        random.setstate(ambient)
    probabilities = {name: {f: (counts[name][f]+.5)/(samples+1.5)
                            for f in FAMILIES} for name in constructors}
    honest_supported = None
    if hypothesis_consistent:
        forecast = forecast_candidate_response(hand, view, packet, rank)["response_forecast"]
        rows = forecast["style_distributions"]["honest_max"]
        honest_supported = forecast["style_metadata"]["honest_max"]["allocation_count"] > 0
        honest = {f: 0. for f in FAMILIES}
        for row in rows:
            family = "challenge" if row["kind"] == "call_bluff" else row["kind"]
            honest[family] += row["probability"]
        probabilities["honest"] = honest if honest_supported else {f: 1/3 for f in FAMILIES}
        probabilities["random"] = {f: 1/3 for f in FAMILIES}
    return {"forced": False, "samples": samples,
            "proposal_is_history_posterior": False,
            "hypothesis_consistent": hypothesis_consistent,
            "honest_allocation_supported": honest_supported,
            "probabilities": probabilities}
