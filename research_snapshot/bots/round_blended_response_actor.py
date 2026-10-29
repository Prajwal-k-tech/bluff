"""Unregistered actor-conditional extension of a frozen response forecast.

Not an identity posterior or calibrated human policy. Physical play conditional
on choosing play is uniform quantity/subset; openings use Math. Learned family
forecasts are frozen, while the separate public-history surrogate may observe
simulated public evidence. Never supply an evaluator-owned opponent profile.
"""
from copy import deepcopy

from bots.round_blended_response_learner import BlendedResponseLearner
from bots.round_pickup_adaptive_bot import RoundPickupAdaptiveBot
from bots.round_policy import _RoundPolicy, RoundMathBot, RoundHonestBot
from bots.round_response_family_actor import RoundResponseFamilyActor
from game_v2 import RULESET_ID


def _validate_view(hand, state):
    if (not isinstance(state, dict)
            or set(state).intersection({"game", "hands", "opponent_hand", "truth", "target"})
            or state.get("own_hand") != list(hand)
            or type(state.get("player_id")) is not int
            or state["player_id"] not in (0, 1)
            or state.get("viewer") != state["player_id"]):
        raise ValueError("actor requires its own legal hand and observation")


class RoundBlendedResponseActor(_RoundPolicy):
    policy_name = "round_frozen_blended_response_actor_research_v1"
    requires_complete_private_history_for_search = False
    response_model_class = BlendedResponseLearner

    def __init__(self, response_profile, *, inferred_public_profile=None,
                 inferred_public_actor=None, seed=None):
        super().__init__(seed)
        self.model = self.response_model_class()
        self.model.restore(deepcopy(response_profile))
        self._adaptive = RoundPickupAdaptiveBot(seed=0)
        if inferred_public_actor is not None:
            if type(inferred_public_actor) is not RoundPickupAdaptiveBot or inferred_public_profile is not None:
                raise ValueError("one reconstructed public actor or pre-deal profile required")
            # Preserve deal-local consumed IDs as well as counts. Restoring a
            # current end-profile would replay and double-count old evidence.
            self._adaptive = deepcopy(inferred_public_actor)
        if inferred_public_profile is not None:
            self._adaptive.restore_opponent_profile(deepcopy(inferred_public_profile))
        self._math = RoundMathBot(seed=0)

    def observe_action(self, action, opponent_hand_size):
        self._adaptive.player_id = self.player_id
        self._adaptive.observe_action(action, opponent_hand_size)

    def observe_player_view(self, state):
        _validate_view(state.get("own_hand", []), state)
        self._adaptive.observe_player_view(state)

    def reset(self):
        super().reset()
        self._adaptive.reset()
        self._math.reset()

    def family_forecast(self, hand, state):
        """Pure forecast: expert clones consume actor-legal inputs, not root truth."""
        _validate_view(hand, state)
        if state.get("active_rank") is None:
            raise ValueError("response forecast requires an active round")
        legality = _RoundPolicy()
        legality.player_id = state["player_id"]
        legal = legality._legal_families(list(hand), state)
        names = ["challenge" if name == "call_bluff" else name for name in legal]
        experts = {}
        for name, actor in (("honest", RoundHonestBot(seed=0)),
                            ("math", RoundMathBot(seed=0)),
                            ("pickup_adaptive", deepcopy(self._adaptive))):
            actor.player_id = state["player_id"]
            choice = actor.choose_action(list(hand), deepcopy(state))
            family = "challenge" if choice["action"] == "call_bluff" else choice["action"]
            if family not in names:
                raise ValueError("expert returned an unavailable family")
            experts[name] = {f: float(f == family) for f in ("play", "pass", "challenge")}
        experts["random"] = {f: (1/len(names) if f in names else 0.)
                             for f in ("play", "pass", "challenge")}
        probabilities = self.model.predict(experts)
        total = sum(probabilities[f] for f in names)
        return {"probabilities": {f: probabilities[f]/total for f in names},
                "experts": experts,
                "scope": "actor-conditional extension, not a calibrated identity model"}

    def choose_action(self, hand, state):
        # Validate even opening inputs before choosing; normal views contain
        # detached own_hand/viewer. No sampled opponent truth is accepted.
        _validate_view(hand, state)
        self._set_viewer(state)
        if state.get("active_rank") is None:
            return self._opening_choice(hand, state)
        legal = self._legal_families(list(hand), state)
        if RoundResponseFamilyActor._truthful_finish(hand, state, legal):
            return self._play(list(hand), state["active_rank"])
        forecast = self.family_forecast(hand, state)
        probabilities = forecast["probabilities"]
        family = RoundResponseFamilyActor._sample_family(self, probabilities)
        if family == "play":
            # ponytail: explicit uniform conditional law; learned quantities /
            # truth-conditioned packets require separately validated models.
            quantity = self._sample_packet_quantity(len(hand))
            choice = self._play(self.rng.sample(list(hand), quantity), state["active_rank"])
        else:
            choice = self._simple("call_bluff" if family == "challenge" else family)
        self.last_decision_telemetry = {"policy": self.policy_name,
            "response_family_probabilities": probabilities, "frozen_response_model": True,
            "scope": forecast["scope"]}
        return choice

    def _sample_packet_quantity(self, hand_size):
        return self.rng.choice(range(1, min(4, hand_size)+1))

    def _opening_choice(self, hand, state):
        self._math.player_id = self.player_id
        return self._math.choose_action(list(hand), state)

    def to_dict(self):
        return {"version": 1, "policy": self.policy_name, "ruleset_id": RULESET_ID,
                "response_model": deepcopy(self.model.to_dict()),
                "inferred_public_model": deepcopy(self._adaptive.to_dict())}

    def restore_opponent_profile(self, data):
        if (not isinstance(data, dict) or set(data) != set(self.to_dict())
                or type(data.get("version")) is not int or data["version"] != 1
                or data.get("policy") != self.policy_name or data.get("ruleset_id") != RULESET_ID):
            raise ValueError("unsupported frozen blended actor profile")
        staged = type(self)(data["response_model"],
                            inferred_public_profile=data["inferred_public_model"], seed=0)
        self.model, self._adaptive = staged.model, staged._adaptive
        self.reset()
