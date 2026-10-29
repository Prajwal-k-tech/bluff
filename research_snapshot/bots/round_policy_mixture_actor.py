"""Frozen per-trajectory opponent policy mixture for research rollouts."""
from copy import deepcopy
import math

from bots.round_policy import _RoundPolicy, RoundHonestBot, RoundMathBot, RoundRandomBot


class RoundPolicyMixtureActor(_RoundPolicy):
    """Draw one opponent hypothesis per rollout and retain it throughout."""

    policy_name = "round_policy_mixture_actor_research"

    def __init__(self, weights, seed=None):
        super().__init__(seed)
        if (not isinstance(weights, dict) or set(weights) != {"Honest", "Math", "Random", "unknown"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                       for v in weights.values())
                or not math.isclose(sum(weights.values()), 1., abs_tol=1e-10)):
            raise ValueError("normalized four-policy weights required")
        self.weights = dict(weights)
        self._actors = {"Honest": RoundHonestBot(seed), "Math": RoundMathBot(seed),
                        "Random": RoundRandomBot(seed), "unknown": RoundRandomBot(seed)}
        self._selected = None

    def reset(self):
        super().reset()
        self._selected = None
        for actor in self._actors.values():
            actor.reset()

    def observe_action(self, action, opponent_hand_size):
        for actor in self._actors.values():
            actor.player_id = self.player_id
            actor.observe_action(action, opponent_hand_size)

    def choose_action(self, hand, state):
        self._set_viewer(state)
        if self._selected is None:
            names = tuple(self.weights)
            self._selected = self.rng.choices(names, [self.weights[n] for n in names])[0]
        actor = self._actors[self._selected]
        actor.player_id = self.player_id
        actor.rng = self.rng
        result = actor.choose_action(hand, state)
        self.last_decision_telemetry = {"policy": self.policy_name,
                                        "actor_hypothesis": self._selected}
        return result

    def to_dict(self):
        return {**super().to_dict(), "weights": deepcopy(self.weights)}

    def restore_opponent_profile(self, data):
        if not isinstance(data, dict) or set(data) != {"version", "policy", "ruleset_id", "weights"}:
            raise ValueError("unsupported mixture actor profile")
        super().restore_opponent_profile(data)
        staged = type(self)(data["weights"])
        self.weights = staged.weights
        self.reset()
