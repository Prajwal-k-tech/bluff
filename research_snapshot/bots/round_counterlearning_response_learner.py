"""Versioned four-family response learner with a public adaptive surrogate.

Internally the parent learner's ``unknown`` key is reused solely as a
compatibility slot for ``PublicAdaptive``. It does not mean unknown-human
belief and old profile schemas are never reinterpreted.
"""
from copy import deepcopy
import math

from bots.round_policy_response_learner import ConsistentPolicyResponseLearner
from bots.round_public_policy_response_likelihood import update_posterior, uniform_posterior
from game_v2 import RULESET_ID

FAMILIES = ("play", "pass", "challenge")
INPUT_NAMES = {"honest", "math", "random", "pickup_adaptive"}
EXTERNAL_NAMES = {"Honest": "honest", "Math": "math", "Random": "random",
                  "PublicAdaptive": "pickup_adaptive"}


class CounterlearningResponseLearner(ConsistentPolicyResponseLearner):
    """Consistent mixture update using a named public-history adaptive model."""

    model_id = "legal-policy-response-mixture-public-adaptive-v3"

    @staticmethod
    def _likelihoods(experts):
        if not isinstance(experts, dict) or set(experts) != INPUT_NAMES:
            raise ValueError("honest, math, random, and pickup_adaptive experts required")
        baseline = {name: deepcopy(experts[name]) for name in ("honest", "math", "random")}
        adaptive = deepcopy(experts["pickup_adaptive"])
        # Parent validates the unchanged three experts and builds the canonical
        # internal alias row. Replace only that row with the adaptive law.
        likelihoods = ConsistentPolicyResponseLearner._likelihoods(baseline)
        likelihoods["unknown"] = {"family": adaptive}
        # Reuse the same strict categorical validator for the fourth row.
        update_posterior(uniform_posterior(), likelihoods, "family", "play",
                         restart_share=0.0)
        for row in likelihoods.values():
            distribution = row["family"]
            if not math.isclose(sum(distribution.values()), 1.0,
                                rel_tol=0.0, abs_tol=1e-10):
                raise ValueError("expert family distributions must sum to one")
        if any(likelihoods["Random"]["family"][family] <= 0 for family in FAMILIES):
            raise ValueError("Random expert must retain full support")
        return likelihoods

    def to_dict(self):
        internal = dict(self.weights)
        external = {
            "Honest": internal["Honest"],
            "Math": internal["Math"],
            "Random": internal["Random"],
            "PublicAdaptive": internal["unknown"],
        }
        return {"model_id": self.model_id, "ruleset_id": RULESET_ID,
                "weights": external, "updates": self.updates}

    def restore(self, profile):
        expected_keys = {"model_id", "ruleset_id", "weights", "updates"}
        if (not isinstance(profile, dict) or set(profile) != expected_keys
                or profile.get("model_id") != self.model_id
                or profile.get("ruleset_id") != RULESET_ID
                or type(profile.get("updates")) is not int
                or profile["updates"] < 0):
            raise ValueError("unsupported public-adaptive response profile")
        external = profile.get("weights")
        if not isinstance(external, dict) or set(external) != set(EXTERNAL_NAMES):
            raise ValueError("profile must contain exactly the four named experts")
        internal = {"Honest": external["Honest"], "Math": external["Math"],
                    "Random": external["Random"],
                    "unknown": external["PublicAdaptive"]}
        # Validate the complete conversion before delegating; parent.restore is
        # transactional and accepts only the exact four canonical internal keys.
        neutral = {expert: {"family": {family: 1/3 for family in FAMILIES}}
                   for expert in internal}
        update_posterior(internal, neutral, "family", "play", restart_share=0.0)
        total = sum(internal.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("profile weights must sum to one")
        staged = {"model_id": self.model_id, "ruleset_id": RULESET_ID,
                  "weights": internal, "updates": profile["updates"]}
        super().restore(staged)


__all__ = ["CounterlearningResponseLearner"]
