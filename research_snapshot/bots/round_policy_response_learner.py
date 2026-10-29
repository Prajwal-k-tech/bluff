"""Research-only prequential mixture of legal-view policy response forecasts."""
from copy import deepcopy
import math

from bots.round_public_policy_response_likelihood import update_posterior, uniform_posterior
from game_v2 import RULESET_ID

NAMES = {"Honest": "honest", "Math": "math", "Random": "random"}
FAMILIES = ("play", "pass", "challenge")


class PolicyResponseLearner:
    model_id = "legal-policy-response-mixture-v1"

    def __init__(self):
        self.weights = uniform_posterior()
        self.updates = 0
        self.begin_deal()


    def begin_deal(self):
        self._pending = {}
        self._seen = set()

    @staticmethod
    def _likelihoods(experts):
        if not isinstance(experts, dict) or set(experts) != set(NAMES.values()):
            raise ValueError("three policy experts required")
        likelihoods = {name: {"family": deepcopy(experts[key])} for name, key in NAMES.items()}
        likelihoods["unknown"] = {"family": {f: 1/3 for f in FAMILIES}}
        # Reuse strict categorical probability validation before any mutation.
        update_posterior(uniform_posterior(), likelihoods, "family", "play", restart_share=0.)
        if any(not math.isclose(sum(row["family"].values()), 1., abs_tol=1e-10)
               for row in likelihoods.values()):
            raise ValueError("expert distributions must be normalized")
        return likelihoods

    def predict(self, experts):
        likelihoods = self._likelihoods(experts)
        return {f: sum((.98*self.weights[e]+.005)*likelihoods[e]["family"][f]
                       for e in self.weights) for f in FAMILIES}

    def forecast(self, index, experts):
        if type(index) is not int or index < 0 or index in self._seen:
            raise ValueError("invalid or closed action index")
        if self._pending and index not in self._pending:
            raise ValueError("an earlier response is still pending")
        likelihoods = self._likelihoods(experts)
        if index in self._pending:
            if self._pending[index][0] != likelihoods:
                raise ValueError("pending forecast changed")
            return dict(self._pending[index][1])
        prediction = self.predict(experts)
        self._pending[index] = (likelihoods, prediction, dict(self.weights))
        return dict(prediction)

    def observe(self, index, family, *, forced=False):
        if type(index) is not int or family not in FAMILIES or type(forced) is not bool:
            raise ValueError("invalid response metadata")
        if forced and family != "challenge":
            raise ValueError("forced response must be a challenge")
        if index not in self._pending:
            return False
        likelihoods, _, prior = self._pending.pop(index)
        self._seen.add(index)
        if forced:
            return False
        self.weights = update_posterior(prior, likelihoods, "family", family, restart_share=.02)
        self.updates += 1
        return True

    def discard_forecast(self, index):
        if type(index) is not int or index < 0:
            raise ValueError("invalid action index")
        return self._pending.pop(index, None) is not None

    def to_dict(self):
        return {"model_id": self.model_id, "ruleset_id": RULESET_ID,
                "weights": dict(self.weights), "updates": self.updates}

    def restore(self, profile):
        if (not isinstance(profile, dict) or set(profile) != {"model_id", "ruleset_id", "weights", "updates"}
                or profile["model_id"] != self.model_id or profile["ruleset_id"] != RULESET_ID
                or type(profile["updates"]) is not int or profile["updates"] < 0):
            raise ValueError("unsupported policy response profile")
        weights = profile["weights"]
        neutral = {e: {"family": {f: 1/3 for f in FAMILIES}} for e in uniform_posterior()}
        update_posterior(weights, neutral, "family", "play", restart_share=0.)
        if not math.isclose(sum(weights.values()), 1., abs_tol=1e-10):
            raise ValueError("weights must be normalized")
        self.weights = dict(weights)
        self.updates = profile["updates"]
        self.begin_deal()


class ConsistentPolicyResponseLearner(PolicyResponseLearner):
    model_id = "legal-policy-response-mixture-consistent-v2"
